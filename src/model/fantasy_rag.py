import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
from typing import List, Dict
from functools import lru_cache
from datetime import datetime, timedelta

import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer

from ..system_prompts import player_entry_format, club_entry_format
from ..utils.utils import load_dataframes, ROOT_DIR, get_hftoken


class DataFilterUtils:
    @staticmethod
    def filter_and_sort_by_date(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        return df.loc[(df['date'] >= start) & (df['date'] < end)].sort_values('date')

    @staticmethod
    def filter_by_range_date(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        return df.loc[(df['date'] > start) & (df['date'] <= end)]


class StatsProcessor:

    def __init__(self, players_df, clubs_df):

        self.players_df = players_df
        self.clubs_df = clubs_df

    def initialize_clubs_stats(self) -> Dict[int, Dict]:
        club_stats = self.clubs_df.set_index('club_id').apply(
            lambda row: {
                'name': row['club_name'],
                'cl_titles': row['number_of_champions_league_titles'],
                'seasonal': {'win': 0, 'lose': 0, 'tie': 0, 'games': 0},
                'last_5_cl': [],
                'last_5_dl': []
            },
            axis=1
        ).to_dict()

        return club_stats

    def initialize_players_stats(self, season: str) -> Dict[int, Dict]:
        season = f'{season}/{int(season[-2:]) + 1}'
        player_stats = self.players_df.set_index('player_id').apply(
            lambda row: {
                'name': row['player_name'],
                'position': row['position'],
                'club_id': row[f'{season}_club_id'],
                'club_name': self._get_club_name(row[f'{season}_club_id']),
                'cost': row[f'{season}_cost'],
                'seasonal': {'goals': 0, 'assists': 0, 'lineups': 0},
                'last_5': []
            },
            axis=1
        ).to_dict()

        return player_stats

    def _get_club_name(self, club_id):
        if np.isnan(club_id) or club_id not in self.clubs_df['club_id'].values:
            return ''
        club_name = self.clubs_df.loc[self.clubs_df['club_id'] == club_id, 'club_name'].iloc[0]
        return club_name

    @staticmethod
    def update_club_stats(club_stats: Dict[int, Dict], games_df: pd.DataFrame):

        for _, game in games_df.iterrows():
            home_id, away_id = game['home_club_id'], game['away_club_id']
            home_goals, away_goals = game['home_club_goals'], game['away_club_goals']
            is_cl = game['competition_type'] == 'champions_league'
            is_dl = 'domestic' in game['competition_type']

            assert (home_id in club_stats) or (away_id in club_stats), \
                f"Neither home club {home_id} nor away club {away_id} found in club_stats for game id {game['game_id']}"

            if not is_dl and not is_cl:
                continue

            for cid, is_home in [(home_id, True), (away_id, False)]:
                if cid not in club_stats:
                    continue

                club_stats[cid]['seasonal']['games'] += 1
                result = 'win' if (is_home and home_goals > away_goals) or (
                            not is_home and away_goals > home_goals) else \
                    'lose' if (is_home and home_goals < away_goals) or (
                                not is_home and away_goals < home_goals) else 'tie'

                club_stats[cid]['seasonal'][result] += 1
                target_list = 'last_5_cl' if is_cl else 'last_5_dl'
                club_stats[cid][target_list].append(result)
                if len(club_stats[cid][target_list]) > 5:
                    club_stats[cid][target_list].pop(0)

    @staticmethod
    def update_player_stats(player_stats: Dict[int, Dict], events_df: pd.DataFrame, lineups_df: pd.DataFrame):

        events_array = events_df[['player_id', 'game_id', 'event_type', 'date']].to_numpy()
        lineups_array = lineups_df[lineups_df['type'] == 'starting_lineup'][['player_id', 'game_id']].to_numpy()

        player_events = {}
        player_lineups = {}

        for pid, gid, etype, date in events_array:
            if pid not in player_events:
                player_events[pid] = {'goals': 0, 'assists': 0, 'games': set(), 'last_5': []}
            player_events[pid]['goals'] += (etype == 'Goals')
            player_events[pid]['assists'] += (etype == 'Assists')
            player_events[pid]['games'].add((gid, date))

        for pid, gid in lineups_array:
            if pid not in player_lineups:
                player_lineups[pid] = set()
            player_lineups[pid].add(gid)

        # Update player stats
        for pid, stats in player_stats.items():
            if pid in player_events:
                events = player_events[pid]
                stats['seasonal']['goals'] = events['goals']
                stats['seasonal']['assists'] = events['assists']
                stats['seasonal']['lineups'] = len(player_lineups.get(pid, set()))

                # Sort games by date and get the last 5
                last_5_games = sorted(events['games'], key=lambda x: x[1])[-5:]
                existing_gids = [item['game_id'] for item in stats['last_5']]
                for gid, _ in last_5_games:
                    if gid in existing_gids:
                        continue
                    game_events = events_array[(events_array[:, 0] == pid) & (events_array[:, 1] == gid)]
                    curr_perf = {
                        'game_id': gid,
                        'goals': int(np.sum(game_events[:, 2] == 'Goals')),
                        'assists': int(np.sum(game_events[:, 2] == 'Assists')),
                        'lineups': int(gid in player_lineups.get(pid, set()))
                    }
                    stats['last_5'].append(curr_perf)
                    existing_gids.append(gid)
                    if len(stats['last_5']) > 5:
                        removed_game = stats['last_5'].pop(0)
                        existing_gids.remove(removed_game['game_id'])


class SentenceEncoder:
    def __init__(self, model_name: str):
        self.model = self._load_embedding_model(model_name)

    def encode(self, texts, season, etype, batch_size=64):
        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        results = []
        for batch in tqdm(batches, file=sys.stdout, total=len(batches), colour='WHITE',
                          desc=f'Encoding {season} {etype.capitalize()} Data'):
            results.append(self.encode_batch(batch))

        return np.vstack(results)

    def encode_batch(self, batch):
        return self.model.encode(batch)

    def save(self, outf):
        self.model.save(outf)

    @staticmethod
    def _load_embedding_model(model_name: str) -> SentenceTransformer:
        token_path = os.path.join(ROOT_DIR, 'data/keys/huggingface_token.txt')
        hf_token = get_hftoken(token_path)
        return SentenceTransformer(model_name, token=hf_token)


class SeasonSpecificRAG:
    # Class variables for entry formats
    club_entry_format = club_entry_format
    player_entry_format = player_entry_format

    def __init__(self, cfg: DictConfig, device: str = 'cpu'):

        self.data_dir = cfg.rag_dir
        self.csvs_dir = cfg.csvs_dir
        self.embedding_model_name = cfg.embedding_model_name
        self.device = device

        self.indices = {}
        self.seasons = []
        self.rag_data = {}
        self.cached_club_data = {}
        self.cached_player_data = {}
        self.season_start_date = '08-10'
        self.season_end_date = '06-20'
        self.decision_points = self._generate_decision_points()

        self.dataframes = load_dataframes(self.csvs_dir)

        self.encoder = SentenceEncoder(self.embedding_model_name)
        self.dataframe_util = DataFilterUtils()
        self.stats_processor = StatsProcessor(self.dataframes['players'], self.dataframes['clubs'])
        self.generate_jsons = cfg.generate_jsons
        self.jsons_out_dir = cfg.estimation_data_dir

    def prepare_rag_data(self):
        players_df, games_df, events_df, clubs_df, lineups_df = self._get_dataframes()

        for df in [games_df, events_df, lineups_df]:
            df['date'] = pd.to_datetime(df['date'])

        self.seasons = sorted(set(games_df['date'].dt.strftime('%Y')))

        for season in self.seasons:
            if season in ['2017', '2024']:  # ignore 2017 and 2025 seasons
                continue
            self.rag_data[season] = {}
            self._process_season_data(season, players_df, clubs_df, games_df, events_df, lineups_df)

    def _get_dataframes(self):
        return (self.dataframes['players'], self.dataframes['games'], self.dataframes['events'],
                self.dataframes['clubs'], self.dataframes['lineups'])

    def _process_season_data(self, season: str, players_df: pd.DataFrame, clubs_df: pd.DataFrame,
                             games_df: pd.DataFrame, events_df: pd.DataFrame, lineups_df: pd.DataFrame):

        season_start = datetime.strptime(f'{season}-{self.season_start_date}', '%Y-%m-%d')
        season_end = datetime.strptime(f'{int(season) + 1}-{self.season_end_date}', '%Y-%m-%d')

        season_games = self.dataframe_util.filter_and_sort_by_date(games_df, season_start, season_end)
        season_events = self.dataframe_util.filter_and_sort_by_date(events_df, season_start, season_end)
        season_lineups = self.dataframe_util.filter_and_sort_by_date(lineups_df, season_start, season_end)

        clubs_stats = self.stats_processor.initialize_clubs_stats()
        players_stats = self.stats_processor.initialize_players_stats(season)

        clubs_entries = []
        players_entries = []
        seen_entries = set()
        player_infer_data = {}
        club_infer_data = {}
        prev_dec_date = season_start
        for i, decision_date in enumerate(
                tqdm(self.decision_points[1:], file=sys.stdout, colour='WHITE', desc=f'Process {season} Data')):
            if int(decision_date.split('-')[0]) < 7:
                decision_date = datetime.strptime(f'{int(season) + 1}-{decision_date}', '%Y-%m-%d')
            else:
                decision_date = datetime.strptime(f'{season}-{decision_date}', '%Y-%m-%d')
            curr_games = self.dataframe_util.filter_by_range_date(season_games, prev_dec_date, decision_date)
            curr_events = self.dataframe_util.filter_by_range_date(season_events, prev_dec_date, decision_date)
            curr_lineups = self.dataframe_util.filter_by_range_date(season_lineups, prev_dec_date, decision_date)

            self.stats_processor.update_club_stats(clubs_stats, curr_games)
            self.stats_processor.update_player_stats(players_stats, curr_events, curr_lineups)

            # Deduplicate club entries
            for club_entry in self._prepare_club_entries(clubs_df, clubs_stats):
                entry_key = (decision_date, club_entry)
                if entry_key not in seen_entries:
                    clubs_entries.append((club_entry, decision_date.strftime('%Y-%m-%d')))
                    seen_entries.add(entry_key)

            # Deduplicate player entries
            for player_entry in self._prepare_player_entries(players_df, players_stats):
                entry_key = (decision_date, player_entry)
                if entry_key not in seen_entries:
                    players_entries.append((player_entry, decision_date.strftime('%Y-%m-%d')))
                    seen_entries.add(entry_key)

            prev_dec_date = decision_date

            # this is for the jsons data
            if season == '2023':
                player_infer_data[str(decision_date.date())] = players_stats
                club_infer_data[str(decision_date.date())] = clubs_stats

        self.rag_data[season]['clubs'] = Dataset.from_dict({
            "text": [entry[0] for entry in clubs_entries],
            "date": [entry[1] for entry in clubs_entries]
        })
        self.rag_data[season]['players'] = Dataset.from_dict({
            "text": [entry[0] for entry in players_entries],
            "date": [entry[1] for entry in players_entries]
        })

        if self.generate_jsons and season == '2023':
            player_infer_data = self.restructure_dict(player_infer_data)
            club_infer_data = self.restructure_dict(club_infer_data)
            self.save_jsons(player_infer_data, club_infer_data)

    @staticmethod
    def restructure_dict(original_dict):
        restructured_dict = {}

        for date, elems in original_dict.items():
            restructured_dict[date] = {}
            for eid, el_info in elems.items():
                el_name = el_info['name']
                restructured_dict[date][el_name] = el_info.copy()
                del restructured_dict[date][el_name]['name']

        return restructured_dict

    def save_jsons(self, player_stats, clubs_stats):
        with open(f'{self.jsons_out_dir}/players.json', 'w') as f:
            json.dump(player_stats, f, indent=True)
        with open(f'{self.jsons_out_dir}/clubs.json', 'w') as f:
            json.dump(clubs_stats, f, indent=True)

    def _generate_decision_points(self) -> List[str]:
        start = datetime.strptime(self.season_start_date, '%m-%d')
        end = datetime.strptime(self.season_end_date, '%m-%d')
        total_weeks = (end - start).days // 7 + 53
        return [(start + timedelta(weeks=i)).strftime('%m-%d') for i in range(total_weeks)]

    def _prepare_club_entries(self, clubs_df: pd.DataFrame, club_stats: Dict[int, Dict]) -> List[str]:
        return [self.club_entry_format.format(
            stats['name'],
            stats['last_5_cl'],
            stats['last_5_dl'],
            stats['seasonal']['win'], stats['seasonal']['lose'], stats['seasonal']['tie'],
            stats['cl_titles']
        ) for club_id in clubs_df['club_id'] for stats in [club_stats[club_id]]]

    def _prepare_player_entries(self, players_df: pd.DataFrame, player_stats: Dict[int, Dict]) -> List[str]:
        return [self.player_entry_format.format(
            stats['name'],
            stats['position'],
            stats['club_name'],
            stats['cost'],
            *self._calculate_last_5_stats(stats['last_5']),
            stats['seasonal']['goals'], stats['seasonal']['assists'], stats['seasonal']['lineups']
        ) for player_id in players_df['player_id'] for stats in [player_stats[player_id]]]

    @staticmethod
    def _calculate_last_5_stats(last_5: List[Dict]) -> np.ndarray:
        if not last_5:
            return np.array([0, 0, 0])
        arr = np.array([(d['goals'], d['assists'], d['lineups']) for d in last_5])
        return arr.sum(axis=0)

    def build_indices(self, batch_size=64):
        print('Building RAG indices...')
        for season, datasets in self.rag_data.items():
            self.indices[season] = {}
            for entry_type, dataset in datasets.items():
                texts = dataset['text']
                embeddings = self.encoder.encode(texts, season, entry_type, batch_size)

                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings.astype('float32'))
                self.indices[season][entry_type] = index
            print(f"Index built for season {season}")

    def save(self):
        print('Saving RAG dataset...')
        os.makedirs(self.data_dir, exist_ok=True)
        for season, datasets in self.rag_data.items():
            for entry_type, dataset in datasets.items():
                out_dir = os.path.join(self.data_dir, f'rag_dataset_{season}')
                dataset.save_to_disk(f'{out_dir}/{entry_type}')
                faiss.write_index(self.indices[season][entry_type], f'{out_dir}/rag_index_{entry_type}.faiss')
        self.encoder.save(os.path.join(self.data_dir, 'embedding_model'))
        with open(os.path.join(self.data_dir, 'seasons.txt'), 'w') as f:
            f.write('\n'.join(self.rag_data.keys()))

    @lru_cache(maxsize=None)
    def get_cached_club_data(self, season, index):
        return self.cached_club_data[season][index]

    @lru_cache(maxsize=None)
    def get_cached_player_data(self, season, index):
        return self.cached_player_data[season][index]

    @classmethod
    def load(cls, input_dir: str):
        print('\nLoad RAG Dataset')
        instance = cls.__new__(cls)
        instance.encoder = SentenceTransformer(os.path.join(input_dir, 'embedding_model'))
        with open(os.path.join(input_dir, 'seasons.txt'), 'r') as f:
            seasons = [line.strip() for line in f]

        instance.seasons = seasons
        instance.rag_data = {}
        instance.indices = {}
        instance.cached_club_data = {}
        instance.cached_player_data = {}
        instance.season_start_date = '08-10'
        instance.season_end_date = '06-20'
        instance.decision_points = instance._generate_decision_points()

        for season in seasons:
            instance.rag_data[season] = {}
            instance.indices[season] = {}
            dirpath = os.path.join(input_dir, f'rag_dataset_{season}')
            for entry_type in tqdm(["clubs", "players"], file=sys.stdout, colour='WHITE',
                                   desc=f'\tLoad RAG data for {season=}'):
                # load dataset and FAISS index
                instance.rag_data[season][entry_type] = Dataset.load_from_disk(f'{dirpath}/{entry_type}')
                instance.indices[season][entry_type] = faiss.read_index(f'{dirpath}/rag_index_{entry_type}.faiss')

            # cache club and player data to avoid repetitions
            instance.cached_club_data[season] = instance.rag_data[season]['clubs']['text']
            instance.cached_player_data[season] = instance.rag_data[season]['players']['text']

        return instance

    def find_nearest_decision_points(self, dates_batch: List[str]):

        query_dates = pd.to_datetime(dates_batch, format='%Y-%m-%d')
        query_dates_md = query_dates.strftime('%m-%d')
        decision_points_arr = pd.to_datetime(self.decision_points, format='%m-%d').strftime('%m-%d').to_numpy()
        decision_points_arr = np.sort(decision_points_arr)

        indices = np.searchsorted(decision_points_arr, query_dates_md, side='right') - 1
        indices = np.clip(indices, 0, len(decision_points_arr) - 1)

        nearest_dps = decision_points_arr[indices]
        nearest_dps = np.array(
            [datetime.strptime(f'{d.year}-{str(md)}', '%Y-%m-%d') for d, md in zip(query_dates, nearest_dps)])
        nearest_dps = [x.strftime('%Y-%m-%d') for x in nearest_dps]

        assert len(dates_batch) == len(nearest_dps), "Output order doesn't match input order"

        return nearest_dps

    def retrieve_relevant_info(self, teams_batch: List[List[str]], dates_batch: List[str],
                               seasons_batch: List[str], k: int = 200) -> List[Dict[str, List[str]]]:

        relevant_info = [{"teams": [], "players": []} for _ in range(len(teams_batch))]

        # get the nearest decision points for each sample
        nearest_dps = self.find_nearest_decision_points(dates_batch)

        # create team and player queries
        # the input is flattened because the encoder expect list of sequences
        team_queries_batch = [
            f"{team}" for teams, season in zip(teams_batch, seasons_batch) for team in teams
        ]
        player_queries_batch = list(set(
            f"{team} player" for teams, season in zip(teams_batch, seasons_batch) for team in teams
        ))

        teams_embeddings_batch = self.encoder.encode(team_queries_batch).astype('float32')
        player_embeddings_batch = self.encoder.encode(player_queries_batch).astype('float32')
        cached_club_data = {}
        cached_player_data = {}
        batch_start = 0
        for q_idx, (teams, season) in enumerate(zip(teams_batch, seasons_batch)):

            # if season is of the format YYYY/YY instead of YYYY
            if '/' in season:
                season = season.split('/')[0]

            # handle clubs info
            teams_set = set(teams)
            _, teams_idxs = self.indices[season]['clubs'].search(teams_embeddings_batch[batch_start:batch_start + len(teams)], k)
            teams_valid_idxs = np.where(np.array(self.rag_data[season]['clubs']['date']) == nearest_dps[q_idx])[0]

            for t_idxs in teams_idxs:
                t_valid_idxs = set(t_idxs) & set(teams_valid_idxs)
                for i in sorted(t_valid_idxs)[:k]:
                    if i not in cached_club_data:
                        cached_club_data[i] = self.get_cached_club_data(season, i)
                    res = cached_club_data[i]
                    team_name = res.split('\n')[0].split(': ')[1]
                    if team_name in teams_set:
                        relevant_info[q_idx]["teams"].append(res)

            # handle players info
            _, players_idxs = self.indices[season]['players'].search(player_embeddings_batch[batch_start:batch_start + len(teams)], k * 5)
            players_valid_idxs = np.where(np.array(self.rag_data[season]['players']['date']) == nearest_dps[q_idx])[0]

            for p_idxs in players_idxs:
                p_valid_idxs = set(p_idxs) & set(players_valid_idxs)
                filtered_players_info = []
                seen_players = set()

                for i in p_valid_idxs:
                    if i not in cached_player_data:
                        cached_player_data[i] = self.get_cached_player_data(season, i)
                    p_text = cached_player_data[i]
                    p_lines = p_text.split('\n')
                    if len(p_lines) < 2:
                        continue

                    p_name = p_lines[0].split(': ')[-1]
                    p_club = p_lines[2].split(': ')[-1]
                    if p_club not in teams:
                        continue
                    if p_name not in seen_players:
                        filtered_players_info.append(p_text)
                        seen_players.add(p_name)

                    if len(filtered_players_info) >= k:
                        break
                relevant_info[q_idx]['players'].extend(filtered_players_info)

        return relevant_info
