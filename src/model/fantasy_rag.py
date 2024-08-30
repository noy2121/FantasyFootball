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


class DataFrameUtils:
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
        self.rag_data = {}
        self.cached_club_data = {}
        self.cached_player_data = {}
        self.dataframes = load_dataframes(self.csvs_dir)

        self.encoder = SentenceEncoder(self.embedding_model_name)
        self.dataframe_util = DataFrameUtils()
        self.stats_processor = StatsProcessor(self.dataframes['players'], self.dataframes['clubs'])
        self.generate_jsons = cfg.generate_jsons
        self.jsons_out_dir = cfg.estimation_data_dir

    def prepare_rag_data(self):
        players_df, games_df, events_df, clubs_df, lineups_df = self._get_dataframes()

        for df in [games_df, events_df, lineups_df]:
            df['date'] = pd.to_datetime(df['date'])

        seasons = sorted(set(games_df['date'].dt.strftime('%Y')))
        for season in seasons:
            if season in ['2017', '2024']:  # ignore 2017 and 2025 seasons
                continue
            self.rag_data[season] = {}
            self._process_season_data(season, players_df, clubs_df, games_df, events_df, lineups_df)

    def _get_dataframes(self):
        return (self.dataframes['players'], self.dataframes['games'], self.dataframes['events'],
                self.dataframes['clubs'], self.dataframes['lineups'])

    def _process_season_data(self, season: str, players_df: pd.DataFrame, clubs_df: pd.DataFrame,
                             games_df: pd.DataFrame, events_df: pd.DataFrame, lineups_df: pd.DataFrame):
        season_start = datetime(int(season), 8, 10)
        season_end = datetime(int(season) + 1, 6, 20)
        decision_points = self._generate_decision_points(season_start, season_end)

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
                tqdm(decision_points[1:], file=sys.stdout, colour='WHITE', desc=f'Process {season} Data')):
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

    def _generate_decision_points(self, start: datetime, end: datetime) -> List[datetime]:
        return [start + timedelta(days=i * 7) for i in range((end - start).days // 7)]

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
        instance = cls.__new__(cls)
        instance.encoder = SentenceTransformer(os.path.join(input_dir, 'embedding_model'))
        with open(os.path.join(input_dir, 'seasons.txt'), 'r') as f:
            seasons = [line.strip() for line in f]

        instance.rag_data = {}
        instance.indices = {}
        instance.cached_club_data = {}
        instance.cached_player_data = {}

        for season in seasons:
            instance.rag_data[season] = {}
            instance.indices[season] = {}
            dirpath = os.path.join(input_dir, f'rag_dataset_{season}')
            for entry_type in ["clubs", "players"]:
                # load dataset and FAISS index
                instance.rag_data[season][entry_type] = Dataset.load_from_disk(f'{dirpath}/{entry_type}')
                instance.indices[season][entry_type] = faiss.read_index(f'{dirpath}/rag_index_{entry_type}.faiss')

            # cache club and player data to avoid repetitions
            instance.cached_club_data[season] = instance.rag_data[season]['clubs']['text']
            instance.cached_player_data[season] = instance.rag_data[season]['players']['text']

        return instance

    def retrieve_relevant_info(self, teams: List[str], date: str, season: str, k: int = 50) -> Dict[str, List[str]]:
        if season not in self.indices:
            raise ValueError(f"No data available for season {season}")

        query_date = datetime.strptime(date, '%Y-%m-%d')
        decision_points = np.array([datetime.strptime(d, '%Y-%m-%d') for d in set(self.rag_data[season]['clubs']['date'])])
        decision_points = np.sort(decision_points)

        # find nearest decision point
        idx = np.searchsorted(decision_points, query_date, side='left')
        if idx >= len(decision_points):
            idx = len(decision_points) - 1
        elif idx > 0 and query_date - decision_points[idx - 1] < decision_points[idx] - query_date:
            idx -= 1

        nearest_decision_point = decision_points[idx].strftime('%Y-%m-%d')
        clubs_valid_indices = np.where(np.array(self.rag_data[season]['clubs']['date']) == nearest_decision_point)[0]
        players_valid_indices = np.where(np.array(self.rag_data[season]['players']['date']) == nearest_decision_point)[0]

        relevant_info = {"teams": [], "players": []}
        for team in teams:

            team_query_embedding = self.encoder.encode([f"{team} team performance {season}"]).astype('float32')
            player_query_embedding = self.encoder.encode([f"{team} player stats {season}"]).astype('float32')

            # Perform team-level search
            _, team_indices = self.indices[season]['clubs'].search(team_query_embedding, k)
            team_indices = np.intersect1d(team_indices[0], clubs_valid_indices, assume_unique=True)
            team_info = [self.get_cached_club_data(season, i) for i in team_indices[:k]]
            relevant_info["teams"].extend(team_info)

            # Perform player-level search
            _, player_indices = self.indices[season]['players'].search(player_query_embedding, k * 10)
            player_indices = np.intersect1d(player_indices[0], players_valid_indices, assume_unique=True)

            # Pre-filter player text data to avoid unnecessary loops
            filtered_player_info = []
            seen_players = set()

            for i in player_indices:
                player_text = self.get_cached_player_data(season, i)
                player_lines = player_text.split('\n')
                if len(player_lines) < 2:
                    continue

                player_name = player_lines[0].split(': ')[-1]
                player_team = player_lines[2].split(': ')[-1]

                if player_name not in seen_players and player_team == team:
                    filtered_player_info.append(player_text)
                    seen_players.add(player_name)

                if len(filtered_player_info) >= k:
                    break

            relevant_info["players"].extend(filtered_player_info)

        return relevant_info


if __name__ == '__main__':
    # Usage example
    out_dir = os.path.join(ROOT_DIR, 'data/rag')
    data_dir = os.path.join(ROOT_DIR, 'data/csvs')
    rag_system = SeasonSpecificRAG(data_dir)
    rag_system.prepare_rag_data()
    rag_system.build_indices()
    rag_system.save(out_dir)

    # # Later, in your main application
    # loaded_rag = SeasonSpecificRAG.load('path/to/save/rag/data')
    #
    # query = "Top scorers in the Premier League"
    # season = "2022/2023"
    # relevant_info = loaded_rag.retrieve_relevant_info(query, season)
    # print(f"Relevant information for '{query}' in season {season}:")
    # for info in relevant_info:
    #     print(info)
    # print("\n")
