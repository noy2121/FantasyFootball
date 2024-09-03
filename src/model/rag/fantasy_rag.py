import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
from typing import List, Dict
from functools import lru_cache
from collections import defaultdict
from datetime import datetime, timedelta

import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from ...utils.utils import load_dataframes, get_club_id_by_club_name
from .rag_utility_classes import DataFilterUtils, StatsProcessor, SentenceEncoder
from ..metrics.fantasy_metrics import last_5_games_avg_score, seasonal_avg_score
from ...system_prompts import player_entry_format, club_entry_format


class SeasonSpecificRAG:
    # Class variables for entry formats
    club_entry_format = club_entry_format
    player_entry_format = player_entry_format

    def __init__(self, cfg: DictConfig, device: str = 'cpu'):

        self.device = device
        self.data_dir = cfg.rag_dir
        self.csvs_dir = cfg.csvs_dir
        self.embedding_model_name = cfg.embedding_model_name
        self.generate_jsons = cfg.generate_jsons
        self.jsons_out_dir = cfg.estimation_data_dir

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

    def prepare_rag_data(self):
        players_df, games_df, events_df, clubs_df, lineups_df = self._get_dataframes()

        for df in [games_df, events_df, lineups_df]:
            df['date'] = pd.to_datetime(df['date'])

        self.seasons = sorted(set(games_df['date'].dt.strftime('%Y')))

        for season in self.seasons:
            if season in ['2017', '2024']:  # ignore 2017 and 2025 seasons
                continue
            self.rag_data[season] = defaultdict(dict)
            self.indices[season] = {}
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

            # update stats
            self.stats_processor.update_club_stats(clubs_stats, curr_games)
            self.stats_processor.update_player_stats(players_stats, curr_events, curr_lineups)

            # create entries
            clubs_entries = self._prepare_club_entries(clubs_df, clubs_stats)
            players_entries = self._prepare_player_entries(players_df, players_stats, clubs_stats)

            prev_dec_date = decision_date

            # TODO: remove evaluation jsons?
            # this is for the jsons data
            if season == '2023':
                player_infer_data[str(decision_date.date())] = players_stats
                club_infer_data[str(decision_date.date())] = clubs_stats

            self.rag_data[season][decision_date] = {
                'clubs': Dataset.from_dict({'text': [entry for entry in clubs_entries]}),
                'players': Dataset.from_dict({'text': [entry for entry in players_entries]})
            }

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
        ) for club_id in clubs_df['club_id'] for stats in [club_stats[club_id]]]

    def _prepare_player_entries(self, players_df: pd.DataFrame, player_stats: Dict[int, Dict], clubs_stats: Dict[int, Dict]) -> List[str]:
        entries = []
        for pid in players_df['player_id']:
            for stats in [player_stats[pid]]:
                cid = get_club_id_by_club_name(stats['club_name'])
                num_games = clubs_stats[cid]['seasonal']['games']
                entries.append(self.player_entry_format.format(
                    stats['name'],
                    stats['position'],
                    stats['club_name'],
                    stats['cost'],
                    last_5_games_avg_score(stats),
                    seasonal_avg_score(stats, num_games)
                ))

        return entries

    def build_indices(self, batch_size=64):
        print('Building RAG indices...')
        for season, dps in self.rag_data.items():
            for decision_point, datasets in dps.items():
                for entry_type in ['clubs', 'players']:
                    texts = datasets[entry_type]['text']
                    embeddings = self.encoder.encode(texts, season, entry_type, batch_size)
                    index = faiss.IndexFlatL2(embeddings.shape[1])
                    index.add(embeddings.astype('float32'))
                    self.indices[season][decision_point][entry_type] = index
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

    @staticmethod
    def parse_player_entry(entry):
        parts = entry.split(', ')
        name = parts[0].split(': ')[1]
        club = parts[2].split(': ')[1]
        last_5_score = float(parts[4].split(': ')[1])
        return {'name': name, 'club': club, 'last_5_score': last_5_score, 'full_entry': entry}

    def retrieve_relevant_info(self, teams_batch: List[List[str]], dates_batch: List[str],
                               seasons_batch: List[str], k: int = 30) -> List[Dict[str, List[str]]]:

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
        for q_idx, (teams, season, nearest_dp) in enumerate(zip(teams_batch, seasons_batch, nearest_dps)):

            # if season is of the format YYYY/YY instead of YYYY
            if '/' in season:
                season = season.split('/')[0]

            # handle clubs info
            teams_set = set(teams)
            _, teams_idxs = self.indices[season][nearest_dp]['clubs'].search(teams_embeddings_batch[batch_start:batch_start + len(teams)], k)

            for t_idxs in teams_idxs:
                for i in sorted(t_idxs):
                    if i not in cached_club_data:
                        cached_club_data[i] = self.get_cached_club_data(season, i)
                    res = cached_club_data[i]
                    team_name = res.split('\n')[0].split(': ')[1]
                    if team_name in teams_set:
                        relevant_info[q_idx]["teams"].append(res)

            # handle players info
            _, players_idxs = self.indices[season][nearest_dp]['players'].search(player_embeddings_batch[batch_start:batch_start + len(teams)], k * 5)

            players_by_team = {team: [] for team in teams_set}
            for p_idxs in players_idxs:
                for i in sorted(p_idxs):
                    if i not in cached_player_data:
                        cached_player_data[i] = self.get_cached_player_data(season, nearest_dp, i)
                    player_info = cached_player_data[i]

                    player_entry = self.parse_player_entry(player_info)
                    if player_entry['club'] in teams_set:
                        players_by_team[player_entry['club']].append(player_entry)

            # sort players by Last 5 games average score and select the top 5 players from each team
            top_players = []
            for team, players in players_by_team.items():
                team_top_players = sorted(players, key=lambda x: x['last_5_score'], reverse=True)[:5]
                top_players.extend(team_top_players)

            relevant_info[q_idx]['players'] = [p['full_entry'] for p in top_players]

        return relevant_info
