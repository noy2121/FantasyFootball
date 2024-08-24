import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys

import faiss
import bisect
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import List, Dict
from datasets import Dataset
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer

from system_prompts import player_entry_format, club_entry_format
from utils.utils import load_dataframes, ROOT_DIR, get_hftoken


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
                'seasonal': {'win': 0, 'lose': 0, 'tie': 0},
                'last_5_cl': [],
                'last_5_dl': []
            },
            axis=1
        ).to_dict()

        return club_stats

    def initialize_player_stats(self, season: str) -> Dict[int, Dict]:

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

        club_name = self.clubs_df.loc[self.clubs_df['club_id'] == club_id, 'club_name']

    @staticmethod
    def update_club_stats(club_stats: Dict[int, Dict], games_df: pd.DataFrame):

        for _, game in games_df.iterrows():
            home_id, away_id = game['home_club_id'], game['away_club_id']
            home_goals, away_goals = game['home_club_goals'], game['away_club_goals']
            is_cl = game['competition_type'] == 'champions_league'

            if home_id not in club_stats and away_id not in club_stats:
                raise ValueError(f"Neither home club {home_id} nor away club {away_id} found in club_stats")

            for cid, is_home in [(home_id, True), (away_id, False)]:
                if cid not in club_stats:
                    continue

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
    def update_player_stats(player_stats: Dict[int, Dict], events_df: pd.DataFrame, games_df: pd.DataFrame,
                            lineups_df: pd.DataFrame):

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
                        'goals': np.sum(game_events[:, 2] == 'Goals'),
                        'assists': np.sum(game_events[:, 2] == 'Assists'),
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
        self.num_processes = max(1, cpu_count() - 1)

    def encode(self, texts, season, batch_size=64):
        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        inputs = [(self.model, batch) for batch in batches]
        with Pool(self.num_processes) as pool:
            results = list(
                pool.starmap(self.encode_batch, tqdm(inputs,
                                                     file=sys.stdout,
                                                     total=len(inputs),
                                                     colour='WHITE',
                                                     desc=f'Encoding {season} Data'))
            )

        return np.vstack(results)

    @staticmethod
    def encode_batch(model, batch):
        return model.encode(batch)

    @staticmethod
    def _load_embedding_model(model_name: str) -> SentenceTransformer:
        token_path = os.path.join(ROOT_DIR, 'data/keys/huggingface_token.txt')
        hf_token = get_hftoken(token_path)
        return SentenceTransformer(model_name, token=hf_token)


class SeasonSpecificRAG:
    # Class variables for entry formats
    club_entry_format = club_entry_format
    player_entry_format = player_entry_format

    def __init__(self, data_dir: str, embedding_model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self.data_dir = data_dir
        self.indices = {}
        self.rag_data = {}
        self.dataframes = load_dataframes(data_dir)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.encoder = SentenceEncoder(embedding_model_name)

        self.dataframe_util = DataFrameUtils()
        self.stats_processor = StatsProcessor(self.dataframes['players'], self.dataframes['clubs'])

    def prepare_rag_data(self):
        players_df, games_df, events_df, clubs_df, lineups_df = self._get_dataframes()

        for df in [games_df, events_df, lineups_df]:
            df['date'] = pd.to_datetime(df['date'])

        seasons = sorted(set(games_df['date'].dt.strftime('%Y')))
        for season in seasons:
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
        player_stats = self.stats_processor.initialize_player_stats(season)

        rag_entries = []
        prev_dec_date = season_start
        for i, decision_date in enumerate(
                tqdm(decision_points[1:], file=sys.stdout, colour='WHITE', desc=f'Process {season} Data')):
            curr_games = self.dataframe_util.filter_by_range_date(season_games, prev_dec_date, decision_date)
            curr_events = self.dataframe_util.filter_by_range_date(season_events, prev_dec_date, decision_date)
            curr_lineups = self.dataframe_util.filter_by_range_date(season_lineups, prev_dec_date, decision_date)

            self.stats_processor.update_club_stats(clubs_stats, curr_games)
            self.stats_processor.update_player_stats(player_stats, curr_events, curr_games, curr_lineups)

            rag_entries.extend(self._prepare_club_entries(clubs_df, clubs_stats))
            rag_entries.extend(self._prepare_player_entries(players_df, player_stats))

            prev_dec_date = decision_date

        assert len(rag_entries) == len(decision_points[1:]) * (len(clubs_df) + len(players_df)), \
            f'Number of rag_enries is different than number of date!!!'

        self.rag_data[season] = Dataset.from_dict({
            "text": rag_entries,
            "date": [d.strftime('%Y-%m-%d') for d in decision_points[1:]] * (len(clubs_df) + len(players_df))
        })

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

    def build_indices(self, batch_size=64, device='cpu'):
        print('Building RAG indices...')
        for season, dataset in self.rag_data.items():
            texts = dataset['text']
            embeddings = self.encoder.encode(texts, season, batch_size)

            # TODO: add gpu option
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.astype('float32'))
            self.indices[season] = index
            print(f"Index built for season {season}")

    def save(self, output_dir: str):
        print('Saving RAG dataset...')
        os.makedirs(output_dir, exist_ok=True)
        for season, dataset in self.rag_data.items():
            dataset.save_to_disk(os.path.join(output_dir, f'rag_dataset_{season}'))
            faiss.write_index(self.indices[season], os.path.join(output_dir, f'rag_index_{season}.faiss'))
        self.embedding_model.save(os.path.join(output_dir, 'embedding_model'))
        with open(os.path.join(output_dir, 'seasons.txt'), 'w') as f:
            f.write('\n'.join(self.rag_data.keys()))

    @classmethod
    def load(cls, input_dir: str):
        instance = cls.__new__(cls)
        instance.embedding_model = SentenceTransformer(os.path.join(input_dir, 'embedding_model'))
        with open(os.path.join(input_dir, 'seasons.txt'), 'r') as f:
            seasons = [line.strip() for line in f]
        instance.rag_data = {season: Dataset.load_from_disk(os.path.join(input_dir, f'rag_dataset_{season}')) for season
                             in seasons}
        instance.indices = {season: faiss.read_index(os.path.join(input_dir, f'rag_index_{season}.faiss')) for season in
                            seasons}
        return instance

    def retrieve_relevant_info(self, query: str, date: str, season: str, k: int = 5) -> List[str]:
        if season not in self.indices:
            raise ValueError(f"No data available for season {season}")

        query_date = datetime.strptime(date, '%Y-%m-%d')
        decision_points = [datetime.strptime(d, '%Y-%m-%d') for d in set(self.rag_data[season]['date'])]
        decision_points.sort()

        # find nearest decision point
        idx = bisect.bisect_left(decision_points, query_date)
        if idx == len(decision_points):
            idx -= 1
        elif idx > 0 and query_date - decision_points[idx - 1] < decision_points[idx] - query_date:
            idx -= 1

        # filter the dataset to include only entries up to the nearest decision point
        nearest_decision_point = decision_points[idx].strftime('%Y-%m-%d')
        valid_indices = [i for i, d in enumerate(self.rag_data[season]['date']) if d <= nearest_decision_point]

        query_embedding = self.embedding_model.encode([query])
        _, indices = self.indices[season].search(query_embedding.astype('float32'), k)

        # filter the results to include only valid indices
        filtered_indices = [i for i in indices[0] if i in valid_indices]

        return [self.rag_data[season]['text'][i] for i in filtered_indices[:k]]


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
