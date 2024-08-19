import os
import faiss
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from src.system_prompts import player_entry_format, club_entry_format
from src.utils.utils import load_dataframes, ROOT_DIR, get_hftoken


import os
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import faiss
from datasets import Dataset
from sentence_transformers import SentenceTransformer


class DataFrameUtils:
    @staticmethod
    def filter_and_sort_by_date(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        return df.loc[(df['date'] >= start) & (df['date'] < end)].sort_values('date')

    @staticmethod
    def filter_by_range_date(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        return df.loc[(df['date'] > start) & (df['date'] <= end)]


class StatsProcessor:
    @staticmethod
    def initialize_clubs_stats(clubs_df: pd.DataFrame) -> Dict[int, Dict]:
        return {row['club_id']: {
            'name': row['club_name'],
            'cl_titles': row['number_of_champions_league_titles'],
            'seasonal': defaultdict(int),
            'last_5_cl': [],
            'last_5_dl': []
        } for _, row in clubs_df.iterrows()}

    @staticmethod
    def initialize_player_stats(player_df: pd.DataFrame, clubs_df: pd.DataFrame) -> Dict[int, Dict]:
        return {row['player_id']: {
            'name': row['player_name'],
            'position': row['position'],
            'club_id': row['club_id'],
            'club_name': clubs_df.loc[clubs_df['club_id'] == row['club_id'], 'club_name'].iloc[0],
            'seasonal': defaultdict(int),
            'last_5': []
        } for _, row in player_df.iterrows()}

    @staticmethod
    def update_club_stats(club_stats: Dict[int, Dict], games_df: pd.DataFrame):
        for _, game in games_df.iterrows():
            home_id, away_id = game['home_club_id'], game['away_club_id']
            home_goals, away_goals = game['home_club_goals'], game['away_club_goals']
            is_cl = game['competition_type'] == 'champions_league'

            for club_id, is_home in [(home_id, True), (away_id, False)]:
                result = 'win' if (is_home and home_goals > away_goals) or (not is_home and away_goals > home_goals) else \
                         'lose' if (is_home and home_goals < away_goals) or (not is_home and away_goals < home_goals) else 'tie'

                club_stats[club_id]['seasonal'][result] += 1
                target_list = 'last_5_cl' if is_cl else 'last_5_dl'
                club_stats[club_id][target_list].append(result)
                if len(club_stats[club_id][target_list]) > 5:
                    club_stats[club_id][target_list].pop(0)

    @staticmethod
    def update_player_stats(player_stats: Dict[int, Dict], events_df: pd.DataFrame, games_df: pd.DataFrame, lineups_df: pd.DataFrame):
        for _, erow in events_df.iterrows():
            player_id = erow['player_id']
            if erow['event_type'] == 'Goals':
                player_stats[player_id]['seasonal']['goals'] += 1
            elif erow['event_type'] == 'Assists':
                player_stats[player_id]['seasonal']['assists'] += 1

        for _, lrow in lineups_df.iterrows():
            if lrow['type'] == 'starting_lineup':
                player_stats[lrow['player_id']]['seasonal']['lineups'] += 1

        for _, game in games_df.iterrows():
            game_events = events_df[events_df['game_id'] == game['game_id']]
            game_lineups = lineups_df[lineups_df['game_id'] == game['game_id']]
            for club_id in [game['home_club_id'], game['away_club_id']]:
                for player_id, stats in player_stats.items():
                    if stats['club_id'] == club_id:
                        player_events = game_events[game_events['player_id'] == player_id]
                        curr_perf = {
                            'goals': player_events[player_events['event_type'] == 'Goals'].shape[0],
                            'assists': player_events[player_events['event_type'] == 'Assists'].shape[0],
                            'lineups': int(game_lineups[(game_lineups['player_id'] == player_id) & (game_lineups['type'] == 'starting_lineup')].shape[0] > 0)
                        }
                        stats['last_5'].append(curr_perf)
                        if len(stats['last_5']) > 5:
                            stats['last_5'].pop(0)


class SeasonSpecificRAG:
    # Class variables for entry formats
    club_entry_format = club_entry_format
    player_entry_format = player_entry_format

    def __init__(self, data_dir: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.indices = {}
        self.rag_data = {}
        self.dataframes = load_dataframes(data_dir)
        self.embedding_model = self._load_embedding_model(embedding_model)

        self.dataframe_util = DataFrameUtils()
        self.stats_processor = StatsProcessor()

    def _load_embedding_model(self, model_name: str) -> SentenceTransformer:
        token_path = os.path.join(os.path.dirname(__file__), 'data/keys/huggingface_token.txt')
        hf_token = get_hftoken(token_path)
        return SentenceTransformer(model_name, token=hf_token)

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

        season_start = datetime(int(season), 8, 1)
        season_end = datetime(int(season) + 1, 8, 1)
        decision_points = self._generate_decision_points(season_start, season_end)

        season_games = self.dataframe_util.filter_and_sort_by_date(games_df, season_start, season_end)
        season_events = self.dataframe_util.filter_and_sort_by_date(events_df, season_start, season_end)
        season_lineups = self.dataframe_util.filter_and_sort_by_date(lineups_df, season_start, season_end)

        clubs_stats = self.stats_processor.initialize_clubs_stats(clubs_df)
        player_stats = self.stats_processor.initialize_player_stats(players_df, clubs_df)

        rag_entries = []
        prev_dec_date = season_start
        for decision_date in decision_points:
            curr_games = self.dataframe_util.filter_by_range_date(season_games, prev_dec_date, decision_date)
            curr_events = self.dataframe_util.filter_by_range_date(season_events, prev_dec_date, decision_date)
            curr_lineups = self.dataframe_util.filter_by_range_date(season_lineups, prev_dec_date, decision_date)

            self.stats_processor.update_club_stats(clubs_stats, curr_games)
            self.stats_processor.update_player_stats(player_stats, curr_events, curr_games, curr_lineups)

            rag_entries.extend(self._prepare_club_entries(clubs_df, clubs_stats))
            rag_entries.extend(self._prepare_player_entries(players_df, player_stats))

            prev_dec_date = decision_date

        self.rag_data[season] = Dataset.from_dict({
            "text": rag_entries,
            "date": [d.strftime('%Y-%m-%d') for d in decision_points] * (len(clubs_df) + len(players_df))
        })

    def _generate_decision_points(self, start: datetime, end: datetime) -> List[datetime]:
        return [start + timedelta(days=i*7) for i in range((end - start).days // 7)]

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
            *self._calculate_last_5_stats(stats['last_5']),
            stats['seasonal']['goals'], stats['seasonal']['assists'], stats['seasonal']['lineups']
        ) for player_id in players_df['player_id'] for stats in [player_stats[player_id]]]

    @staticmethod
    def _calculate_last_5_stats(last_5):
        return tuple(sum(game[stat] for game in last_5) for stat in ['goals', 'assists', 'lineups'])

    def build_indices(self):
        for season, dataset in self.rag_data.items():
            embeddings = self.embedding_model.encode(dataset['text'])
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.astype('float32'))
            self.indices[season] = index

    def save(self, output_dir: str):
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
        instance.rag_data = {season: Dataset.load_from_disk(os.path.join(input_dir, f'rag_dataset_{season}')) for season in seasons}
        instance.indices = {season: faiss.read_index(os.path.join(input_dir, f'rag_index_{season}.faiss')) for season in seasons}
        return instance

    def retrieve_relevant_info(self, query: str, season: str, k: int = 5) -> List[str]:
        if season not in self.indices:
            raise ValueError(f"No data available for season {season}")
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.indices[season].search(query_embedding.astype('float32'), k)
        return [self.rag_data[season]['text'][i] for i in indices[0]]


    def build_indices(self):
        for season, dataset in self.rag_data.items():
            embeddings = self.embedding_model.encode(dataset['text'])
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.astype('float32'))
            self.indices[season] = index

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        # Save the RAG datasets
        for season, dataset in self.rag_data.items():
            dataset.save_to_disk(os.path.join(output_dir, f'rag_dataset_{season}'))

        # Save the FAISS indices
        for season, index in self.indices.items():
            faiss.write_index(index, os.path.join(output_dir, f'rag_index_{season}.faiss'))

        # Save the embedding model
        self.embedding_model.save(os.path.join(output_dir, 'embedding_model'))

        # Save the list of seasons
        with open(os.path.join(output_dir, 'seasons.txt'), 'w') as f:
            for season in self.rag_data.keys():
                f.write(f"{season}\n")

    @classmethod
    def load(cls, input_dir: str):
        instance = cls()

        # Load the embedding model
        instance.embedding_model = SentenceTransformer(os.path.join(input_dir, 'embedding_model'))

        # Load the list of seasons
        with open(os.path.join(input_dir, 'seasons.txt'), 'r') as f:
            seasons = [line.strip() for line in f]

        # Load the RAG datasets and FAISS indices
        for season in seasons:
            instance.rag_data[season] = Dataset.load_from_disk(os.path.join(input_dir, f'rag_dataset_{season}'))
            instance.indices[season] = faiss.read_index(os.path.join(input_dir, f'rag_index_{season}.faiss'))

        return instance

    def retrieve_relevant_info(self, query: str, season: str, k: int = 5) -> List[str]:
        if season not in self.indices:
            raise ValueError(f"No data available for season {season}")

        query_embedding = self.embedding_model.encode([query])
        _, indices = self.indices[season].search(query_embedding.astype('float32'), k)
        return [self.rag_data[season]['text'][i] for i in indices[0]]



# Usage example
data_dir = os.path.join(ROOT_DIR, 'data/preprocessed/train/csvs')
rag_system = SeasonSpecificRAG(data_dir)
rag_system.prepare_rag_data()
rag_system.build_indices()
rag_system.save()

# Later, in your main application
loaded_rag = SeasonSpecificRAG.load('path/to/save/rag/data')

query = "Top scorers in the Premier League"
season = "2022/2023"
relevant_info = loaded_rag.retrieve_relevant_info(query, season)
print(f"Relevant information for '{query}' in season {season}:")
for info in relevant_info:
    print(info)
print("\n")


# # Using in FantasyFootballRAGChunked
# class FantasyFootballRAGChunked:
#     def __init__(self, model, tokenizer, rag_system: SeasonSpecificRAG):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.rag_system = rag_system
#
#     def generate_team_chunked(self, prompt: str, season: str) -> str:
#         relevant_info = self.rag_system.retrieve_relevant_info(prompt, season)
#         augmented_prompt = f"{prompt}\nSeason: {season}\n\nRelevant Information:\n" + "\n".join(relevant_info)
#
#         inputs = self.tokenizer(augmented_prompt, return_tensors="pt", max_length=1024, truncation=True)
#         outputs = self.model.generate(**inputs, max_length=2048)
#         return self.tokenizer.decode(outputs[0])
#
#
# # Usage in main script
# rag_system = SeasonSpecificRAG.load('path/to/save/rag/data')
# fantasy_rag = FantasyFootballRAGChunked(fine_tuned_model, tokenizer, rag_system)
#
# prompt = "Create a fantasy team with the best performers"
# season = "2022/2023"
# response = fantasy_rag.generate_team_chunked(prompt, season)
# print(response)
