import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from tqdm import tqdm
from typing import Dict
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ...utils.utils import ROOT_DIR, get_hftoken


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

    def encode(self, texts, season, week, etype, batch_size=64):
        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        results = []
        for batch in batches:
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