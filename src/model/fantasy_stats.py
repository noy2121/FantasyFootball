from typing import Dict, List, Tuple, Union
import json
import pandas as pd


class DataStatsCache:

    def __init__(self, data_dir: str):

        self.players_cache, self.clubs_cache = self.load_stats(data_dir)

    def load_stats(self, data_dir: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:

        with open(f'{data_dir}/players.json', 'r') as f:
            players = json.load(f)

        with open(f'{data_dir}/clubs.json', 'r') as f:
            clubs = json.load(f)

        return players, clubs

    def get(self, date: str, player_name: str) -> Tuple[Dict[str, any], Dict[str, any]]:

        player_stats = self.players_cache[date][player_name]
        club_name = player_stats['club_name']
        club_stats = self.clubs_cache[date][club_name]

        return player_stats, club_stats


