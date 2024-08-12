import os
from typing import Dict, Set
from datetime import datetime

import pandas as pd

from players_wrangler import create_players_df
from games_wrangler import create_games_df
from events_wrangler import create_events_df
from wrangler_utils import get_relevant_club_ids
from src.utils.utils import get_root_dir, set_random_seed, save_dataframes

ROOT_DIR = get_root_dir()


def prepare_dataframes(raw_dfs: Dict[str, pd.DataFrame], club_ids: Set[int], start_year: int, curr_year: int, out_dir: str):

    # create players df
    players_df = create_players_df(raw_dfs, club_ids, start_year, curr_year)
    games_df = create_games_df(raw_dfs['games'], club_ids, start_year, curr_year)
    events_df = create_events_df(raw_dfs['game_events'], games_df, start_year)

    print("Save DataFrames...")
    filepaths = ['players_data', 'games_data', 'events_data']
    filepaths = [f'{out_dir}/{fp}.csv' for fp in filepaths]
    save_dataframes([players_df, games_df, events_df], filepaths)


def wrangler():

    # TODO: define config file
    data_dir = os.path.join(ROOT_DIR, 'data/raw_csvs')
    dataframes = {}
    for filename in os.listdir(data_dir):
        if filename == 'clubs.csv':
            continue
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            dataframes[filename[:-4]] = pd.read_csv(filepath)

    out_dir = os.path.join(ROOT_DIR, 'data/csvs')
    dataframes['clubs'] = pd.read_csv(f'{out_dir}/clubs_data.csv')
    relevant_club_ids = get_relevant_club_ids(dataframes['clubs'])

    start_year = 2019
    current_year = datetime.now().year

    prepare_dataframes(dataframes, relevant_club_ids, start_year, current_year, out_dir)


if __name__ == '__main__':
    set_random_seed()
    wrangler()
