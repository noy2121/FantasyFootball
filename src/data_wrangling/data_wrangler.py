import os
from typing import Dict, Set, List, Tuple
from datetime import datetime

import hydra
import pandas as pd
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame

from players_wrangler import create_players_df, create_text_players_df
from games_wrangler import create_games_df, create_text_games_df, create_text_clubs_df
from events_wrangler import create_events_df, create_text_events_df
from wrangler_utils import get_relevant_club_ids
from src.utils.utils import ROOT_DIR, set_random_seed, save_dataframes


def prepare_dataframes(raw_dfs: Dict[str, pd.DataFrame], club_ids: Set[int], start_year: int,
                       curr_year: int) -> Dict[str, pd.DataFrame]:

    clubs_df = raw_dfs['clubs']
    # create players df
    players_df = create_players_df(raw_dfs, club_ids, start_year, curr_year)
    games_df = create_games_df(raw_dfs['games'], club_ids, start_year, curr_year)
    events_df = create_events_df(raw_dfs['game_events'], games_df, start_year)

    return {'players': players_df, 'clubs': clubs_df, 'games': games_df, 'events': events_df}


def split_df(df: pd.DataFrame, test_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Create masks for train, validation, and test sets
    test_date = f'{test_year}-08-01'
    train_mask = df['date'] < test_date
    test_mask = df['date'] >= test_date

    # Split the dataframe
    train_df = df[train_mask]
    test_df = df[test_mask]

    return train_df, test_df


def train_test_split(dfs: Dict[str, pd.DataFrame], test_year: int) -> Dict[str, Dict[str, pd.DataFrame]]:

    players_train, players_test = split_df(dfs['players'], test_year)
    games_train, games_test = split_df(dfs['games'], test_year)
    events_train, events_test = split_df(dfs['events'], test_year)

    return {
        'train': {
            'players': players_train,
            'games': games_train,
            'events': events_train
        },
        'test': {
            'players': players_test,
            'games': games_test,
            'events': events_test
        }
    }


def convert_dfs_to_text(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:

    players_text = create_text_players_df(dfs['players'])
    clubs_text = create_text_clubs_df(dfs['clubs'])
    games_text = create_text_games_df(dfs['games'])
    events_text = create_text_events_df(dfs)

    return {'players': players_text, 'clubs': clubs_text, 'games': games_text, 'events': events_text}


@hydra.main(config_path='../config', config_name='conf')
def wrangler(cfg):

    data_dir = os.path.join(ROOT_DIR, cfg.data.data_dir)
    start_year = cfg.data.start_year
    test_year = cfg.data.test_year
    current_year = datetime.now().year

    if cfg.data.process_raw_data:
        raw_dataframes = {}
        raw_csvs_dir = f'{data_dir}/raw_csvs'
        for filename in os.listdir(raw_csvs_dir):
            if filename == 'clubs.csv':
                continue
            if filename.endswith('.csv'):
                filepath = os.path.join(raw_csvs_dir, filename)
                raw_dataframes[filename[:-4]] = pd.read_csv(filepath)

        raw_dataframes['clubs'] = pd.read_csv(f'{raw_csvs_dir}/clubs_data.csv')
        relevant_club_ids = get_relevant_club_ids(raw_dataframes['clubs'])

        dataframes = prepare_dataframes(raw_dataframes, relevant_club_ids, start_year, current_year)
        print("Save DataFrames...")
        out_dir = f'{data_dir}/csvs'
        save_dataframes(dataframes, out_dir)

    # TODO: save train/val/test in different folders
    if cfg.data.split_data:
        dataframes = load_dataframes()
        train_test_dfs = train_test_split(dataframes, val_year, test_year)
        print("Save train/val/test DataFrames...")
        for key, dfs in train_test_dfs.items():
            out_dir = f'{data_dir}/preprocessed/{key}/csvs'
            save_dataframes(dfs, out_dir)

    if cfg.data.create_text_data:
        dataframes = load_dataframes()
        dataframes = convert_dfs_to_text(dataframes)
        print("Save text DataFrames...")
        out_dir = f'{data_dir}/preprocessed/'
        save_dataframes(dataframes, f'{out_dir}/csvs')


if __name__ == '__main__':

    set_random_seed()
    wrangler()
