import os
from typing import Dict, Set, List, Tuple
from datetime import datetime

import hydra
import pandas as pd

from players_wrangler import create_players_df, create_text_players_df
from games_wrangler import create_games_df, create_text_games_df, create_text_clubs_df
from events_wrangler import create_events_df, create_text_events_df
from wrangler_utils import get_relevant_club_ids, fix_name_format
from src.utils.utils import ROOT_DIR, set_random_seed, save_dataframes, load_dataframes


def prepare_dataframes(raw_dfs: Dict[str, pd.DataFrame], club_ids: Set[int], start_year: int,
                       curr_year: int) -> Dict[str, pd.DataFrame]:

    clubs_df = raw_dfs['clubs']
    clubs_df = fix_name_format(clubs_df, 'club_name')
    # create players df
    players_df = create_players_df(raw_dfs, club_ids, start_year, curr_year)
    games_df = create_games_df(raw_dfs['games'], club_ids, start_year, curr_year)
    events_df = create_events_df(raw_dfs['game_events'], games_df, start_year)

    return {'players': players_df, 'clubs': clubs_df, 'games': games_df, 'events': events_df}


def split_players_df(df: pd.DataFrame, test_year: int, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if name == 'players':
        base_columns = ['player_id', 'player_name', 'club_id', 'position', 'date_of_birth']
    elif name == 'clubs':
        base_columns = ['club_id', 'club_name', 'number_of_champions_league_titles']
    else:
        raise ValueError(f'Value of name must be in ["player", "clubs"]. Got {name=} instead!')

    test_mask = df.columns.str.contains(str(test_year), case=False)
    train_df = df[list(df.columns[~test_mask])]
    test_df = df[base_columns + list(df.columns[test_mask])]

    return train_df, test_df


def split_df(df: pd.DataFrame, test_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Create masks for train, validation, and test sets
    test_date = f'{test_year}-08-01'
    test_mask = df['date'] >= test_date

    # Split the dataframe
    train_df = df[~test_mask]
    test_df = df[test_mask]

    return train_df, test_df


def train_test_split(dfs: Dict[str, pd.DataFrame], test_year: int) -> Dict[str, Dict[str, pd.DataFrame]]:

    players_train, players_test = split_players_df(dfs['players'], test_year, 'players')
    clubs_train, clubs_test = split_players_df(dfs['clubs'], test_year, 'clubs')
    games_train, games_test = split_df(dfs['games'], test_year)
    events_train, events_test = split_df(dfs['events'], test_year)

    return {
        'train': {
            'players': players_train,
            'clubs': clubs_train,
            'games': games_train,
            'events': events_train
        },
        'test': {
            'players': players_test,
            'clubs': clubs_test,
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
        raw_csvs_dir = f'{data_dir}/raw-csvs'
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
        dataframes = load_dataframes(f'{data_dir}/csvs')
        train_test_dfs = train_test_split(dataframes, test_year)
        print("Save train/test DataFrames...")
        for key, dfs in train_test_dfs.items():
            out_dir = f'{data_dir}/preprocessed/{key}/csvs'
            save_dataframes(dfs, out_dir)

    if cfg.data.create_text_data:
        for key in ['train', 'test']:
            dataframes = load_dataframes(f'{data_dir}/preprocessed/{key}/csvs')
            dataframes = convert_dfs_to_text(dataframes)
            print(f"Save {key} Text-DataFrames...")
            out_dir = f'{data_dir}/preprocessed/{key}/text_data'
            save_dataframes(dataframes, out_dir)


if __name__ == '__main__':

    set_random_seed()
    wrangler()
