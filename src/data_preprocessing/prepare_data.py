import os
from pathlib import Path
from functools import reduce
from typing import List, Dict
from datetime import datetime

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from unidecode import unidecode

from datasets_structure import teams, events_cols, games_cols
from src.utils.utils import get_root_dir, set_random_seed, save_dataframes

ROOT_DIR = get_root_dir()


def get_relevant_club_ids(df: pd.DataFrame) -> List[int]:
    """
    Retrieve relevant club IDs based on the club names matching the predefined teams list.
    """
    assert set(df['club_name']) == teams, f'Clubs in the DataFrame are different in {teams} !'
    return df['club_id'].tolist()


def filter_data_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Filter DataFrame by a given season year, considering the season starts in August.
    """
    query = df['date'] >= f'{year}-08-01'
    return df.query("@query")


def filter_data_by_club_id(df: pd.DataFrame, colnames: List[str], club_ids: List[int]) -> pd.DataFrame:
    """
    Filter DataFrame by club IDs.
    """
    mask = df[colnames].isin(club_ids).all(axis=1)
    return df[mask]


def fix_name_format(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """
    Standardize the name format by removing accents and special characters.
    """
    df[colname] = df[colname].apply(unidecode)
    return df


def add_period_to_df(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Adds "period" column to dataframe for better date filtering.
    """
    years = end - start
    df['date'] = pd.to_datetime(df['date'])
    bins = [pd.Timestamp(f'{start + i}-08-01') for i in range(years + 1)]
    labels = [f'{(start + i) % 100}/{(start + i + 1) % 100}' for i in range(years)]
    df['period'] = pd.cut(df['date'], bins=bins, labels=labels)

    return df


def get_player_stats(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """
    Aggregate player statistics over different periods.
    """
    stats_df = df.groupby(['player_id', 'period'])[colname].sum().reset_index(name=colname)
    stats_per_year_df = stats_df.pivot(index='player_id', columns='period', values=colname).fillna(0)
    total_stats = stats_per_year_df.sum(axis=1)

    # Create the final DataFrame
    return pd.DataFrame({
        'player_id': stats_per_year_df.index,
        f'{colname}_per_year': stats_per_year_df.values.tolist(),
        f'total_{colname}': total_stats
    })


def merge_player_stats(players_df: pd.DataFrame, stat_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges player statistics from multiple dataframes into a single comprehensive dataframe.

    Parameters:
    ----------
    main_df (pd.DataFrame): DataFrame with columns including ["player_id", "player_name", "position", "club_id", "date_of_birth"]
    stat_dfs (list): List of DataFrames with columns ["player_id", "<stat>_per_year", "total_<stat>"]

    Returns:
    ----------
    pd.DataFrame: A comprehensive DataFrame with specified player information and statistics
    """

    # Check if main_df has all required columns
    main_cols = ["player_id", "name", "position", "current_club_id", "date_of_birth"]
    if not all(col in players_df.columns for col in main_cols):
        raise ValueError(f"main_df must have all of these columns: {main_cols}")

    # Select only the required columns from main_df
    players_df.query("last_season >= 2023", inplace=True)
    main_df_subset = players_df[main_cols]

    # Check if all stat_dfs have 'player_id' column
    if not all('player_id' in df.columns for df in stat_dfs):
        raise ValueError("All stat dataframes must have a 'player_id' column")

    # Merge all dataframes
    dfs_to_merge = [main_df_subset] + stat_dfs
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='player_id', how='left'), dfs_to_merge)

    # Fill NaN values with appropriate defaults
    for col in merged_df.columns:
        if col.endswith('_per_year'):
            merged_df[col] = merged_df[col].apply(lambda x: x if isinstance(x, list) else [0, 0, 0])
        elif col.startswith('total_'):
            merged_df[col] = merged_df[col].fillna(0)

    merged_df.rename(columns={'name': 'player_name', 'current_club_id': 'club_id'}, inplace=True)
    return merged_df


def get_lineups(lineups_df: pd.DataFrame, players_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Collect players lineup data and merge it with player stats df.
    Parameters
    ----------
    lineups_df (pd.DataFrame): DataFrame with columns including ["date", "club_id", "player_id", "type"]
    players_df (pd.DataFrame)

    Returns
    -------
    pd.DataFrame: A comprehensive DataFrame with specified player information and statistics
    """

    lineups_df = filter_data_by_year(lineups_df, year)
    lineups_sum_df = pd.get_dummies(lineups_df['type']).groupby(lineups_df['player_id']).sum().reset_index()

    return pd.merge(players_df, lineups_sum_df, on='player_id', how='left')


def split_cards_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split 'Cards' event type to 'yellow-card', 'red-card', 'second-yellow-card' event types.
    """
    new_events_df = events_df.copy()

    cards_mask = new_events_df['type'] == 'Cards'
    cards_df = new_events_df[cards_mask]

    yellows = cards_df[cards_df['description'].str.contains('yellow', case=False, na=False) &
                       ~cards_df['description'].str.contains('second yellow', case=False, na=False)].copy()
    second_yellows = cards_df[cards_df['description'].str.contains('second yellow', case=False, na=False)].copy()
    reds = cards_df[cards_df['description'].str.contains('red', case=False, na=False)].copy()

    assert (yellows['type'] == 'Cards').all(), f'Some yellow card events have wrong event type !'
    assert (second_yellows['type'] == 'Cards').all(), f'Some second yellow card events have wrong event type !'
    assert (reds['type'] == 'Cards').all(), f'Some red card events have wrong event type !'

    # change event type Cards -> yellow/second/red
    yellows['type'] = 'yellow-card'
    second_yellows['type'] = 'second-yellow-card'
    reds['type'] = 'red-card'

    new_cards = pd.concat([yellows, second_yellows, reds])
    new_events_df = pd.concat([new_events_df[~cards_mask], new_cards]).reset_index(drop=True)
    new_events_df.rename(columns={'game_event_id': 'event_id', 'type': 'event_type'})

    return new_events_df


def create_players_df(dfs: Dict[str, pd.DataFrame], club_ids: List[int], start_year: int, curr_year: int) -> pd.DataFrame:
    """
    Create a comprehensive players DataFrame.
    """
    print('Extract players data...')

    app_df = filter_data_by_year(dfs['appearances'], start_year)
    app_df = filter_data_by_club_id(app_df, ['player_current_club_id'], club_ids)
    app_df = add_period_to_df(app_df, start_year, curr_year)

    stats_dfs = []
    for colname in ['goals', 'assists', 'red_cards', 'yellow_cards']:
        curr = get_player_stats(app_df, colname)
        curr.reset_index(drop=True, inplace=True)
        stats_dfs.append(curr)

    players_df = merge_player_stats(dfs['players'], stats_dfs)
    players_df = get_lineups(dfs['game_lineups'], players_df, start_year)
    players_df = fix_name_format(players_df, 'player_name')

    return players_df


def create_games_df(raw_games_df: pd.DataFrame, club_ids: List[int], start_year: int, curr_year: int) -> pd.DataFrame:
    """
    Create a comprehensive games DataFrame.
    """
    print('Extract games data...')

    games_df = filter_data_by_year(raw_games_df, start_year)
    games_df = filter_data_by_club_id(games_df, ['home_club_id', 'away_club_id'], club_ids)
    games_df = add_period_to_df(games_df, start_year, curr_year)

    return games_df[games_cols]


def create_events_df(raw_events_df: pd.DataFrame, games_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    print('Extract events data...')
    # filter by game_id
    # no need to filter by year or club_id, games_df has already been filtered this way
    relevant_game_ids = games_df['game_id'].tolist()
    events_df = raw_events_df[raw_events_df['game_id'].isin(relevant_game_ids)]

    events_df['date'] = pd.to_datetime(events_df['date'])
    cutoff_date = pd.Timestamp(f'{start_year}-08-01')
    assert (events_df['date'] >= cutoff_date).all(), f'Some dates in the DataFrame are before {cutoff_date}'

    events_df = split_cards_events(events_df)
    events_df = events_df[events_cols]

    return events_df


def prepare_dataframes(raw_dfs: Dict[str, pd.DataFrame], club_ids: List[int], start_year: int, curr_year: int, out_dir: str):

    # create players df
    players_df = create_players_df(raw_dfs, club_ids, start_year, curr_year)
    games_df = create_games_df(raw_dfs['games'], club_ids, start_year, curr_year)
    events_df = create_events_df(raw_dfs['game_events'], games_df, start_year)

    print("Save DataFrames...")
    filepaths = ['players_data', 'games_data', 'events_data']
    filepaths = [f'{out_dir}/{fp}.csv' for fp in filepaths]
    save_dataframes([players_df, games_df, events_df], filepaths)


def preprocess_data():

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
    preprocess_data()
