import os
from pathlib import Path
from functools import reduce
from typing import List, Dict

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from unidecode import unidecode

from datasets_structure import teams, events_cols, matches_cols
from src.utils.utils import get_root_dir, set_random_seed

ROOT_DIR = get_root_dir()


def get_relevant_club_ids(df: pd.DataFrame) -> List[int]:
    """
    Retrieve relevant club IDs based on the club names matching the predefined teams list.
    """
    assert df['club_name'].tolist() == teams
    return df['club_id'].tolist()


def filter_data_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Filter DataFrame by a given season year, considering the season starts in August.
    """
    query = df['date'] >= f'{year}-08-01'
    return df.query("@query")


def filter_data_by_club_id(df: pd.DataFrame, colname: str, club_ids: List[int]) -> pd.DataFrame:
    """
    Filter DataFrame by club IDs.
    """
    return df[df[colname].isin(club_ids)]


def fix_name_format(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """
    Standardize the name format by removing accents and special characters.
    """
    df[colname] = df[colname].apply(unidecode)
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


def create_players_df(dfs: Dict[str, pd.DataFrame], club_ids: List[int], start_year: int, out_dir: str):
    """
    Create and save a comprehensive players DataFrame.
    """
    print('Create players dataframe...')

    app_df = filter_data_by_year(dfs['appearances'], start_year)
    app_df = filter_data_by_club_id(app_df, 'player_current_club_id', club_ids)
    app_df['date'] = pd.to_datetime(app_df['date'])

    bins = [pd.Timestamp(f'{start_year + i}-08-01') for i in range(6)]
    labels = [f'{(start_year + i) % 100}/{(start_year + i + 1) % 100}' for i in range(5)]
    app_df['period'] = pd.cut(app_df['date'], bins=bins, labels=labels)

    stats_dfs = []
    for colname in ['goals', 'assists', 'red_cards', 'yellow_cards']:
        curr = get_player_stats(app_df, colname)
        curr.reset_index(drop=True, inplace=True)
        stats_dfs.append(curr)

    players_df = merge_player_stats(dfs['players'], stats_dfs)
    players_df = get_lineups(dfs['game_lineups'], players_df, start_year)
    players_df = fix_name_format(players_df, 'player_name')

    print("Save players DataFrame...")
    players_df.to_csv(f'{out_dir}/players_df.csv', index=False)


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

    dataframes['clubs'] = pd.read_csv(cfg.clubs_df_filepath)
    relevant_club_ids = get_relevant_club_ids(dataframes['clubs'])

    starting_year = 2019
    out_dir = os.path.join(ROOT_DIR, 'data/csvs')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    create_players_df(dataframes, relevant_club_ids, starting_year, out_dir)


if __name__ == '__main__':
    set_random_seed()
    preprocess_data()
