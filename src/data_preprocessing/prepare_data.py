import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from unidecode import unidecode
from pathlib import Path
from functools import reduce
from datasets_structure import teams, players, clubs, events, matches
from utils.utils import get_root_dir, set_random_seed
ROOT_DIR = get_root_dir()


def get_relevant_club_ids(df):
    df = df.loc[df['name'].isin(teams)]

    return df['club_id']


def filter_data_by_year(df, year):
    query = df['date'] >= f'{year}-08-01'
    df.query("@query", inplace=True)

    return df


def filter_data_by_club_id(df, colname, club_ids):
    df = df.loc[df[colname].isin(club_ids)]

    return df


def fix_name_format(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    df[colname] = df[colname].apply(unidecode)

    return df


def get_player_stats(df, colname):
    stats_df = df.groupby(['player_id', 'period'])[colname].sum().reset_index(name=colname)
    stats_per_year_df = stats_df.pivot(index='player_id', columns='period', values=colname).fillna(0)
    total_stats = stats_per_year_df.sum(axis=1)

    # Create the final DataFrame
    result = pd.DataFrame({
        'player_id': stats_per_year_df.index,
        f'{colname}_per_year': stats_per_year_df.values.tolist(),
        f'total_{colname}': total_stats
    })

    return result


def merge_player_stats(players_df, stat_dfs) -> pd.DataFrame:
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
    query = players_df['last_season'] >= 2023
    players_df.query("@query", inplace=True)
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


def get_lineups(lineups_df, players_df, year) -> pd.DataFrame:
    """
    collect players lineup data and merge it with player stats df.
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

    merged_df = pd.merge(players_df, lineups_sum_df, on='player_id', how='left')
    return merged_df


def create_players_df(dfs, club_ids, s_year):
    print('create players dataframe')

    app_df = dfs['appearances']
    app_df = filter_data_by_year(app_df, s_year)
    app_df = filter_data_by_club_id(app_df, 'player_current_club_id', club_ids)
    app_df['date'] = pd.to_datetime(app_df['date'])
    bins = [pd.Timestamp(f'{s_year + i}-08-01') for i in range(6)]
    labels = [f'{(s_year + i) % 100}/{(s_year + i + 1)%100}' for i in range(5)]
    app_df['period'] = pd.cut(app_df['date'],
                              bins=bins,
                              labels=labels)

    stats_dfs = []
    for colname in ['goals', 'assists', 'red_cards', 'yellow_cards']:
        curr = get_player_stats(app_df, colname)
        curr.reset_index(drop=True, inplace=True)
        stats_dfs.append(curr)

    players_df = merge_player_stats(dfs['players'], stats_dfs)
    players_df = get_lineups(dfs['game_lineups'], players_df, s_year)
    players_df = fix_name_format(players_df, 'player_name')

    out_dir = os.path.join(ROOT_DIR, 'data/csvs')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    players_df.to_csv(f'{out_dir}/players_df.csv', index=False)


def preprocess_data():

    data_dir = os.path.join(ROOT_DIR, 'data/raw_csvs')
    dataframes = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            dataframes[filename[:-4]] = pd.read_csv(filepath)

    rel_club_ids = get_relevant_club_ids(dataframes["clubs"])
    starting_year = 2019
    create_players_df(dataframes, rel_club_ids, starting_year)


if __name__ == '__main__':
    set_random_seed()
    preprocess_data()
