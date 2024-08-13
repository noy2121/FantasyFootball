from functools import reduce
from typing import Dict, List, Set

import pandas as pd
from unidecode import unidecode

from wrangler_utils import filter_data_by_year, filter_data_by_club_id, add_period_to_df, get_club_name_by_club_id


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
        if col.endswith('in_last_three_seasons'):
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

    df = pd.merge(players_df, lineups_sum_df, on='player_id', how='left')
    df.rename(columns={'substitutes': 'substitute'}, inplace=True)

    return df


def create_players_df(dfs: Dict[str, pd.DataFrame], club_ids: Set[int], start_year: int, curr_year: int) -> pd.DataFrame:
    """
    Create a comprehensive players DataFrame.
    """
    # TODO: add clean sheets calculation
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


def create_text_players_df(df: pd.DataFrame) -> pd.DataFrame:
    def format_value(val):
        if isinstance(val, list):
            return str(val) if val else ''
        elif pd.isna(val):
            return ''
        else:
            return val

    def format_line(col, val):
        if col == 'club_id':
            val = get_club_name_by_club_id(val)
            col = 'club_name'
        else:
            val = format_value(val)
        return f'{col}: {val}'

    def format_row(row):
        return ', '.join(f"{format_line(col, val)}" for col, val in row.items() if col != 'player_id')

    return pd.DataFrame({'text': df.apply(format_row, axis=1)})
