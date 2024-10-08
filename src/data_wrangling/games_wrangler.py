from typing import Set

import pandas as pd

from wrangler_utils import filter_data_by_year, filter_data_by_club_id, add_period_to_df, get_club_name_by_club_id
from datasets_structure import games_cols


def clean_formations(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean formations from text, to have format a-b-c-d
    """
    games_df['home_club_formation'] = games_df['home_club_formation'].str.split().str[0]
    games_df['away_club_formation'] = games_df['away_club_formation'].str.split().str[0]

    return games_df


def clean_competitions(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean blanks and irrelevant competitions.
    """
    def update_row(row):
        if row['competition_type'] in ['domestic_league', 'domestic_cup', 'other']:
            return row['competition_type']
        elif row['competition_type'] == 'international_cup':
            if row['competition_id'] == 'EL':
                return 'europa_league'
            elif row['competition_id'] == 'CL':
                return 'champions_league'
            else:
                return 'other'
        else:
            return 'other'

    games_df['competition_type'] = games_df.apply(update_row, axis=1)
    return games_df


def create_games_df(raw_games_df: pd.DataFrame, club_ids: Set[int], start_year: int, curr_year: int) -> pd.DataFrame:
    """
    Create a comprehensive games DataFrame.
    """
    print('Extract Games data...')

    games_df = filter_data_by_year(raw_games_df, start_year)
    games_df = filter_data_by_club_id(games_df, ['home_club_id', 'away_club_id'], club_ids)
    games_df = add_period_to_df(games_df, start_year, curr_year)
    games_df = clean_formations(games_df)
    games_df = clean_competitions(games_df)

    return games_df[games_cols]


def create_text_clubs_df(df: pd.DataFrame) -> pd.DataFrame:
    print('Convert Clubs data to text...')

    def format_row(row):
        return ', '.join(f"{col}: {val}" for col, val in row.items() if col != 'club_id')

    return pd.DataFrame({'text': df.apply(format_row, axis=1)})


def create_text_games_df(df: pd.DataFrame) -> pd.DataFrame:
    print('Convert Games data to text...')

    def format_line(col, val):
        if 'club_id' in col:
            val = get_club_name_by_club_id(val)
            col = f'{col.split("_")[0]}_club_name'
            return f'{col}: {val}'
        else:
            return f'{col}: {val}'

    def format_row(row):
        return ', '.join(f"{format_line(col, val)}" for col, val in row.items() if col != 'game_id')

    return pd.DataFrame({'text': df.apply(format_row, axis=1)})