from typing import List, Set

import numpy as np
import pandas as pd

from datasets_structure import teams


def get_relevant_club_ids(df: pd.DataFrame) -> Set[int]:
    """
    Retrieve relevant club IDs based on the club names matching the predefined teams list.
    """
    assert set(df['club_name']) == teams, f'Clubs in the DataFrame are different in {teams} !'
    return set(df['club_id'])


def filter_data_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Filter DataFrame by a given season year, considering the season starts in August.
    """
    query = df['date'] >= f'{year}-08-01'
    return df.query("@query")


def filter_data_by_club_id(df: pd.DataFrame, colnames: List[str], club_ids: Set[int]) -> pd.DataFrame:
    """
    Filter DataFrame by club IDs.
    """
    if len(colnames) == 1:
        return df[df[colnames[0]].isin(club_ids)]
    else:
        mask = np.zeros(len(df), dtype=bool)
        for col in colnames:
            mask |= np.isin(df[col].values, list(club_ids))
        return df[mask]


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
