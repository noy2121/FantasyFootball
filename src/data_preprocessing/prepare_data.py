import os
import sys
import pandas as pd
pd.set_option("display.max_columns", None)

from datasets_structure import teams, players, clubs, events, matches
from pathlib import Path
from utils.utils import get_root_dir, set_random_seed
ROOT_DIR = get_root_dir()


def get_relevant_club_ids(df):
    df = df.loc[df['name'].isin(teams)]

    return df['club_id']


def filter_data_by_year(df):
    query = df['date'] >= '2021-08-01'
    df.query("@query", inplace=True)

    return df


def filter_data_by_club_id(df, colname, club_ids):
    df = df.loc[df[colname].isin(club_ids)]

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


def create_players_df(dfs, club_ids):

    import pdb
    players_df_columns = {k: None for k in players}

    app_df = dfs['appearances']
    app_df = filter_data_by_year(app_df)
    app_df = filter_data_by_club_id(app_df, 'player_current_club_id', club_ids)
    app_df['date'] = pd.to_datetime(app_df['date'])
    bins = [pd.Timestamp('2021-08-01'), pd.Timestamp('2022-08-01'),
            pd.Timestamp('2023-08-01'), pd.Timestamp('2024-08-01')]
    app_df['period'] = pd.cut(app_df['date'],
                              bins=bins,
                              labels=['21/22', '22/23', '23/24'])

    results = []
    for colname in ['goals, assists, red_cards, yellow_cards']:
        results.append(get_player_stats(app_df, colname))

def preprocess_data():

    data_dir = os.path.join(ROOT_DIR, 'data/csvs')
    dataframes = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            dataframes[filename[:-4]] = pd.read_csv(filepath)

    rel_club_ids = get_relevant_club_ids(dataframes["clubs"], )
    create_players_df(dataframes, rel_club_ids)


if __name__ == '__main__':
    preprocess_data()
