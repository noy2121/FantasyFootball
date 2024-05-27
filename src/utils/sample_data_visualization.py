# imports

import os
import numpy as np
import pandas as pd
import plotly.express as px

from pathlib import Path


def load_datasets():
    # load data
    dfs = {str(Path(filename).stem): pd.read_csv(f'../data/{filename}', nrows=10000, keep_default_na=False)
           for filename in os.listdir('data') if Path(filename).suffix == '.csv'}
    print(f'We have {len(dfs)} different datasets:')
    for k, v in dfs.items():
        print(f'\t- {k}: shape {v.shape}')

    return dfs


def plot_goals_number_in_leagues(dfs):
    # plot number of goals of the top 5 leagues per year

    top5leagues = {'ES1': 'laliga', 'FR1': 'ligue-1', 'IT1': 'seria-a', 'GB1': 'preimer-ligue', 'L1': 'bundesliga'}
    games_df = dfs['games']

    games_filtered_df = games_df[games_df['competition_id'].isin(top5leagues.keys())]
    games_filtered_df['date'] = pd.to_datetime(games_filtered_df['date'], format='%Y-%m-%d')

    # change date format to year only
    games_filtered_df['date'] = games_filtered_df['date'].dt.year.astype(str)

    # add total numbers of goals
    games_filtered_df['total_goals'] = games_filtered_df['home_club_goals'] + games_filtered_df['away_club_goals']

    # calculate total number of goals per year per league
    total_goals_per_year_per_league_df = games_filtered_df.groupby(['competition_id', 'date'])[
        'total_goals'].sum().reset_index()

    fig = px.line(total_goals_per_year_per_league_df, x='date', y='total_goals', color='competition_id',
                  labels={'date': 'Year', 'total_goals': 'Goals', 'competition_id': 'League'},
                  title='Number of Goals per league')
    fig.show()


def plot_players_worth_dist(dfs):
    # plot distribution of players worth per country
    pv_df = dfs['player_valuations']
    players_df = dfs['players']

    # Create a dictionary from players_df with player_id as keys and country as values
    country_map = players_df.set_index('player_id')['country_of_citizenship'].to_dict()

    # Use the map function to add the country column to pv_df
    pv_df['country'] = pv_df['player_id'].map(country_map)
    fig = px.histogram(pv_df, x='market_value_in_eur', y='country')
    fig.show()


def run_sample():
    dfs = load_datasets()
    plot_goals_number_in_leagues(dfs)
    plot_players_worth_dist(dfs)


if __name__ == '__main__':
    run_sample()