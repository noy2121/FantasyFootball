import os
from typing import Dict, Set
from datetime import datetime

import hydra
import pandas as pd

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


def convert_dfs_to_text(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:

    players_text = create_text_players_df(dfs['players'])
    clubs_text = create_text_clubs_df(dfs['clubs'])
    games_text = create_text_games_df(dfs['games'])
    events_text = create_text_events_df(dfs)

    return {'players': players_text, 'clubs': clubs_text, 'games': games_text, 'events': events_text}


@hydra.main(config_path='../config', config_name='conf')
def wrangler(cfg):

    data_dir = os.path.join(ROOT_DIR, cfg.data.raw_csvs_path)
    raw_dataframes = {}
    for filename in os.listdir(data_dir):
        if filename == 'clubs.csv':
            continue
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            raw_dataframes[filename[:-4]] = pd.read_csv(filepath)

    out_dir = os.path.join(ROOT_DIR, cfg.data.csvs_path)
    raw_dataframes['clubs'] = pd.read_csv(f'{out_dir}/clubs.csv')
    relevant_club_ids = get_relevant_club_ids(raw_dataframes['clubs'])

    start_year = cfg.data.start_year
    current_year = datetime.now().year

    dataframes = prepare_dataframes(raw_dataframes, relevant_club_ids, start_year, current_year)
    print("Save DataFrames...")
    save_dataframes(dataframes, out_dir)

    dataframes = convert_dfs_to_text(dataframes)
    out_dir = os.path.join(ROOT_DIR, cfg.data.preprocessed_data_path)
    print("Save text DataFrames...")
    save_dataframes(dataframes, out_dir)


if __name__ == '__main__':
    set_random_seed()
    wrangler()
