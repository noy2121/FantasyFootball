import pandas as pd
from src.utils.utils import load_dataframes


def load_raw_data(raw_data_dir):
    """
    Load raw CSV files from S3 into pandas DataFrames.

    Returns:
        dict: A dictionary of DataFrames keyed by dataset name.
    """

    raw_data = load_dataframes(raw_data_dir)
    for name, df in raw_data.items():
        if "player_id" in df.columns:
            df.dropna(subset=["player_id"])
        elif "club_id" in df.columns:
            df.dropna(subset=["club_id"])
        elif "game_id" in df.columns:
            df.dropna(subset=["game_id"])



    return raw_data
