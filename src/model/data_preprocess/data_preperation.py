import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from src.utils.utils import load_datasets
from src.model.data_preprocess.football_datasets import FootballDataset


def prepare_data(cfg):

    # load datasets
    filenames = cfg.datasets
    logger.info(f'Preparing {len(filenames)} datasets')
    dfs = load_datasets(filenames)
    football_datasets_dict = {k: FootballDataset(k, dfs) for k in dfs.keys()}

    # get text dfs
    dfs2text_names = ['games', 'players', 'player_valuations']   # set from config
    text_dfs = {name: football_datasets_dict[name].create_text_df() for name in dfs2text_names}

    return text_dfs


if __name__ == '__main__':
    prepare_data()
