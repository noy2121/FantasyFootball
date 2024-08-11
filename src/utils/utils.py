import os
import logging
from typing import List
logger = logging.getLogger(__name__)

import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path


def set_random_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_root_dir():
    root_dir = Path(__file__).parent.parent.parent.absolute()

    return str(root_dir)


def load_datasets(filenames=None):

    root_dir = get_root_dir()
    data_dir = os.path.join(root_dir, 'data')

    if filenames is None:  # load all datasets
        dfs = {str(Path(fn).stem): pd.read_csv(f'{data_dir}/{fn}', keep_default_na=False) for fn in
               os.listdir(data_dir) if Path(fn).suffix == '.csv'}
    else:  # load only datasets specified in config
        dfs = {fn: pd.read_csv(f'{data_dir}/{fn}', keep_default_na=False) for fn in filenames
               if Path(fn).exists() and Path(fn).suffix == '.csv'}
    print(f'Load {len(dfs)} different datasets:')
    for k, v in dfs.items():
        print(f'\t- {k}: shape {v.shape}')

    return dfs


def save_dataframe(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)


def save_dataframes(dfs: List[pd.DataFrame], filepaths: List[str]):
    for df, fp in zip(dfs, filepaths):
        save_dataframe(df, fp)
