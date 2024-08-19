import os
from typing import List, Dict

import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path


ROOT_DIR = str(Path(__file__).parent.parent.parent.absolute())


def set_random_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def print_df_sample(df: pd.DataFrame):
    pd.set_option('display.max_columns', None)
    print(df.head(5))


def print_dfs_sample(dfs: List[pd.DataFrame]):
    for df in dfs:
        print_df_sample(df)
        print('')
        print('-'*100)
        print('')


def save_dataframe(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)


def save_dataframes(dfs: Dict[str, pd.DataFrame], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        filepath = f'{out_dir}/{name}.csv'
        save_dataframe(df, filepath)


def load_dataframes(data_dir: str, filenames: List[str] = None) -> Dict[str, pd.DataFrame]:
    if filenames is None:  # load all datasets
        dfs = {str(Path(fn).stem): pd.read_csv(f'{data_dir}/{fn}') for fn in
               os.listdir(data_dir) if Path(fn).suffix == '.csv'}
    else:
        dfs = {str(Path(fn).stem): pd.read_csv(f'{data_dir}/{fn}') for fn in filenames
               if Path(f'{data_dir}/{fn}').exists() and Path(fn).suffix == '.csv'}
    print(f'Load {len(dfs)} different datasets:')
    for k, v in dfs.items():
        print(f'\t- {k}: shape {v.shape}')

    return dfs


def get_hftoken(token_path):
    with open(token_path, 'r') as f:
        hf_token = f.read()

    return hf_token
