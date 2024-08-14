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
    for name, df in dfs.items():
        filepath = f'{out_dir}/{name}.csv'
        save_dataframe(df, filepath)
