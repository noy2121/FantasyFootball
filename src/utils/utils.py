import os
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


def load_datasets():
    root_dir = get_root_dir()
    data_dir = os.path.join(root_dir, 'data')
    dfs = {str(Path(filename).stem): pd.read_csv(f'{data_dir}/{filename}', keep_default_na=False) for filename in
           os.listdir(data_dir) if Path(filename).suffix == '.csv'}
    print(f'Load {len(dfs)} different datasets:')
    for k, v in dfs.items():
        print(f'\t- {k}: shape {v.shape}')

    return dfs