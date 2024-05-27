
import torch
from torch.utils.data import Dataset


class FootballTorchDataset(Dataset):

    def __init__(self, ds_name, encodings, transform=None):

        self.name = ds_name
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)