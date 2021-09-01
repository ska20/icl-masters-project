from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np

class lifesci(Dataset):
    def __init__(self, fpath_real, remove_real=True, transform=None):

        self.transform = transform

        real_data = pd.read_csv(fpath_real, header=None )
        if remove_real:
            real_data = real_data[1:]

        self.data = real_data

    def __len__(self):
        return self.data.shape[0]

    def __dim__(self):
        return self.data.shape[1]

    def __getitem__(self, index):

        x = torch.tensor(self.data.iloc[index, :].to_numpy(dtype='float32'))

        return x
