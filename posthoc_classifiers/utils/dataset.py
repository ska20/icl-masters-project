from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np

'''
Custom Dataset class to wrap lifesci dataset. Used when training the torch
neural network classifiers.
'''

class lifesci(Dataset):
    '''
    Initializes dataset.
    INPUT: fpath_real - path to real data file
    INPUT: fpath_fake - path to synthetic data file
    INPUT: transform - list of data transforms
    '''
    def __init__(self, fpath_real, fpath_fake, transform=None, shuffle_cols):

        self.transform = transform # No transform is assigned

        real_data = pd.read_csv(fpath_real, header=None ) # Read real data
        real_data = real_data[1:] # Remove header from real data

        fake_data = pd.read_csv(fpath_fake, header=None) # Read fake data

        if 'datasynthesizer' in fpath_fake:
            fake_data = fake_data[1:] #remove header for DataSynthesizer
            fake_data.columns = pd.RangeIndex(fake_data.columns.size)

        if 'dpwgan' in fpath_fake:
            fake_data = fake_data.iloc[1:, :] #remove header for dpwgan
            fake_data = fake_data.iloc[: , 1:] #remove index col for dpwgan
            fake_data.columns = pd.RangeIndex(fake_data.columns.size)
            fake_data[10][fake_data[10] < 0.5] = 0 #threshold
            fake_data[10][fake_data[10] >= 0.5] = 1

        if 'dp-copula' in fpath_fake:
            fake_data.columns = pd.RangeIndex(fake_data.columns.size)

        if 'pategan' in fpath_fake:
            fake_data.columns = pd.RangeIndex(fake_data.columns.size)
            fake_data[10][fake_data[10] < 0.5] = 0 #threshold
            fake_data[10][fake_data[10] >= 0.5] = 1

        real_vector = [1 for i in range(real_data.shape[0])]
        fake_vector = [0 for i in range(fake_data.shape[0])]

        real_data[11] = real_vector #attach class label
        fake_data[11] = fake_vector #attach class label

        if shuffle_cols:
            fake_data.apply(lambda x: x.sample(frac=1).values)

        frames = [real_data, fake_data]

        concat = pd.concat(frames, sort=False)
        shuffle = concat.sample(frac=1).reset_index(drop=True) #shuffle dataset
        self.labels = shuffle.iloc[:, [-1]] #detach shuffled labels
        self.data = shuffle.drop([11], axis=1) #drop label col
    '''
    Returns number of elements in the dataset
    '''
    def __len__(self):
        return self.data.shape[0]

    '''
    Returns a (Data, label) pair.
    '''
    def __getitem__(self, index):

        x = torch.tensor(self.data.iloc[index, :].to_numpy(dtype='float32'))
        y = torch.tensor(self.labels.iloc[index].to_numpy(dtype='float32'))

        return(x, y)
