import torch
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils.architectures import Discriminator
from utils.dataset import lifesci
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

results_type = 'datasynthesizer'
shuffle_cols = False
num_classifiers = 10 #Define number of classifiers trained in parallel

real_label = 1 #Define data labels
fake_label = 0

'''
eps_list: List of privacy budgets to iterate remove_real
train_list: list of dataset numbers to iterate over
'''
eps_list = [0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100]
train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

stats_list = [] # Initialize array for statistics

Hyperparams = collections.namedtuple(
        'Hyperparams',
        ['batch_size', 'lr', 'epochs'])
Hyperparams.__new__.__defaults__ = (None, None, None)
hparams = Hyperparams(batch_size=64, lr=1e-3, epochs=10) #instantiate hyperparameters

real_data_path = f'../datasets/lifesci.csv' #Define path to real data


for item in train_list:
    auc_max = []
    for epsilon in eps_list:
        auc_scores = []
        fake_data_path = f'../Results/{type}/datasets_eps_{epsilon}/synthetic_data_{epsilon}_{item}.csv' #Construct filepath to fake data
        dataset = lifesci(real_data_path, fake_data_path, None, shuffle_cols) #Instantiate lifesci dataset object
        train, test = data_utils.random_split(dataset, [35822, 17644]) #Split into train and test sets
        train_loader = DataLoader(dataset=train, shuffle=True, batch_size=512, num_workers=8, pin_memory=True) #Initialize dataloaders
        test_loader = DataLoader(dataset=test, shuffle=True, batch_size=17644, num_workers=8, pin_memory=True)
        for i in range(num_classifiers):
            print("Training classifier "+str(i+1)+" on dataset "+str(item)+" of epsilon "+str(epsilon))
            disc = Discriminator(11).cuda() # Instantiate discriminator
            disc.train(train_loader, hparams) # Train discriminator
            roc = disc.test(test_loader) # Obtain AUROC score
            auc_scores.append(roc) # Append to list
        auc_max.append(max(auc_scores)) # Take maximum AUROC across all classifiers
    stats_list.append(auc_max) #
stats_list = np.array(stats_list)
if shuffle_cols:
    np.savetxt(f'../posthoc_results/{results_type}_shuffle_aucs/nn.csv', stats_list, delimiter=',')
else:
    np.savetxt(f'../posthoc_results/{results_type}_aucs/nn.csv', stats_list, delimiter=',') #Save file
