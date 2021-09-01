import torch
import pandas as pd
from models import dpwgan
from sklearn.model_selection import train_test_split
import collections


X_train = pd.read_csv("../../datasets/lifesci.csv", header=1) # Read original data

input_dim = X_train.shape[1] #Infer input dimension from data

z_dim = 25 # Latent dimension is pre-defined

eps_list =  [
(0.01, 64), (0.0316, 16), (0.1, 4),
(0.316, 2), (1, 2), (3.16, 2),
(10, 2), (31.6, 0.5), (100, 0.5)
] # eps_list contains (epsilon, noise_scale) pairs for training

target_delta = 1e-5 # Privacy leakage parameter set in advance

GPU = True # Choose whether to use GPU
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

Hyperparams = collections.namedtuple(
        'Hyperparams',
        ['batch_size', 'clamp_floor', 'clamp_ceil', 'clip_coeff', 'nc', 'lr', 'scale'])

Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)

for epsilon, sigma in eps_list:
    hparams = Hyperparams(batch_size=64, clamp_floor=-0.01, clamp_ceil=0.01, clip_coeff=0.03934, nc=5, lr=5e-5, scale=sigma) #Instantiate hyperparameters
    for i in range(10):
        print("Training model " + str(i+1) + " at target epsilon " + str(epsilon) + ", scale "+str(sigma)+", and target delta " + str(target_delta) +".")
        model = dpwgan.DPWGAN(input_dim, z_dim, epsilon, target_delta, device) #Initialise model
        model.train(X_train, hparams) # Train model
        print("Generating synthetic data for model " + str(i+1) + "...")
        fname = f'dpwgan_results/datasets_eps_{epsilon}/data_{epsilon}_{i+1}.csv'
        model.generate_synthetics(X_train.shape[0], fname, False) # Generate synthetic data
