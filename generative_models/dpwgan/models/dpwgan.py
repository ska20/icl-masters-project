import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from utils.architectures import Generator, Discriminator
from utils.rdp_accountant import compute_rdp, get_privacy_spent
from utils.helper import weights_init
import collections

class DPWGAN(object):
    '''
    Initialize parameters for DPWGAN network
    '''
    def __init__(self, input_size, latent_size, target_epsilon, target_delta, device):
        self.latent_size = latent_size # initialize GAN latent dimension
        self.input_size = input_size # initialize GAN input dimension
        self.target_epsilon = target_epsilon # initialize max budget
        self.target_delta = target_delta # initialize privacy leakage parameter
        self.device = device # assign device for computation

        self.generator = Generator(latent_size, input_size).to(device) # initialize generator network
        self.discriminator = Discriminator(input_size).to(device) # initialize discriminator network

        self.generator.apply(weights_init) # initialize network weights
        self.discriminator.apply(weights_init)

        self.target_epsilon = target_epsilon # assign maximum privacy budget
        self.target_delta = target_delta # assign privacy leakage parameter

    '''
    Train the DPWGAN network
    INPUT: train - training dataset
    INPUT: hyperparams - hyperparameters for the network
    '''
    def train(self, train, hyperparams):
        grad_norms = [] # initialize array for gradient norms
        batch_size = hyperparams.batch_size # assign batch size
        lr = hyperparams.lr # assign learning rate
        clamp_ceil = hyperparams.clamp_ceil # assign ceiling for gradient clamp
        clamp_floor = hyperparams.clamp_floor # assign floor for gradient clamp
        clip_coeff = hyperparams.clip_coeff # assign clipping coefficient
        nc = hyperparams.nc
        sigma = hyperparams.scale # Manually set noise scale
        data_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.cuda.FloatTensor(train.to_numpy())),
                                            batch_size=batch_size, shuffle=True) # Dataloader for training dataset

        optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=5e-5) # Generator optimizer
        optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-5) # Discriminator optimizer

        for parameter in self.discriminator.parameters():
            parameter.register_hook(
                lambda grad: grad / max(1, grad.data.norm(2)) # Clip norms as suggested in DP-SGD on a per-gradient basis
            )

            parameter.register_hook(
                lambda grad: grad + (1 / batch_size) * (clip_coeff*sigma) * torch.randn(parameter.shape).to(self.device) # Add noise on a per-gradient basis
            )

        epsilon = 0
        steps = 0
        curr_epoch = 0

        while epsilon < self.target_epsilon:
            # while the privacy budget is not expended
            for i, data in enumerate(data_loader, 0):
                # load in a batch of data
                train_loss_D = 0
                train_loss_G = 0

                ##########################
                # (1) Update discriminator
                ##########################

                self.discriminator.zero_grad()
                real_data = data[0].to(self.device) #Sample real data
                b_size = real_data.size(0)
                noise = torch.randn(b_size, self.latent_size, device=self.device) #sample from prior
                fake_data  = self.generator(noise).detach()
                loss_D = -(torch.mean(self.discriminator(real_data)) - torch.mean(self.discriminator(fake_data))) # Wasserstein loss

                loss_D.backward() #backpropagate loss
                optimizer_D.step() #step discriminator

                for parameter in self.discriminator.parameters():
                    parameter.data.clamp_(clamp_floor, clamp_ceil) # Needed for Wasserstein GANs - clamp parameters to hypercube

                steps += 1

                if i % nc == 0:
                    #######################
                    # (2) Update generator
                    #     every 'nc' critic
                    #     iterations
                    #######################

                    optimizer_G.zero_grad()
                    gen_data = self.generator(noise)
                    loss_G = -torch.mean(self.discriminator(gen_data)) # Generator loss

                    loss_G.backward() # Backpropagate loss
                    optimizer_G.step() #Run optimizer

            ############################
            # (3) Calculate spent budget
            ############################

            curr_epoch += 1
            max_lmbd = 20
            lmbds = range(2, max_lmbd + 1)
            rdp = compute_rdp(batch_size / train.shape[0], sigma, steps, lmbds) #USES EXTERNAL LIBRARY FUNCTION
            epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
            print("Epoch :", curr_epoch, "Loss D : ", loss_D.item(), "Loss G : ", loss_G.item(), "Epsilon : ", epsilon)

    def generate_synthetics(self, num, fname, plot_hists):
        input_noise = torch.randn(num, self.latent_size, device=self.device)
        with torch.no_grad():
            generated = self.generator(input_noise).cpu()

        df_syn = generated.numpy()
        np.savetxt(fname, df_syn, delimiter=',')

        if plot_hists == True:
            bins = [i for i in range(-18,10,1)]
            df_orig = pd.read_csv('../../datasets/lifesci.csv', sep=',', header=0, names=[0,1,2,3,4,5,6,7,8,9,10])
            fig, axs = plt.subplots(11, tight_layout=True, figsize=(8,44))

            for i in range(11):
                axs[i].hist(df_syn[[i]], bins=bins, alpha=0.75, label='Synthetic')
                axs[i].hist(df_orig[[i]], bins=bins, alpha=0.75, label='Original')
                axs[i].title.set_text('Histogram for attribute '+str(i+1))
                axs[i].set_xlabel('Value')
                axs[i].set_ylabel('Count')
                axs[i].legend()

            plt.savefig('gen_hists.png', facecolor='w')
