import torch.nn as nn
'''
Class defining the Generator network for a GAN
__init__(): initialises the network and defines the forward pass
forward() : runs the forward pass for a given datum x
'''
class Generator(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Generator, self).__init__()

        l_s = latent_size # The Generator is not conditional
        o_s = output_size

        self.main = nn.Sequential(
            nn.Linear(l_s, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, o_s)
        )

    def forward(self, x):
        return self.main(x)

'''
Class defining the Discriminator network for a GAN
__init__(): initialises the network and defines the forward pass
forward() : runs the forward pass for a given datum x
'''
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        i_s = input_size

        self.main = nn.Sequential(
            nn.Linear(i_s, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.main(x)
