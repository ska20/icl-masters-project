import torch.nn as nn

'''
Weights init initializes the network weights according to the
Xavier uniform distribution.
'''
def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
