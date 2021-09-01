import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.backends.cudnn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        i_s = input_size # Specify NN input size


        self.main = nn.Sequential(
            nn.Linear(i_s, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    '''
    Forward function for the network.
    INPUT: x, a point of data
    OUTPUT: Network prediction
    '''
    def forward(self, x):
        return self.main(x) #Call forward function

    '''
    Trains network given data and hyper-parameters
    INPUT: loader, a torch dataloader
    INPUT: hyperparams, a dictionary of hyperparameters
    OUTPUT: None
    '''

    def train(self, loader, hyperparams):
        torch.backends.cudnn.benchmark = True # Supposedly improves speed
        print_every = 100 # Control how often information is printed
        batch_size = hyperparams.batch_size # assign batch size
        lr = hyperparams.lr # assign learning rate
        epochs = hyperparams.epochs  # assign number of training epochs

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)  # define optimizer


        for e in tqdm(range(epochs)):
            for t, (x, y) in enumerate(loader, 0):
                #load in a batch of data
                train_loss = 0
                x = x.to(device='cuda')  # move to device, e.g. GPU
                y = y.to(device='cuda')  # move to device, e.g. GPU
                scores = self.forward(x) # Call forward function
                loss = F.binary_cross_entropy(scores, y) # Calculate loss
                optimizer.zero_grad() # Zero gradients
                loss.backward() # Back-propagate
                optimizer.step() # Run optimizer


    '''
    Runs network given test data.
    INPUT: loader, a torch dataloader
    OUTPUT: AUROC score, corresponding to classifier performance
    '''
    def test(self, loader):
        num_correct = 0
        num_samples = 0
        scores = []
        with torch.no_grad(): # disable gradient updates
            for (x, y) in loader:
                x = x.to(device='cuda')  # move to device
                y = y.to(device='cuda')  # move to device
                scores = self.forward(x)
                roc_score = roc_auc_score(y.cpu(), scores.cpu())
                return roc_score
