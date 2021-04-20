import argparse
import time
import torch
import pickle
import numpy as np
import itertools
# from viz import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cuda = False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
def sample_image(n_row):
    z = Variable(Tensor(np.random.normal(0,1,(n_row**2, 10))))
    print(z.shape)
    z_cat = Tensor(np.array([1,0,0,0,0,0,0,0,0,0]))
    

def sample_categorical(batch_size, n_classes=10):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.autograd.Variable with the sample
    '''
    cat = np.random.randint(0, 10, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat

def get_categorical(labels, n_classes=10):
    cat = np.array(labels.data.tolist())
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat

def classification_accuracy(Q, data_loader):
    Q.eval()
    labels = []
    test_loss = 0
    correct = 0
    cuda = True
    for batch_idx, (X, target) in enumerate(data_loader):
        # X = X * 0.3081 + 0.1307
        # X.resize_(data_loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        labels.extend(target.data.tolist())
        # Reconstruction phase
        output = Q(X)[1]

        test_loss += F.nll_loss(output, target).item()

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader)
    return 100. * correct / len(data_loader.dataset)



if __name__ == "__main__":
    a = sample_categorical(5, n_classes=10)
    print(a.shape)
    sample_image(10)


    z_cat = Tensor(np.array([1,0,0,0,0,0,0,0,0,0]))
    hat = z_cat.repeat(2,1)
    print(hat)