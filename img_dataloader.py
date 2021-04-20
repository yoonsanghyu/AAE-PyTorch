import pickle
import torch
import torch.utils.data as data
import numpy as np


def load_data(data_path='data/'):
    print('loading data!')
    trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
    trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
    # Set -1 as labels for unlabeled data
    trainset_unlabeled.targets = torch.from_numpy(np.array([-1] * 47000))
    validset = pickle.load(open(data_path + "validation.p", "rb"))
    train_batch_size = 64
    valid_batch_size = 64
    train_labeled_loader = data.DataLoader(trainset_labeled, batch_size=train_batch_size, shuffle=True)
    train_unlabeled_loader = data.DataLoader(trainset_unlabeled, batch_size=train_batch_size, shuffle=True)
    valid_loader = data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)

    return train_labeled_loader, train_unlabeled_loader, valid_loader
    

if __name__ == "__main__":
    trainset_labeled = pickle.load(open('data/'+ "train_labeled.p", "rb"))
    print(trainset_labeled)
    