from __future__ import print_function
import pickle
import numpy as np
import torch
from torchvision import datasets, transforms
from sub import subMNIST

transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

trainset_original = datasets.MNIST('data', train=True, download=True,
                                   transform=transform)

train_label_index = []
valid_label_index = []
for i in range(10):
    train_label_list = trainset_original.train_labels.numpy()
    label_index = np.where(train_label_list == i)[0]
    label_subindex = list(label_index[:300])
    valid_subindex = list(label_index[300: 1000 + 300])
    train_label_index += label_subindex
    valid_label_index += valid_subindex

trainset_np = trainset_original.train_data.numpy()
trainset_label_np = trainset_original.train_labels.numpy()
train_data_sub = torch.from_numpy(trainset_np[train_label_index])
train_labels_sub = torch.from_numpy(trainset_label_np[train_label_index])

trainset_new = subMNIST(root='data', train=True, download=True, transform=transform, k=3000)
trainset_new.data = train_data_sub.clone()
trainset_new.targets = train_labels_sub.clone()

pickle.dump(trainset_new, open("data/train_labeled.p", "wb"))


validset_np = trainset_original.train_data.numpy()
validset_label_np = trainset_original.train_labels.numpy()
valid_data_sub = torch.from_numpy(validset_np[valid_label_index])
valid_labels_sub = torch.from_numpy(validset_label_np[valid_label_index])


validset = subMNIST(root='data', train=False, download=True, transform=transform, k=10000)
validset.data = valid_data_sub.clone()
validset.targets = valid_labels_sub.clone()

pickle.dump(validset, open("data/validation.p", "wb"))


train_unlabel_index = []
for i in range(60000):
    if i in train_label_index or i in valid_label_index:
        pass
    else:
        train_unlabel_index.append(i)

trainset_np = trainset_original.train_data.numpy()
trainset_label_np = trainset_original.train_labels.numpy()
train_data_sub_unl = torch.from_numpy(trainset_np[train_unlabel_index])
train_labels_sub_unl = torch.from_numpy(trainset_label_np[train_unlabel_index])

trainset_new_unl = subMNIST(root='data', train=True, download=True, transform=transform, k=47000)
trainset_new_unl.data = train_data_sub_unl.clone()
trainset_new_unl.targets = None      # Unlabeled

trainset_new_unl.targets

pickle.dump(trainset_new_unl, open("data/train_unlabeled.p", "wb"))
