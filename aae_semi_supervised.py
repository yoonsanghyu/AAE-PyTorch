import argparse
import os
import numpy as np 
import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from img_dataloader import load_data
from utills import *

#######################################################
# hyperparameter setting
#######################################################

parser = argparse.ArgumentParser("semi-supervised aae model by hyu")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes of image datasets")

parser.add_argument("--img_dir", type=str, default='image_semi_supervised', help="number of classes of image datasets")

args = parser.parse_args()
print(args)

# config cuda
cuda = torch.cuda.is_available()
img_shape = (args.channels, args.img_size, args.img_size)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#######################################################
# Define Networks
#######################################################

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lin_D = nn.Linear(512, args.latent_dim)
        self.lin_D_cat = nn.Sequential(nn.Linear(512, args.n_classes),nn.Softmax())


    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)

        z_gauss = self.lin_D(x)
        z_cat = self.lin_D_cat(x)

        return z_gauss, z_cat

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class Discriminator_category(nn.Module):
    def __init__(self):
        super(Discriminator_category, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


#######################################################
# Preparation
#######################################################

# data
train_labeled_loader = load_data('data/')[0]
train_unlabeled_loader = load_data('data/')[1]
valid_loader = load_data('data/')[2]

# define model
# 1) generator
encoder = Encoder()
decoder = Decoder()
# 2) discriminator for z and y(category)
discriminator = Discriminator()
discriminator_cat = Discriminator_category()

# loss
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# optimizer
# 1) basic optimizer for G(encoder-decoder) and D(discriminator) 
optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# 2) additional optimizer for E-semi(encoder for semi-supervised learning) and D_cat(Discriminator for category)
optimizer_E_semi = torch.optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D_cat = torch.optim.Adam(discriminator_cat.parameters(), lr=args.lr, betas=(args.b1, args.b2))

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    discriminator_cat.cuda()
    adversarial_loss.cuda()
    reconstruction_loss.cuda()


def sample_image(n_row, epoch, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    z = nn.Parameter(Tensor(np.random.normal(0,1,(n_row**2, args.latent_dim))))
    generated_imgs = decoder(z)
    save_image(generated_imgs.data, os.path.join(img_dir, "%depoch.png" % epoch), nrow = n_row, normalize = True)

#######################################################
# Training part
#######################################################

# training phase
for epoch in range(args.n_epochs):
    encoder.train()
    decoder.train()
    discriminator.train()
    discriminator_cat.train()
    for (x_l, idx_l), (x_u, idx_u) in zip(train_labeled_loader, train_unlabeled_loader):
        for x, idx in [(x_l, idx_l), (x_u, idx_u)]:
            if idx[0] == -1:
                labeled = False
            else:
                labeled = True
        
            if cuda:
                x = x.cuda()
                idx = idx.cuda()
            
            if labeled: # supervised phase
                optimizer_E_semi.zero_grad()
                _, fake_cat = encoder(x)
                class_loss = F.cross_entropy(fake_cat, idx)
                class_loss.backward()
                optimizer_E_semi.step()

            else: # unsupervised phase

                valid = nn.Parameter(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = nn.Parameter(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

                # 1) reconstruction + generator loss for gaussian and category
                optimizer_G.zero_grad()
                fake_z, fake_cat = encoder(x)
                decoded_x = decoder(fake_z)
                validity_fake_z = discriminator(fake_z)
                G_loss = 0.01*adversarial_loss(validity_fake_z, valid) + 0.01*adversarial_loss(discriminator_cat(fake_cat), valid) + 0.98*reconstruction_loss(decoded_x, x)
                G_loss.backward()
                optimizer_G.step()

                # 2) discriminator loss for gaussian
                optimizer_D.zero_grad()
                real_z = nn.Parameter(Tensor(np.random.normal(0,1,(x.shape[0], args.latent_dim))), requires_grad=False)
                real_loss = adversarial_loss(discriminator(real_z), valid)
                fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
                D_loss = 0.5*(real_loss + fake_loss)
                D_loss.backward()
                optimizer_D.step()

                # 3) discriminator loss for category 
                optimizer_D_cat.zero_grad()
                real_cat = nn.Parameter(sample_categorical(x.shape[0], n_classes=args.n_classes), requires_grad=False).cuda()
                real_cat_loss = adversarial_loss(discriminator_cat(real_cat), valid)
                fake_cat_loss = adversarial_loss(discriminator_cat(fake_cat.detach()), fake)
                D_cat_loss = 0.5*(real_cat_loss + fake_cat_loss)
                D_cat_loss.backward()
                optimizer_D_cat.step()

    train_acc = classification_accuracy(encoder, train_labeled_loader)
    val_acc = classification_accuracy(encoder, valid_loader)

    print(
            "[Epoch %d/%d] [G loss: %f] [D loss: %f] [D_cat loss: %f] [class_loss: %f] [train_acc: %f] [val_acc: %f]"
            % (epoch, args.n_epochs, G_loss.item(), D_loss.item(), D_cat_loss.item(), class_loss.item(), train_acc, val_acc)
         )            
        
    sample_image(n_row = 10, epoch=epoch, img_dir=args.img_dir)
