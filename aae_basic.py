import argparse
import os
import numpy as np 
import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets

from img_dataloader import load_data

#######################################################
# hyperparameter setting
#######################################################

parser = argparse.ArgumentParser("Basic aae model by hyu")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")

parser.add_argument("--img_dir", type=str, default='image_basic', help="number of classes of image datasets")

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
            nn.Linear(512, args.latent_dim)
        )

        self.mu = nn.Linear(512, args.latent_dim)
        self.logvar = nn.Linear(512, args.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        # mu = self.mu(x)
        # logvar = self.logvar(x)
        # z = reparameterization(mu, logvar)
        return x


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


#######################################################
# Preparation part
#######################################################

# data
train_labeled_loader = load_data('data/')[1]

# define model
# 1) generator
encoder = Encoder()
decoder = Decoder()
# 2) discriminator
discriminator = Discriminator()

# loss
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# optimizer
optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    reconstruction_loss.cuda()


#######################################################
# Training part
#######################################################

def sample_image(n_row, epoch, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    z = Variable(Tensor(np.random.normal(0,1,(n_row**2, args.latent_dim))))
    generated_imgs = decoder(z)
    save_image(generated_imgs.data, os.path.join(img_dir, "%depoch.png" % epoch), nrow = n_row, normalize = True)


# training phase
for epoch in range(args.n_epochs):
    for i, (x, idx) in enumerate(train_labeled_loader):

        valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)
        
        if cuda:
            x = x.cuda()
        
        # 1) reconstruction + generator loss
        optimizer_G.zero_grad()
        fake_z = encoder(x)
        decoded_x = decoder(fake_z)
        validity_fake_z = discriminator(fake_z)
        G_loss = 0.001*adversarial_loss(validity_fake_z, valid) + 0.999*reconstruction_loss(decoded_x, x)
        G_loss.backward()
        optimizer_G.step()

        # 2) discriminator loss
        optimizer_D.zero_grad()
        real_z = Variable(Tensor(np.random.normal(0,1,(x.shape[0], args.latent_dim))))
        real_loss = adversarial_loss(discriminator(real_z), valid)
        fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
        D_loss = 0.5*(real_loss + fake_loss)
        D_loss.backward()
        optimizer_D.step()

    # print loss
    print(
            "[Epoch %d/%d] [G loss: %f] [D loss: %f]"
            % (epoch, args.n_epochs, G_loss.item(), D_loss.item())
         )
        
    sample_image(n_row = 10, epoch=epoch, img_dir=args.img_dir)


