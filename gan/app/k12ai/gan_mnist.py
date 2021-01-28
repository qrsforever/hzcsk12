#!/bin/python3

import os
import io
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = True
torch.manual_seed(8888);



class Discriminator(nn.Module):
    def __init__(self, in_shapes=(1, 1, 28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_shapes[1], 64, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(p=0.4),
            nn.Flatten(),
            nn.Linear(64*(in_shapes[2] // 4)*(in_shapes[3] // 4), 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.model(x)


class nn_Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256*7*7, bias=False),
            nn.BatchNorm1d(256*7*7),
            nn.LeakyReLU(),
            nn_Reshape(-1, 256, 7, 7),
            nn.Conv2d(256, 128, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


def train():
    num_epochs = 50
    print_freq = 50

    d_lr = 0.0001
    g_lr = 0.0002

    dataloader = DataLoader(train_dataset, batch_size=512, num_workers=4)

    d_optimizer = optim.Adam(modelD.parameters(), d_lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(modelG.parameters(), g_lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    modelD.cuda().train()
    modelG.cuda().train()

    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(dataloader):
            real_image = image.to('cuda')
            real_label = torch.ones_like(label.unsqueeze(dim=1), dtype=torch.float, device='cuda')
            
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            
            ########### train discriminator model ###########
            set_requires_grad(modelD, True)
            with torch.no_grad():
                latent = torch.rand(image.shape[0], latent_dim, dtype=torch.float)
                fake_image = modelG(latent.to('cuda'))
                fake_label = torch.zeros_like(real_label)
                x = torch.cat([real_image, fake_image], dim=0)
                y = torch.cat([real_label, fake_label], dim=0)
            y_hat = modelD(x)
            d_loss = d_criterion(y_hat, y)
            d_loss.backward()
            d_optimizer.step()
            
            ########### train generator model (frozen D network) ###########
            set_requires_grad(modelD, False)
            latent = torch.rand(image.shape[0], latent_dim, dtype=torch.float)
            fake_image = modelG(latent.to('cuda'))
            fake_label = modelD(fake_image)
            g_loss = criterion(fake_label, torch.ones_like(fake_label))
            g_loss.backward()
            g_optimizer.step()
            
            if i % print_freq == 0:
                acc = (y_hat.detach() > 0.5).eq(y).sum().cpu().numpy() / (2 * image.shape[0])
                dls = d_loss.item()
                gls = g_loss.item()
                d_losses.append(dls)
                g_losses.append(gls)
                print("epoch:", epoch, "batchIdx:", i, "acc:", acc, "d_loss:", dls, "g_loss:", gls)
