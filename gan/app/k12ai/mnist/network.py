#!/usr/bin/python3
# -*- coding: utf-8 -*-

from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_shapes=(1, 1, 28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_shapes[1], 64, 5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Flatten(),
            nn.Linear(128 * (in_shapes[2] // 4) * (in_shapes[3] // 4), 1),
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
            nn.Linear(latent_dim, 256 * 7 * 7, bias=False),
            nn.BatchNorm1d(256 * 7 * 7),
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
