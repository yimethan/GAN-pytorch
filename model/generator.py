import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config

class Generator(nn.Module):
    def __init__(self, latent_dims=Config.latent_dims, momentum=0.8):
        super(Generator, self).__init__()
        self.latent_dims = latent_dims

        self.fc1 = nn.Linear(latent_dims, 128)
        self.lere1 = nn.LeakyReLU(0.2, inplace=True)

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, momentum)
        self.lere2 = nn.LeakyReLU(0.2, inplace=True)

        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512, momentum)
        self.lere3 = nn.LeakyReLU(0.2, inplace=True)

        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024, momentum)
        self.lere4 = nn.LeakyReLU(0.2, inplace=True)

        self.fc5 = nn.Linear(1024, int(np.prod(Config.img_shape)))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.lere1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.lere2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.lere3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.lere4(x)

        x = self.fc5(x)
        x = self.tanh(x)

        return x
