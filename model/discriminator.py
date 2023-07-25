import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from config import Config

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(int(np.prod(Config.img_shape)), 512)
        self.lere1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.lere2 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.lere1(x)
        x = self.fc2(x)
        x = self.lere2(x)
        x = self.fc3(x)
        x = self.sig(x)

        return x
