import torch
import numpy as np
import torch.nn as nn

class View_fun(nn.Module):
       def __init__(self):
            super(View, self).__init__()
        def forward(self, x):
            return x.view(-1) 

# class Video_Gen(nn.Module):
#     def __init__(self, H_in = 64, W_in = 64):
#         """
#         Defining the basic generator for the GAN
#         """
#         super(Video_Gen, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3, stride = 1, padding = 1)

Generator = nn.Sequential(
    nn.Conv2d(3, 1024, 3, stride = 1, padding = 1)),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(1024, 512, 5, stride = 1, padding = 2),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.Conv2d(512, 256, 5, stride = 1, padding = 2),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Conv2d(256, 128, 5, stride = 1, padding = 2)),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 64, 5, stride = 1, padding = 2)),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 32, 5, stride = 1, padding = 2)),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 16, 3, stride = 1, padding = 1)),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 8, 3, stride = 1, padding = 1)),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.Conv2d(8, 3, 3, stride = 1, padding = 1)),
)

Discriminator = nn.Sequential(
    nn.Conv3d(3, 64, 3, stride = 1, padding = 1),
    nn.Batchnorm3d(64),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(64, 128, 3, stride = 1, padding = 1),
    nn.Batchnorm3d(128),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(128, 256, 3, stride = 1, padding = 1),
    nn.Batchnorm3d(256),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(256, 512, 3, stride = 1, padding = 1),
    nn.Batchnorm3d(512),
    nn.LeakyReLU(0.2, True),
    View_fun()
    nn.Linear(32*64*64*3, 2)
)