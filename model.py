import torch
import numpy as np
import torch.nn as nn

class View(nn.Module):
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
#         self.conv1 = nn.Conv2d(3, 64, 3, stride = 1, padding = 2)

Video_gen = nn.Sequential(
    nn.Conv2d
    nn.BatchNorm2d
    nn.Conv2d
    nn.BatchNorm2d
    nn.Conv2d
    nn.BatchNorm2d
    nn.Conv2d
    nn.BatchNorm2d
    nn.Conv2d
    nn.BatchNorm2d
)