import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

class View_fun(nn.Module):
        def __init__(self):
            super(View_fun, self).__init__()
        def forward(self, x):
            return x.view(x.shape[0], -1) 

class Gen(nn.Module):
    def __init__(self, batch_size):
        super(Gen, self).__init__()
        self.batch_size = batch_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv3d(3, 8, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv3d(8, 16, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv3d(16, 32, 5, stride = 1, padding = 2)
        self.conv4 = nn.Conv3d(32, 64, 5, stride = 2, padding = 2)
        self.conv5 = nn.Conv3d(64, 128, 5, stride = 1, padding = 2)
        self.conv6 = nn.Conv3d(128, 256, 5, stride = 2, padding = 2)
        self.conv7 = nn.Conv3d(256, 512, 5, stride = 1, padding = 2)
        self.conv8 = nn.Conv3d(512, 1024, 4, stride = 4)#2,2,1
        self.conv10 = nn.ConvTranspose3d(1024, 512, 5, stride = 1)#6, 6, 5
        self.conv11 = nn.ConvTranspose3d(512, 256, 5, stride = 2, padding = (0, 0, 3))#15, 15, 7
        self.conv12 = nn.ConvTranspose3d(256, 128, 5, stride = 1)#19, 19, 11
        self.conv13 = nn.ConvTranspose3d(128, 64, 5, stride = 1)#23, 23, 15
        self.conv14 = nn.ConvTranspose3d(64, 32, 5, stride = 1)#27, 27, 19
        self.conv15 = nn.ConvTranspose3d(32, 16, 5, stride = 1)#31, 31, 23
        self.conv16 = nn.ConvTranspose3d(16, 8, 3, stride = 1, padding = (0,0,4))#33, 33, 17
        self.conv17 = nn.ConvTranspose3d(8, 3, 3, stride = 2, padding = 2, output_padding = 1)#64, 64, 32
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.relu(self.conv17(x))
        x = 255*self.sigmoid(x)
        return x

Discriminator = nn.Sequential(
    nn.Conv3d(3, 16, 3, stride = 2, padding = 1),
    nn.BatchNorm3d(16),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(16, 32, 5, stride = 2, padding = 1),
    nn.BatchNorm3d(32),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(32, 64, 3, stride = 2, padding = 1),
    nn.BatchNorm3d(64),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(64, 128, 5, stride = 2, padding = 1),
    nn.BatchNorm3d(128),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(128, 256, 3, stride = 1, padding = 1),
    nn.BatchNorm3d(256),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(256, 512, 3, stride = 1, padding = 1),
    nn.BatchNorm3d(512),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(512, 1024, 3, stride = 2, padding = 1),
    nn.BatchNorm3d(1024),
    nn.LeakyReLU(0.2, True),
    View_fun(),
    nn.Linear(9216, 100),
    nn.Linear(100, 1),
    nn.Sigmoid(),
)

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.relu = nn.LeakyReLU(0.01, False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(3, 8, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(8, 16, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(16, 32, 5, stride = 1, padding = 2)
        self.conv4 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.conv5 = nn.Conv2d(64, 128, 5, stride = 1, padding = 2)
        self.conv6 = nn.Conv2d(128, 256, 5, stride = 2, padding = 2)
        self.conv7 = nn.Conv2d(256, 512, 5, stride = 1, padding = 2)
        self.conv8 = nn.Conv2d(512, 256, 8, stride = 1)#256, 1, 1
        self.conv10 = nn.ConvTranspose3d(256, 128, 5, stride = 1)#2, 1, 1
        self.conv11 = nn.ConvTranspose3d(128, 64, 5, stride = 2, padding = (3, 0, 0))#1, 5, 5
        self.conv12 = nn.ConvTranspose3d(64, 32, 5, stride = 2, padding = (2, 0, 0))#1, 13, 13
        self.conv13 = nn.ConvTranspose3d(32, 16, 5, stride = 1, padding = (2, 0, 0))#1, 29, 29
        self.conv14 = nn.ConvTranspose3d(16, 8, 5, stride = 1, padding = (1, 0, 0))#3, 33, 33
        self.conv15 = nn.ConvTranspose3d(8, 3, 5, stride = 2, padding = (1, 3, 3), output_padding = (0, 1, 1))#3, 64, 64
        
    def forward(self, x):
#         original = x[:,:,:,:,:1]
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.relu(self.conv5(out))
        out = self.relu(self.conv6(out))
        out = self.relu(self.conv7(out))
        out = self.relu(self.conv8(out))
        print(out.shape)
        out = torch.cat((out.unsqueeze(2), torch.randn((out.shape[0], out.shape[1], out.shape[2], out.shape[3])).unsqueeze(2).cuda()), dim = 2)
        print(out.shape)
        out = self.relu(self.conv10(out))
        out = self.relu(self.conv11(out))
        out = self.relu(self.conv12(out))
        out = self.relu(self.conv13(out))
        out = self.relu(self.conv14(out))
        out = self.relu(self.conv15(out))
        out = self.relu(self.conv16(out))
        out = self.relu(self.conv17(out))
        out = self.tanh(out) + original
        return out


