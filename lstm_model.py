import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import random
import torchvision

class View_fun(nn.Module):
        def __init__(self):
            super(View_fun, self).__init__()
        def forward(self, x):
            return x.view(x.shape[0], -1)
use_tensorboard = True
if use_tensorboard == True:
    from tensorboardX import SummaryWriter
    board = SummaryWriter()

            
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
save_path = '/home/prerna/cs231n/output/'
loc = "/home/prerna/data"
f_list = []
for filename in os.listdir(loc):
    f_list.append(loc+ '/' + filename)

batch_size = 8
num_batches = int(len(f_list)/batch_size)
batches = []
for i in range(num_batches):
    batches.append(f_list[i*batch_size:(i+1)*batch_size])
print(len(batches))

class ConvLSTM(nn.Module):
    def __init__(self, batch_size):
        super(ConvLSTM, self).__init__()
        self.batch_size = batch_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 8, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(8, 16, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(16, 32, 5, stride = 1, padding = 2)
        self.conv4 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.conv5 = nn.Conv2d(64, 128, 5, stride = 1, padding = 2)
        self.conv6 = nn.Conv2d(128, 256, 5, stride = 2, padding = 2)
        self.conv7 = nn.Conv2d(256, 512, 5, stride = 1, padding = 2)
        self.conv8 = nn.Conv2d(512, 1024, 8, stride = 1)
        self.rnn9 = nn.LSTM(input_size = 1024, hidden_size = 1024, batch_first = True)
        self.conv10 = nn.ConvTranspose2d(1024, 512, 5, stride = 1)
        self.conv11 = nn.ConvTranspose2d(512, 256, 5, stride = 2)#13
        self.conv12 = nn.ConvTranspose2d(256, 128, 5, stride = 1)#17
        self.conv13 = nn.ConvTranspose2d(128, 64, 5, stride = 1)#21
        self.conv14 = nn.ConvTranspose2d(64, 32, 5, stride = 1)#25
        self.conv15 = nn.ConvTranspose2d(32, 16, 5, stride = 1)#29
        self.conv16 = nn.ConvTranspose2d(16, 8, 5, stride = 1)#33
        self.conv17 = nn.ConvTranspose2d(8, 3, 3, stride = 2, padding = 2, output_padding = 1)#64
        
    def forward(self, x):
#         Assuming input of size (32,3,64,64)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        revert_shape = x.shape
        x = x.view(self.batch_size,32,-1)
        x = self.rnn9(x)
        x = self.relu(x)
        x = x.view(revert_shape)
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.relu(self.conv17(x))
        x = self.sigmoid(x)
        return Output

        
        return out