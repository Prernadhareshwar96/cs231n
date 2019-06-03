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

X_traing = torch.zeros((1, batch_size, 3, 64, 64)).cuda()
X_traind = torch.zeros((1, batch_size, 3, 64, 64, 32)).cuda()
for b in tqdm(batches):
    X_gen = torch.zeros((1, 3, 64, 64)).cuda()
    X_dis = torch.zeros((1, 3, 64, 64, 32)).cuda()
    for f_name in b:
        X_full = np.loadtxt(f_name)
        X_full = torch.tensor(X_full).to(device, dtype=torch.float)
        X_full = X_full.view(X_full.shape[0], 64, 64, 3)
        X_full = X_full.permute(0,3,1,2)
        X_full = X_full[10:42, :, :, :]
        if X_full.shape[0] != 32:
            continue
        X_gen = torch.cat((X_gen, X_full[0:1, :, :, :]), dim = 0)
        X_real = X_full.permute(1,2,3,0).unsqueeze(0)
        X_dis = torch.cat((X_dis, X_real), dim = 0)
    if (X_gen.shape[0]) != batch_size+1:
        continue
    X_traing = torch.cat((X_traing, X_gen[1:X_gen.shape[0],:,:,:].unsqueeze(0)), dim = 0)
    X_traind = torch.cat((X_traind, X_dis[1:X_dis.shape[0],:,:,:,:].unsqueeze(0)), dim = 0)
X_traing = X_traing[1:X_traing.shape[0],:,:,:,:]/255.
X_traind = X_traind[1:X_traind.shape[0],:,:,:,:,:].permute(0,1,2,5,3,4)/255. 
        
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.relu = nn.LeakyReLU(0.2, False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(3, 16, 7, stride = 1, padding = 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 128, 7, stride = 2, padding = 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 7, stride = 1, padding = 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 7, stride = 2, padding = 3)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 7, stride = 1, padding = 3)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 5, stride = 2, padding = 2)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, 5, stride = 1, padding = 2)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1024, 4, stride = 4)#2,2
        self.bn8 = nn.BatchNorm2d(1024)
        self.conv10 = nn.ConvTranspose2d(1024, 512, 5, stride = 1)#6, 6
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.ConvTranspose2d(512, 512, 5, stride = 2,)#15, 15
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.ConvTranspose2d(512, 256, 5, stride = 1)#19, 19
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.ConvTranspose2d(256, 128, 5, stride = 1)#23, 23
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.ConvTranspose2d(128, 32, 5, stride = 1)#27, 27
        self.bn14 = nn.BatchNorm2d(32)
        self.conv15 = nn.ConvTranspose2d(32, 16, 5, stride = 1)#31, 31
        self.bn15 = nn.BatchNorm2d(16)
        self.conv16 = nn.ConvTranspose2d(16, 8, 3, stride = 1)#33, 33
        self.bn16 = nn.BatchNorm2d(8)
        self.conv17 = nn.ConvTranspose2d(8, 3, 3, stride = 2, padding = 2, output_padding = 1)#64, 64
        self.bn17 = nn.BatchNorm2d(3)
        self.conv18 = nn.ConvTranspose2d(3, 3, 3, stride = 1, padding = 1)#64, 64
       
    def forward(self, x):
        Output = torch.zeros((x.shape[0],3,32,64,64))
        for i in range(32):
    #         print(x.shape)
            out = self.relu(self.bn1(self.conv1(x)))
    #         print(out.shape)
            out = self.relu(self.bn2(self.conv2(out)))
    #         print(out.shape)
            out = self.relu(self.bn3(self.conv3(out)))
    #         print(out.shape)
            out = self.relu(self.bn4(self.conv4(out)))
    #         print(out.shape)
            out = self.relu(self.bn5(self.conv5(out)))
    #         print(out.shape)
            out = self.relu(self.bn6(self.conv6(out)))
    #         print(out.shape)
            out = self.relu(self.bn7(self.conv7(out)))
    #         print(out.shape)
            out = self.relu(self.bn8(self.conv8(out)))
    #         print(out.shape)
            out = self.relu(self.bn10(self.conv10(out)))
    #         print(out.shape)
            out = self.relu(self.bn11(self.conv11(out)))
    #         print(out.shape)
            out = self.relu(self.bn12(self.conv12(out)))
    #         print(out.shape)
            out = self.relu(self.bn13(self.conv13(out)))
    #         print(out.shape)
            out = self.relu(self.bn14(self.conv14(out)))
    #         print(out.shape)
            out = self.relu(self.bn15(self.conv15(out)))
    #         print(out.shape)
            out = self.relu(self.bn16(self.conv16(out)))
    #         print(out.shape)
            out = self.relu(self.bn17(self.conv17(out)))
    #         print(out.shape)
            out = self.sigmoid(out)   
            Output[:,:,i,:,:] = out
            x = out
        return Output

Discriminator = nn.Sequential(
    nn.Conv3d(3,16, 3, stride = 1, padding = 1),
    nn.BatchNorm3d(16),
    nn.LeakyReLU(0.2, False),
    nn.AvgPool3d(2),
    nn.Conv3d(16, 32, 5, stride = 1, padding = 1),
    nn.BatchNorm3d(32),
    nn.LeakyReLU(0.2, False),
    nn.AvgPool3d(2),
    nn.Conv3d(32, 128, 3, stride = 1, padding = 1),
    nn.BatchNorm3d(128),
    nn.LeakyReLU(0.2, False),
    nn.AvgPool3d(2),
    nn.Conv3d(128, 128, 3, stride = 1, padding = 1),
    nn.BatchNorm3d(128),
    nn.LeakyReLU(0.2, False),
    nn.Conv3d(128, 256, 3, stride = 1, padding = 1),
    nn.BatchNorm3d(256),
    nn.LeakyReLU(0.2, False),
    nn.AvgPool3d(3),
    nn.Conv3d(256, 512, 3, stride = 1, padding = 1),
    nn.BatchNorm3d(512),
    nn.LeakyReLU(0.2, True),
#     nn.Conv3d(512, 1024, 3, stride = 2, padding = 1),
#     nn.BatchNorm3d(1024),
#     nn.LeakyReLU(0.2, True),
    View_fun(),
    nn.Linear(2048, 1),
#     nn.Linear(100, 1),
    nn.Sigmoid(),
)

lr_gen = 0.002
lr_dis = 0.0009
num_epochs = 7000
Generator = Gen().cuda()
Discriminator = Discriminator.cuda()
Generator.load_state_dict(torch.load(save_path + "param_inter_G_300.pt"))
Discriminator.load_state_dict(torch.load(save_path + "param_inter_D_300.pt"))
optim_G = torch.optim.Adam(Generator.parameters(), lr = lr_gen)
optim_D = torch.optim.Adam(Discriminator.parameters(), lr = lr_dis)

loss_gen = []
loss_dis = []
grad_norm_G = []
grad_norm_D = []
batch_size = 8
ctr = 0
for n in tqdm(range(301, num_epochs)):
    if float(n+1)%50 == 0:
        print("Saved")
        path_parameter_G = save_path + "param_inter_G_" + str(n+1) + ".pt"
        path_parameter_D = save_path + "param_inter_D_" + str(n+1) + ".pt"
        torch.save(Generator.state_dict(), path_parameter_G)
        torch.save(Discriminator.state_dict(), path_parameter_D)
        path_loss_G = save_path + "loss_latest_G.csv"
        path_loss_D = save_path + "loss_latest_D.csv"
        np.savetxt(path_loss_G, np.array(loss_gen))
        np.savetxt(path_loss_D, np.array(loss_dis))
    num_train = X_traing.shape[0]
    for t in tqdm(range(int(num_train))):
      #if use_tensorboard:
       # board.add_scalar('Generator Loss', loss_G, ctr)
       # board.add_scalar('Discriminator loss', loss_D, ctr)
      X_gen = torch.cat((X_traing[t,:,:,:,:].unsqueeze(2), torch.rand((batch_size, 3, 31, 64, 64)).cuda()), dim = 2)
      X_dis = X_traind[t,:,:,:,:,:]
      ctr += 1
      
      Y_gen = Generator(X_gen)
      Y_dis = Discriminator(Y_gen)
      Y_real = Discriminator(X_dis)
      nom = random.randint(0, 7)
      Y_plot = Y_gen[nom,:,:,:,:].permute(1,0,2,3).detach()
      grid_img = torchvision.utils.make_grid(Y_plot, nrow=8)
      board.add_image('Generator Output', grid_img, ctr)
      X_in = X_gen[nom,:,0,:,:].detach()
      board.add_image('Input', X_in, ctr)
      while (torch.mean(Y_real)-torch.mean(Y_dis)>0.6):
          Y_gen = Generator(X_gen)
          Y_dis = Discriminator(Y_gen)

          optim_G.zero_grad()
          loss_G = -torch.mean(torch.log(Y_dis))
          loss_G.backward()
          
          grad_norm_G.append(nn.utils.clip_grad_norm_(Generator.parameters(), 5))
          optim_G.step()
          loss_gen.append(loss_G)
      
      while (torch.mean(Y_real)-torch.mean(Y_dis)<0.3):
          Y_gen = Generator(X_gen)
          Y_dis = Discriminator(Y_gen)
          Y_real = Discriminator(X_dis)
#          print("Dis output real: {}".format(Y_real))
#          print("Dis output: {}".format(Y_dis))
          optim_D.zero_grad()
          loss_D = -torch.mean(torch.log(Y_real)) - torch.mean(torch.log(1 - Y_dis))     
          loss_D.backward()
          
          grad_norm_D.append(nn.utils.clip_grad_norm_(Discriminator.parameters(), 5))
          optim_D.step()
          loss_dis.append(loss_D)
          
      if ctr%1 == 0:
          Y_gen = Generator(X_gen)
          Y_dis = Discriminator(Y_gen)
          Y_real = Discriminator(X_dis)
#          print("Dis output real: {}".format(Y_real))
#          print("Dis output: {}".format(Y_dis))
          optim_D.zero_grad()
          loss_D = -torch.mean(torch.log(Y_real)) - torch.mean(torch.log(1 - Y_dis))     
          loss_D.backward()
          
          grad_norm_D.append(nn.utils.clip_grad_norm_(Discriminator.parameters(), 5))
          optim_D.step()
          loss_dis.append(loss_D)
          
      if ctr%1 == 0:
          Y_gen = Generator(X_gen)
          Y_dis = Discriminator(Y_gen)

          optim_G.zero_grad()
          loss_G = -torch.mean(torch.log(Y_dis))
          loss_G.backward()
          
          grad_norm_G.append(nn.utils.clip_grad_norm_(Generator.parameters(), 5))
          optim_G.step()
          loss_gen.append(loss_G)
          
board.close() if use_tensorboard else None  # Closes tensorboard
path_parameter_G = save_path+ "latest_G_" + '_'+ str(n) + '_' +".pt"
path_parameter_D = save_path + "latest_D_"+ '_'+ str(n) + '_' +".pt"
path_loss_G = save_path + "Loss_G_" + '_'+ str(n) + '_' + ".csv"
path_loss_D = save_path + "Loss_D_" + '_'+ str(n) + '_' + ".csv"
np.savetxt(path_loss_G, np.array(loss_gen))
np.savetxt(path_loss_D, np.array(loss_dis))
torch.save(Generator.state_dict(), path_parameter_G)
torch.save(Discriminator.state_dict(), path_parameter_D)
