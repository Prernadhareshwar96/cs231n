import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class View_fun(nn.Module):
        def __init__(self):
            super(View_fun, self).__init__()
        def forward(self, x):
            return x.view(-1)
 
use_tensorboard = True
if use_tensorboard == True:
    from tensorboardX import SummaryWriter
    board = SummaryWriter()

Generator = nn.Sequential(
    nn.Conv2d(3, 1024, 3, stride = 1, padding = 1),
    nn.BatchNorm2d(1024),
    nn.ReLU(),
    nn.Conv2d(1024, 512, 5, stride = 1, padding = 2),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.Conv2d(512, 256, 5, stride = 1, padding = 2),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Conv2d(256, 128, 5, stride = 1, padding = 2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 64, 5, stride = 1, padding = 2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 32, 5, stride = 1, padding = 2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 16, 3, stride = 1, padding = 1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 8, 3, stride = 1, padding = 1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.Conv2d(8, 3, 3, stride = 1, padding = 1),
)

Discriminator = nn.Sequential(
    nn.Conv3d(3, 32, 3, stride = 1, padding = 1),
    nn.MaxPool3d(3),
    nn.BatchNorm3d(32),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(32, 64, 3, stride = 1, padding = 1),
    nn.MaxPool3d(3),
    nn.BatchNorm3d(64),
    nn.LeakyReLU(0.2, True),
    nn.Conv3d(64, 128, 3, stride = 1, padding = 1),
    nn.MaxPool3d(3),
    nn.BatchNorm3d(128),
    nn.LeakyReLU(0.2, True),
    nn.MaxPool3d(2),
    View_fun(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

generator = generator.to(device)
discriminator = discriminator.to(device)

X = {}
loc = "Users/prerna/Documents/Q3/CS231n/data/cartwheel"
f_list = []
for filename in os.listdir(loc):
    f_list.append(loc + '/' + filename)

lr_val = 0.00005
num_epochs = 500
optim_G = torch.optim.RMSprop(Generator.parameters(), lr = lr_val)
optim_D = torch.optim.RMSprop(Discriminator.parameters(), lr = lr_val)

loss_gen = []
loss_dis = []

ctr = 0
t = 0
for n in range(num_epochs):
    for f in f_list:
        ctr += 1
        print(ctr)
        X_full = np.loadtxt(f)
        X_full = torch.tensor(X_full).to(device, dtype=torch.float)
        X_full = X_full.view(X_full.shape[0], 64, 64, 3)
        X_full = X_full.permute(0,3,1,2)
        fac_max = X_full.shape[0]/32
        if fac_max == 0:
            continue
        for idx in range(int(fac_max)):
            t += 1
            X = X_full[idx*32:(idx+1)*32]
            X_real = X.view(1, X.shape[0], X.shape[1], X.shape[2], X.shape[3])
            X_real = X_real.permute(0, 2, 3, 4, 1)
            Y_gen = Generator(X)
            Y_gen = Y_gen.view(1, Y_gen.shape[0], Y_gen.shape[1], Y_gen.shape[2], Y_gen.shape[3])
            Y_gen = Y_gen.permute(0, 2, 3, 4, 1)
            Y_dis = Discriminator(Y_gen)
            
            optim_D.zero_grad()
            loss_D = -torch.mean(torch.log(Discriminator(X_real))) - torch.mean(torch.log(1 - Y_dis))         
            loss_D.backward(retain_graph = True)
            optim_D.step()
            
            optim_G.zero_grad()
            loss_G = -torch.mean(torch.log(Y_dis))
            loss_G.backward()
            optim_G.step()
            
            loss_gen.append(loss_G)
            loss_dis.append(loss_D)
            
            if use_tensorboard:
                board.add_scalar('Generator Loss', loss_G, t)
                board.add_scalar('Discriminator loss', loss_D, t)

board.close() if use_tensorboard else None  # Closes tensorboard, else do nothing
path_parameter_G = "output/" + "latest_G_" +str(lr_val)+ '_'+ str(num_epochs) + ".pt"
path_parameter_D = "output/" + "latest_D_" +str(lr_val)+ '_'+ str(num_epochs) + ".pt"
path_loss_G = "output/" + "Loss_G_" +str(lr_val)+ '_'+ str(num_epochs) + ".csv"
path_loss_D = "output/" + "Loss_D_" +str(lr_val)+ '_'+ str(num_epochs) + ".csv"
np.savetxt(path_loss_G, np.array(loss_G))
np.savetxt(path_loss_D, np.array(loss_D))
torch.save(Generator.state_dict(), path_parameter_G)
torch.save(Discriminator.state_dict(), path_parameter_D)