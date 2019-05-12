import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class View_fun(nn.Module):
       def __init__(self):
            super(View, self).__init__()
        def forward(self, x):
            return x.view(-1) 

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
    nn.Linear(32*64*64*3, 1)
    nn.LogSigmoid()
)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


###############################################################################
################### Loading training and Validation data ###################### 
############################################################################### 

X = {}
loc = "Users/prerna/Documents/Q3/CS231n/data/cartwheel"
f_list = []
for filename in os.listdir(loc):
    f_list.append(loc + filename)
    
###############################################################################
################# Training the model for num_epochs epochs ####################
###############################################################################
g_train = 5
lr_val = 0.00005
generator = Generator()
discriminator = Discriminator()
optim_G = torch.optim.RMSprop(generator.parameters(), lr = lr_val)
optim_D = torch.optim.RMSprop(discriminator.parameters(), lr = lr_val)

for n in num_epochs:
    for f in f_list:
        X_full = np.loadtxt(f)
        X_full = torch.tensor(X).to(device)
        fac_max = X_full.shape[0]/32
        if fac_max == 0:
            continue
        for idx in range(fac_max):
            
            X = X_full[idx*32:(idx+1)*32]
            Y_gen = generator(X)
            Y_dis = discriminator(Y_gen)
            
            optim_D.zero_grad()
            loss_D = -torch.mean(discriminator(X)) + torch.mean(Y_dis)        
            loss_D.backward()
            optim_D.step()
            
            optim_G.zero_grad()
            loss_G = -torch.mean(Y_dis)
            loss_G.backward()
            optim_G.step()
            

 