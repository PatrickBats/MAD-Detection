from sklearn.manifold import TSNE
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch.backends.cudnn as cudnn
import copy
import time
from torch.autograd import Variable

import metrics as util 
import pickle



N_no_w = 60000
N_with_w = 300000
max_c = 2500
Wall = [0]
generation_number = 20
n_epoch = 500
batch_size = 128
n_T = 40000 # 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-5
save_model = True
save_dir_all = ['./data/t500n60k/']


#### Extracting mnist data statistics

tf = transforms.Compose([transforms.ToTensor()])
dataset = MNIST("./data", train=True, download=True, transform=tf)
real_data = dataset.data.float()/255
real_t = dataset.targets
real_features = util.extract_mnist_features(real_data, device)
mu_all = np.mean(np.transpose(real_features), axis = 1)
cov_all = np.cov(np.transpose(real_features))

###########

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

ddpm = util.DDPM(nn_model=util.ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T= n_T, device=device, drop_prob=0.1)
ddpm.to(device)

optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
opt_loss = 1
pbar = tqdm(range(n_epoch),leave = True)
for ep in pbar:
    ddpm.train()
        
            # linear lrate decay
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        
    loss_ema = None
    for x, c in dataloader:
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)
        loss = ddpm(x, c, 1, 1, guide_w=0)
        loss = loss.sum()
        loss.backward()
        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
                
        
if save_model:
    opt_loss = copy.deepcopy(loss_ema)
    for save_dir in save_dir_all: 
        torch.save(ddpm.state_dict(), save_dir + "model_initial.pth")                


for w in Wall:
    print(f'Generate Samples with w = {w}')
    X, C = util.generate_samples(ddpm, N = N_with_w, w = w, max_c = max_c)
    for save_dir in save_dir_all:
        torch.save( (255*X).byte().detach().cpu(), save_dir+ "gen_data_initial" )
        torch.save( C.cpu(),save_dir+ "gen_index_initial" )







