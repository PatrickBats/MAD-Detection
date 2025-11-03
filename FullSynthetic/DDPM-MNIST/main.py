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



N_eval = 1000
N_next = 60000
max_c = 500

w = 0
generation_number = 20
n_epoch = 1
batch_size = 32
n_T = 500 # 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-4
save_model = True
save_dir = './data/diffusion_outputs10/'


#### Extracting mnist data statistics

tf = transforms.Compose([transforms.ToTensor()])
dataset = MNIST("./data", train=True, download=True, transform=tf)
real_data = dataset.data.float()/255
real_t = dataset.targets
real_features = util.extract_mnist_features(real_data, device)
mu_all = np.mean(np.transpose(real_features), axis = 1)
cov_all = np.cov(np.transpose(real_features))

###########




Precision = []
Recall = []
Density = []
Coverage = []
FID = []


for generation in range(generation_number):
    print(f'Generation={generation}')
    #### Loading dataset
    dataloader = util.load_datasets(generation,save_dir, w, batch_size)
    ## Creating model
    ddpm = util.DDPM(nn_model=util.ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T= n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    opt_loss = 1
    pbar = tqdm(range(n_epoch),leave = True)
    for ep in pbar:
        ddpm.train()
    
        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-0.8*ep/n_epoch)
    
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
            pbar.set_description(f"loss={opt_loss}")
            optim.step()
            
        if save_model and loss_ema < opt_loss:
            torch.save(ddpm.state_dict(), save_dir + f"model_{generation}_w{w}.pth")
            opt_loss = copy.deepcopy(loss_ema)
            
    
    print('Generate Samples with w')
    
    X_with_w, C_with_w = util.generate_samples(ddpm, N = N_next, w = w, max_c = max_c)
    torch.save( (255*X_with_w).byte().detach().cpu(), save_dir+ f"gen_data_with_w{generation}_w{w}" )
    torch.save( C_with_w.cpu(),save_dir+ f"gen_index_with_w{generation}_w{w}" )
    print('Generate Samples without w')
    
    X_without_w, C_without_w = util.generate_samples(ddpm, N = N_eval, w = 0, max_c = max_c)
    torch.save( (255*X_without_w).byte().detach().cpu(), save_dir+ f"gen_data_without_w{generation}_w{w}" )
    torch.save( C_without_w.cpu(),save_dir+ f"gen_index_without_w{generation}_w{w}" )
    
    #### extracting data features
    
    generated_features = util.extract_mnist_features(X_without_w, device)
    
    ### computing generated data features
    
    mu_gen = np.mean(np.transpose(generated_features), axis = 1)
    cov_gen = np.cov(np.transpose(generated_features))
    
    
    print('computing metrics')
    
    FID.append( util.calculate_frechet_distance(mu_gen, cov_gen, mu_all, cov_all) )
    
    ### computing metrics
    
    m = util.compute_prdc_slice(real_features, generated_features, 5, 20000, 1)
    Precision.append(m['precision'])
    Recall.append(m['recall'])
    Density.append(m['density'])
    Coverage.append(m['coverage'])
    
    plt.plot(Precision)
    plt.plot(Recall)
    plt.plot(Density)
    plt.plot(Coverage)
    plt.legend(['Precision', 'Recall', 'Density', 'Coverage'])
    plt.show()
    
    #### computing TSNE
    
    print('plotting tsne')
    
    util.mytsne(real_data[0:10000],X_without_w[0:10000], save_dir, generation, w)
    
    
    
    
    
    
    
    
    
    
    
    
    
    