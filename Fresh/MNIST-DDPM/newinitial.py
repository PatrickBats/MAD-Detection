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



N_no_w = 30000
N_with_w_all  = [10000]
N_real = 5000
max_c = 4000
w = 0
generation_number = 1
n_epoch = 1000
batch_size = 128
n_T = 500 # 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-4

save_model = True
save_dir = './data/diffusion_outputs10/'
save_dir2 = './data/N2000/'
resam = True

#### Extracting mnist data statistics

tf = transforms.Compose([transforms.ToTensor()])
dataset = MNIST("./data", train=True, download=True, transform=tf)

a =torch.ones((4000,10))
for i in range(10):
    a[:,i] = a[:,i]*i

a = a.long()

b = torch.zeros((4000,10))
X = dataset.data
C = dataset.targets
D = list(range(C.size()[0]))
for i in range(10):
    temp = torch.where(C == i)
    b[:,i] = temp[0][0:4000]
b =  b.long()
b = b.view(-1)
a = a.view(-1)

X = X[b]
C = a

real_data = X
real_t = C







real_features = util.extract_mnist_features(X.float()/255, device)
mu_all = np.mean(np.transpose(real_features), axis = 1)
cov_all = np.cov(np.transpose(real_features))

   



    
###########

for N_with_w in N_with_w_all:
    
    
    Precision = []
    Recall = []
    Density = []
    Coverage = []
    FID = []
    
    
    
    
    for generation in range(generation_number):
        print(f'Generation={generation}_N={N_with_w}')
        #### Loading dataset
        dataloader = util.load_datasets(generation,save_dir2, w, batch_size, real_data, real_t, N_real, N_with_w ,resam,0)
        ## Creating model
        ddpm = util.DDPM(nn_model=util.ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T= n_T, device=device, drop_prob=0.1)
        ddpm.to(device)
        print(dataloader.dataset.data.size())
        optim = torch.optim.Adam(ddpm.parameters(), lr=lrate, weight_decay = 0)
        #optim = torch.optim.SGD(ddpm.parameters(),lr = lrate, momentum = 0.9)
        opt_loss = 1
        pbar = tqdm(range(n_epoch),leave = True)
        Fid_temp = []
        loss_best = 10000
        fid_best = 1000
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
                pbar.set_description(f"loss: {loss_ema:.4f} loss_best: {loss_best:.4f}")
                #print('whatsup')
                optim.step()
            if loss_ema < loss_best:
                loss_best = loss_ema
                torch.save(ddpm.state_dict(), save_dir2 + f"model_{generation}_N{N_with_w}.pth")
                    
                
            if ep%100 == 99:
                
                ddpmbest = util.DDPM(nn_model=util.ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T= n_T, device=device, drop_prob=0.1)
                
                ddpmbest.to(device)
                
                ddpmbest.load_state_dict( torch.load(save_dir2 + f"model_{generation}_N{N_with_w}.pth"))
                
                X_without_w, C_without_w = util.generate_samples(ddpmbest, N = 10000, w = 0, max_c = max_c)
                
                generated_features = util.extract_mnist_features(X_without_w, device)
                mu_gen = np.mean(np.transpose(generated_features), axis = 1)
                cov_gen = np.cov(np.transpose(generated_features))
                Fid_temp.append( util.calculate_frechet_distance(mu_gen, cov_gen, mu_all, cov_all) )
                print(Fid_temp[-1])
                loss_best = 10
                if Fid_temp[-1] < fid_best:
                    fid_best = Fid_temp[-1]
                    torch.save(ddpmbest.state_dict(), save_dir2 + f"best_model_{generation}_N{N_with_w}.pth")
                    
                    
        
        with open(save_dir2+ f"fid_N={N_with_w}_generation={generation}", "wb") as fp:   #Pickling
            pickle.dump(FID, fp)
        
                
        ####### upload best model
        
        ddpm = util.DDPM(nn_model=util.ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T= n_T, device=device, drop_prob=0.1)
        ddpm.to(device)
        
        ddpm.load_state_dict( torch.load(save_dir2 + f"best_model_{generation}_N{N_with_w}.pth"))
        
        
        #######
        
        print('Generate Samples with w')
        
        X_with_w, C_with_w = util.generate_samples(ddpm, N = N_with_w, w = w, max_c = max_c)
        torch.save( (255*X_with_w).byte().detach().cpu(), save_dir2+ f"gen_data_with_w_N_real{N_real}_w{w}" )
        torch.save( C_with_w.cpu(),save_dir2+ f"gen_index_with_w_N_real{N_real}_w{w}" )
        print('Generate Samples without w')
        
        X_without_w, C_without_w = util.generate_samples(ddpm, N = N_no_w, w = 0, max_c = max_c)
        torch.save( (255*X_without_w).byte().detach().cpu(), save_dir2+ f"gen_data_without_w_N_real{N_real}_w{w}" )
        torch.save( C_without_w.cpu(),save_dir2+ f"gen_index_without_w_N_real{N_real}_w{w}" )
        
        #### extracting data features
        
        generated_features = util.extract_mnist_features(X_without_w, device)
        
        ### computing generated data features
        
        mu_gen = np.mean(np.transpose(generated_features), axis = 1)
        cov_gen = np.cov(np.transpose(generated_features))
        
        #### computing TSNE
        
        #print('plotting tsne')
        
        #util.mytsne(real_data[0:10000].float()/255,X_without_w[0:10000], save_dir2, generation, w)
        
        
        print('computing metrics')
        
        FID.append( util.calculate_frechet_distance(mu_gen, cov_gen, mu_all, cov_all) )
        
        with open(save_dir2+ f"fid_N_real{N_real}_w{w}", "wb") as fp:   #Pickling
            pickle.dump(FID, fp)
        
        ### computing metrics
        
        m = util.compute_prdc_slice(real_features[0:N_no_w], generated_features, 5, 20000, 1)
        Precision.append(m['precision'])
        Recall.append(m['recall'])
        Density.append(m['density'])
        Coverage.append(m['coverage'])
        
      
        with open(save_dir2+ f"metrics_N_real{N_real}_w{w}", "wb") as fp:   #Pickling
            pickle.dump([Precision,Recall,Density,Coverage], fp)
        
