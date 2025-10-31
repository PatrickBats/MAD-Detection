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
N_with_w = 6000
max_c = 2500
#Wall = [0,0.25,0.5,0.75,1,1.25,1.5, 1.75, 2]
Wall = [1.5,1.75,2]
generation_number = 20
n_epoch = 40
batch_size = 128
n_T = 500 # 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-4
save_model = True
save_dir = './data/t500/'


#### Extracting mnist data statistics

tf = transforms.Compose([transforms.ToTensor()])
dataset = MNIST("./data", train=True, download=True, transform=tf)
real_data = dataset.data.float()/255
real_t = dataset.targets
real_features = util.extract_mnist_features(real_data, device)
mu_all = np.mean(np.transpose(real_features), axis = 1)
cov_all = np.cov(np.transpose(real_features))

###########


x = torch.load( save_dir + f"gen_data_with_w_initial_w{0}" )
x = x.to(device)
x = x.float()/255
initial_features = util.extract_mnist_features(x, device)


mu_initial = np.mean(np.transpose(initial_features), axis = 1)
cov_initial = np.cov(np.transpose(initial_features))

FID0 = util.calculate_frechet_distance(mu_initial, cov_initial, mu_all, cov_all)
m = util.compute_prdc_slice(real_features, initial_features, 5, 20000, 1)

Precision0 = m['precision']
Recall0 = m['recall']
Density0 = m['density']
Coverage0 = m['coverage']



for w in Wall:
    
    
    Precision = [Precision0]
    Recall = [Recall0]
    Density = [Density0]
    Coverage = [Coverage0]
    FID = [FID0]
    
    
    
    
    
    for generation in range(1,generation_number):
        print(f'Generation={generation}_w={w}')
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
            torch.save(ddpm.state_dict(), save_dir + f"model_{generation}_w{w}.pth")
            opt_loss = copy.deepcopy(loss_ema)

        #######
        
        print('Generate Samples with w')
        
        X_with_w, C_with_w = util.generate_samples(ddpm, N = N_with_w, w = w, max_c = max_c)
        torch.save( (255*X_with_w).byte().detach().cpu(), save_dir+ f"gen_data_with_w{generation}_w{w}" )
        torch.save( C_with_w.cpu(),save_dir+ f"gen_index_with_w{generation}_w{w}" )
        print('Generate Samples without w')
        
        X_without_w, C_without_w = util.generate_samples(ddpm, N = N_no_w, w = 0, max_c = max_c)
        torch.save( (255*X_without_w).byte().detach().cpu(), save_dir+ f"gen_data_without_w{generation}_w{w}" )
        torch.save( C_without_w.cpu(),save_dir+ f"gen_index_without_w{generation}_w{w}" )
        
        #### extracting data features
        
        generated_features = util.extract_mnist_features(X_without_w, device)
        
        ### computing generated data features
        
        mu_gen = np.mean(np.transpose(generated_features), axis = 1)
        cov_gen = np.cov(np.transpose(generated_features))
        
        #### computing TSNE
        
        print('plotting tsne')
        
        util.mytsne(real_data[0:10000],X_without_w[0:10000], save_dir, generation, w)
        
        
        print('computing metrics')
        
        FID.append( util.calculate_frechet_distance(mu_gen, cov_gen, mu_all, cov_all) )
        
        plt.close('all')
        plt.plot(FID)
        plt.title(f"FID_generation={generation}_w={w}")
        with open(save_dir+ f"fid_w={w}", "wb") as fp:   #Pickling
            pickle.dump(FID, fp)
        plt.savefig(save_dir + f"Allgenrations_fid_w={w}.png")
        
        ### computing metrics
        
        m = util.compute_prdc_slice(real_features, generated_features, 5, 20000, 1)
        Precision.append(m['precision'])
        Recall.append(m['recall'])
        Density.append(m['density'])
        Coverage.append(m['coverage'])
        
        plt.close('all')
        plt.plot(Precision)
        plt.plot(Recall)
        plt.plot(Density)
        plt.plot(Coverage)
        plt.title(f"Metrics_generation={generation}_w={w}")
        plt.legend(['Precision', 'Recall', 'Density', 'Coverage'])
        with open(save_dir+ f"metrics_w={w}", "wb") as fp:   #Pickling
            pickle.dump([Precision,Recall,Density,Coverage], fp)
        plt.savefig(save_dir + f"Allgenrations_metrics_w={w}.png")
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
