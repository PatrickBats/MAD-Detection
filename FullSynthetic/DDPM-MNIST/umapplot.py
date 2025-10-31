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
#import torchvision

import metrics as util 
import pickle





N_no_w = 60000
N_with_w = 6000
max_c = 2500
Wall = [0,0.25,0.5,0.75,1]
#Wall = [1.5,1.75,2]
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
real_features = util.extract_mnist_features(x, device)

print('hello')

Tmax = 50000
import umap
#generated_features = util.extract_mnist_features(real_data, device)
#X = generated_features[0:Tmax]
X = real_data[0:Tmax]
X = torch.squeeze(X)
X = X.view(X.size(0), -1)
X = X.cpu().numpy()
w = 0


Iteration = [1,4,9,19]
for iteration in Iteration:

    X_without_w = torch.load( save_dir + f"gen_data_without_w{iteration}_w{w}" )
    X_without_w = X_without_w.float()/255
    X_without_w = torch.Tensor(X_without_w)
    X_without_w = X_without_w.to(device)


    generated_features = util.extract_mnist_features(X_without_w, device)

                    ### computing generated data features
    mu_gen = np.mean(np.transpose(generated_features), axis = 1)
    cov_gen = np.cov(np.transpose(generated_features))

                    #### computing TSNE

            #torch.save( a.detach().cpu(), save_dir+ f"tsnedata_generation{generation}_w{w}" )


    X2 = X_without_w[0:Tmax]
    #X2 = generated_features[0:Tmax]
    X2 = torch.squeeze(X2)
    X2 = X2.view(X2.size(0), -1)



    X2 = X2.cpu().numpy()
    X = np.concatenate((X,X2), axis = 0)
    
print(np.shape(X))

#tsne = TSNE(n_components = 2, random_state=1)
#tres = tsne.fit_transform(X)
#tres = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.2, FP_ratio=2.0).fit_transform(X, init="pca")

tres = umap.UMAP(n_neighbors=10, min_dist=0.000001,densmap=False).fit_transform(X)

print(np.shape(tres))
#plt.close('all')
#colors = ['r']*X1.shape[0] + ['b']*X2.shape[0]
plt.scatter(tres[:,1], tres[:,0], s = 0.03)
#plt.title(f"All-genration={iteration} w={w}")
#plt.savefig(save_dir + f"All-genration={iteration}w={w}.png")

torch.save( tres, save_dir+ f"tsnedata_w{w}" )



import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

Tmax2 = 10000
Tmax = 50000
w = 0


I = [1,2,3,4]
for i in I:
    ALLG = [2,5,10,20]
    generation = ALLG[i-1]
    
    
    tres = torch.load(save_dir+ f"tsnedata_w{w}" )
    
    sns.set_style('whitegrid')
    font = {'family': 'serif', 'style': 'normal', 'size': 10}
    matplotlib.rc('font', **font)
    sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((0, 0))
    matplotlib.use('Agg')
    
    
    
    # Create a figure and axis object
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    
    # Plot the scatter plot
    
    colors = ['#1b9e77']*Tmax2 + ['red']*Tmax2
    
    
    
    scatter2 = ax.scatter(
            tres[0:Tmax2, 0],
            tres[0:Tmax2, 1],
            s=0.05,
            color = '#1b9e77',
            alpha=0.8,
            marker = '.',
    )
    
    
    
    
    scatter1 = ax.scatter(
            tres[i*Tmax:i*Tmax + Tmax2, 0],
            tres[i*Tmax:i*Tmax+ Tmax2, 1],
            s=0.05,
            color = 'red',
            alpha= 0.8,
            marker = '.',
    )
    red_patch = mpatches.Patch(color='#1b9e77', label='Real')
    Blue_patch = mpatches.Patch(color='red', label='Synthesized')
    if i == 1:
        plt.legend((scatter1, scatter2,),
                   handles=[red_patch, Blue_patch],
                   scatterpoints=1,
                   loc='upper right',
                   ncol=1,
                   fontsize=12)
    plt.xlim([2,17])
    plt.ylim([-3,15])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    # Disable grid lines
    ax.grid(False)
    
    
    
        # Set the title of the plot
    #ax.set_title('Generation 2')
    
        # Save the plot as an image
    plt.savefig(f"umapdata_generation{generation}_w{w}.jpg",
                    format='jpg',
                    bbox_inches='tight',
                    dpi=200,
    pad_inches=.02)
    
    # Close the figure
    plt.show()

