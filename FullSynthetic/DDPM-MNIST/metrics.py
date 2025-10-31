import numpy as np
import sklearn.metrics
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.datasets import MNIST
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

from scipy import linalg
import os

__all__ = ['compute_prdc']


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k, flag):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0)

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1)

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0)

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    )
    if flag == 1:
        coverage = coverage.mean()
        recall = recall.mean()
        density = density.mean()
        precision = precision.mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)


def compute_nearest_neighbour_distances_slice(input_features, nearest_k, K):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    if input_features.shape[0] <= K:
        distances = compute_pairwise_distance(input_features)
        radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    else:
        k = np.ceil( input_features.shape[0]/K) 
        radii = np.array([])
        for i in range(int(k)):
            distances = compute_pairwise_distance(input_features[i*K:np.min((K*(i+1),input_features.shape[0]))],  input_features)
            radii = np.hstack(( radii,get_kth_value(distances, k=nearest_k + 1, axis=-1) ))
    return np.array(radii)


def compute_prdc_slice(real_features, fake_features, nearest_k, K, flag):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances_slice(real_features, nearest_k,K)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances_slice(fake_features, nearest_k,K)
    
    if fake_features.shape[0] < K:
        distance_real_fake = compute_pairwise_distance(real_features, fake_features)
        precision = (distance_real_fake <np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0)
        recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1)
        density = (1. / float(nearest_k)) * (distance_real_fake <np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0)
        coverage = (distance_real_fake.min(axis=1) <real_nearest_neighbour_distances)
    else:
        kfake = int(np.ceil( fake_features.shape[0]/K )) 
        kreal = int(np.ceil( real_features.shape[0]/K )) 
        precision = np.array([])
        density = np.array([])
        coverage = np.array([])
        recall = np.array([])
        for i in range(kfake):
            distance_real_fake = compute_pairwise_distance(real_features,fake_features[i*K:np.min((K*(i+1),fake_features.shape[0]))] )
            precision = np.hstack(( precision, (distance_real_fake <np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0) ))
            density = np.hstack(( density, (1. / float(nearest_k)) * (distance_real_fake <np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0) ))
        
        for i in range(kreal):
            distance_real_fake = compute_pairwise_distance(real_features[i*K:np.min((K*(i+1),real_features.shape[0]))] ,fake_features)
            recall = np.hstack((recall,  (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1)  ))
            coverage = np.hstack(( coverage, (distance_real_fake.min(axis=1) <real_nearest_neighbour_distances[i*K:np.min((K*(i+1),real_features.shape[0]))])))
        
    if flag == 1:
        
        coverage = coverage.mean()
        recall = recall.mean()
        density = density.mean()
        precision = precision.mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),                         
            nn.MaxPool2d(2),                         
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            )
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc3(out)
        return out



def load_datasets_dump(iteration,save_dir, w, batch_size = 128 ):
    
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    if iteration == 0:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    else:
        X = torch.load( save_dir + f"gen_data_with_w{iteration}_w{w}" )
        C = torch.load(save_dir + f"gen_index_with_w{iteration}_w{w}")
        dataset.data = np.squeeze(X.cpu())
        dataset.targets = C.cpu()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    return dataloader

def load_datasets(iteration,save_dir, w, batch_size = 128 ):
    
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    if iteration == 1:
        X = torch.load( save_dir + f"gen_data_with_w_initial_w{w}" )
        C = torch.load(save_dir + f"gen_index_with_w_initial_w{w}")
        C = torch.tensor(C)
        dataset.data = np.squeeze(X.cpu())
        dataset.targets = C.cpu()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    else:
        X = torch.load( save_dir + f"gen_data_with_w{iteration-1}_w{w}" )
        C = torch.load(save_dir + f"gen_index_with_w{iteration-1}_w{w}")
        dataset.data = np.squeeze(X.cpu())
        dataset.targets = C.cpu()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    return dataloader

def extract_mnist_features(images, device):
    if len(images.shape) == 3:
        images = images[:,None,:,:]
    images = images.to(device)
    model = LeNet()
    model.load_state_dict(torch.load('./data/diffusion_outputs10/' + "prmodel.pth"))
    model = model.to(device)
    model.eval()
    batch_size = 100
    num_batches = int(np.ceil(images.size(0) / batch_size))
    features = []
    for bi in range(num_batches):
        start = bi * batch_size
        end = start + batch_size
        batch = images[start:end]
        before_fc = model.features(batch.cuda())
        before_fc = before_fc.view(-1, 256)
        feature = model.fc(before_fc)
        features.append(feature.cpu().data.numpy())
    return np.concatenate(features, axis=0)



class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        
        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c, flag, n_sample, guide_w):
        """
        this method is used in training, so samples t and noise randomly
        """
        if flag == 1:
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
            noise = torch.randn_like(x)  # eps ~ N(0, 1)
    
            x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
            )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
            # We should predict the "error term" from this x_t. Loss is what we return.
    
            # dropout context with some probability
            context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
            
            # return MSE between added noise, and our predicted noise
            return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
        else:
            size = (1, 28, 28)
            
            x_i = torch.randn(n_sample, *size).to(self.device)  # x_T ~ N(0, 1), sample initial noise
            c_i = torch.arange(0,10).to(self.device) # context for us just cycles throught the mnist labels
            c_i = c_i.repeat(int(np.ceil(n_sample/c_i.shape[0])))
            c_i = c_i[0:n_sample]

            # don't drop context at test time
            context_mask = torch.zeros_like(c_i).to(self.device)

            # double the batch
            c_i = c_i.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 1. # makes second half of batch context free

            
            for i in range(self.n_T, 0, -1):

                t_is = torch.tensor([i / self.n_T]).to(self.device)
                t_is = t_is.repeat(n_sample,1,1,1)

                # double batch
                x_i = x_i.repeat(2,1,1,1)
                t_is = t_is.repeat(2,1,1,1)

                z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0

                # split predictions and compute weighting
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = ((1 + guide_w)*eps1 - guide_w*eps2)

                x_i = x_i[:n_sample]
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
            
            return x_i, c_i

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(np.ceil(n_sample/c_i.shape[0])))
        c_i = c_i[0:n_sample]

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        
        for i in tqdm(range(self.n_T, 0, -1)):
            #print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = ((1 + guide_w)*eps1 - guide_w*eps2)
            #eps = torch.norm(eps1)*eps/torch.norm(eps)
            #eps = eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        
        return x_i, c_i


def generate_samples(model, N, w, max_c):
    model = nn.DataParallel(model)
    gpu_num = torch.cuda.device_count()
    k = int(np.floor(N/(max_c*gpu_num)))
    N_sample = [max_c]*k
    if N - k*gpu_num*max_c > 0:
        N_sample.append( int( np.ceil(   (N - k*gpu_num*max_c)/gpu_num ) ))
    
    X = []
    C = []
    model.eval()
    with torch.no_grad():
        for n in N_sample:
            x_gen, c = model(0 ,0, 0, n, w)
            if n < max_c:
                X.append( x_gen[0:(N - k*gpu_num)] )
                C.append( c[0:(N - k*gpu_num)])
            else:
                X.append(x_gen)
                C.append(c)
    return torch.clip(torch.cat(X),0,1), torch.cat(C)
        
            

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):


    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def mytsne(X1,X2, save_dir, generation, w):

    tsne = TSNE(n_components = 2, random_state=1)
    X1 = torch.squeeze(X1)
    X1 = X1.view(X1.size(0), -1)
    X2 = torch.squeeze(X2)
    X2 = X2.view(X2.size(0), -1)
    
    
    X1 = X1.cpu().numpy()
    X2 = X2.cpu().numpy()
    
    tsne = TSNE(n_components = 2, random_state=1)
    tres = tsne.fit_transform(np.concatenate((X1,X2 ), axis = 0))
    
    plt.close('all')
    colors = ['r']*X1.shape[0] + ['b']*X2.shape[0]
    plt.scatter(tres[:,1], tres[:,0], s = 0.03, color = colors)
    plt.title(f"All-genration={generation} w={w}")
    plt.savefig(save_dir + f"All-genration={generation}w={w}.png")
    
    return tres
