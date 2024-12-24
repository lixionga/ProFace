import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 
import scipy 
import scipy.ndimage 

import functools

class GaussianLayer(nn.Module):
    def __init__(self, radius, sigma):
        super(GaussianLayer, self).__init__()
        assert (radius % 2) > 0, "Radius can not be even!"
        assert sigma > 0, "Sigma should be non-negative!"

        self.seq = nn.Sequential(
            nn.ReflectionPad2d(int((radius-1)/2)),
            nn.Conv2d(3,3,radius,stride=1,padding=0,bias=None,groups=3)
        )
        self.weights_init(radius, sigma)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, radius, sigma):
        n= np.zeros((radius,radius))
        n[int((radius-1)/2),int((radius-1)/2)] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=sigma)
        print('kernel in gaussian')
        print(k)
        for _, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False

class PixelGaussian(nn.Module):
    def __init__(self, radius, image_size=256):
        super(PixelGaussian, self).__init__()
        assert (radius % 2) > 0, "Radius can not be even!"

        self.radius = radius 
        self.image_size = image_size
        self.pad = nn.ReflectionPad2d(int((self.radius-1)//2))
        # init 
        self.origin_weights = [[None for j in range(self.radius)] for i in range(self.radius)]
        a = radius // 2
        b = radius // 2 
        for i in range(self.radius):
            for j in range(self.radius):
                self.origin_weights[i][j] = torch.full((1, 3, image_size, image_size), -((i-a)*(i-a) + (j-b)*(j-b))/2, dtype=torch.float32).cuda()

    def forward(self, x, sigma):
        self.tmp_weight = self.weights_init(sigma, x)
        
        result = torch.zeros_like(x)
        x = self.pad(x)
        for i in range(self.radius):
            for j in range(self.radius):
                result += x[:, :, i:i+self.image_size, j:j+self.image_size] * self.tmp_weight[i][j]
        return result 

    def weights_init(self, sigma, x):
        tmp = [[None for j in range(self.radius)] for i in range(self.radius)]
        for i in range(self.radius):
            for j in range(self.radius):
                tmp[i][j] = torch.exp(self.origin_weights[i][j].expand(x.shape[0], -1, -1, -1) * sigma)
                if i == 0 and j == 0:
                    tmp_sum = tmp[i][j].clone()
                else:
                    tmp_sum += tmp[i][j]

        for i in range(self.radius):
            for j in range(self.radius):
                tmp[i][j] = tmp[i][j] / tmp_sum
        return tmp 

class PixelFilter(nn.Module):
    def __init__(self, radius=3):
        super(PixelFilter, self).__init__()
        assert (radius % 2) > 0, "Radius can not be even!"

        self.radius = radius 
        self.eps = 1e-9
        self.pad = nn.ReflectionPad2d(int((self.radius-1)//2))

    def forward(self, x, sigma):

        sum_tmp = sigma.sum(dim=1, keepdim=True) 
        sigma = sigma / (sum_tmp + self.eps)

        result = torch.zeros_like(x)
        x = self.pad(x)
        # print(x.shape) 
        for i in range(self.radius):
            for j in range(self.radius):
                result += x[:, :, i:i+256, j:j+256] * sigma[:, self.radius*i+j:self.radius*i+j+1, ...]
        return result 