from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.face_parsing.parsing_model import BiSeNet
from models.networks.architecture import VGG19
from torchvision.utils import save_image

class SemanticCorrespondence(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):

        return parser

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.K = 3
        self.k_means_content = KMeans(self.K)
        self.k_means_style = KMeans(self.K)

    def forward(self, x_features, y_features, x_label,y_label, x_protect):

        if self.opt.no_mask:
            if self.opt.no_soft:
                warped_features = [self.correspondence(x_features[i], y_features[i], 2 ** i) for i in list(range(self.opt.multiscale_level))]
            else:
                warped_features = [self.correspondence_soft(x_features[i], y_features[i], 2 ** i) for i in list(range(self.opt.multiscale_level))]
            return warped_features
        else:
            #Change the 1-chanel mask to n_c-chanel mask if use the MT dataset
            if self.opt.phase =='test' and not self.opt.beyond_mt:
                x_label=self.one2multi_chanels(x_label,self.opt.n_c)
                y_label=self.one2multi_chanels(y_label,self.opt.n_c)
            x_semantics = [F.interpolate(x_label, scale_factor=i, mode='nearest') for i in [0.125, 0.25, 0.5, 1]]
            y_semantics = [F.interpolate(y_label, scale_factor=i, mode='nearest') for i in [0.125, 0.25, 0.5, 1]]
            x_protects = [F.interpolate(x_protect, scale_factor=i, mode='nearest') for i in [0.125, 0.25, 0.5, 1]]

            warped_features = [self.correspondence_soft_mask(x_features[i], y_features[i],
                                                                x_semantics[i], y_semantics[i], 2 ** i)*(1-x_protects[i]) +x_features[i]*x_protects[i] for i in list(range(self.opt.multiscale_level))]                                     
            return warped_features

    def one2multi_chanels(self,one_c_label,n_c):
        b,c,h,w=one_c_label.shape
        label=torch.zeros([b,n_c, h, w], dtype=torch.float).to(one_c_label.device)
        for i in range(b):
            for j in range(n_c):
                label[i,j,:,:]=(one_c_label == j).float()*13.0
        #***********************
        eps=-5
        label=label+eps
        label= torch.softmax(label, 1)
        return label  

    def image_clustering(self, x, y, K):
        """
        :param x_features: shape -> [batch_size, c, h, w]
        :param y_features: shape -> [batch_size, c, h, w]
        :return: x_labels: shape -> [batch_size, c, h, w]
        :return: y_labels: shape -> [batch_size, c, h, w]
        """
        device = x.device
        b, c, h_x, w_x = x.shape
        b, c, h_y, w_y = y.shape
        content_features = x.squeeze().permute(1, 2, 0).reshape(-1, c) # (h*w, c)
        style_features = y.squeeze().permute(1, 2, 0).reshape(-1, c)   # (h*w, c)
        one_hot_label = torch.eye(K).to(device)
        self.k_means_content.fit(content_features.to('cpu'))
        content_labels = torch.Tensor(self.k_means_content.labels_).to(device)  # (h*w,)
        content_labels = content_labels.reshape(h_x, w_x)

        self.k_means_style.fit(style_features.to('cpu'))
        style_labels = torch.Tensor(self.k_means_style.labels_).to(device)  # (h*w,)
        style_labels = style_labels.reshape(h_y, w_y)
        y_labels = torch.zeros(h_y, w_y, K).to(device)
        for i in range(K):
            y_labels[style_labels==i] = one_hot_label[i]
        y_labels = y_labels.permute(2, 0, 1).unsqueeze(0)

        content_cluster_centers = torch.Tensor(self.k_means_content.cluster_centers_).to(device)    # (K, c)
        style_cluster_centers = torch.Tensor(self.k_means_style.cluster_centers_).to(device)    # (K, c)
        x_labels = torch.zeros(h_x, w_x, K).to(device)
        for i in range(K):
            index = F.softmax(torch.cosine_similarity(content_cluster_centers[i].repeat(K, 1), style_cluster_centers))
            x_labels[content_labels==i] = index
        x_labels = x_labels.permute(2, 0, 1).unsqueeze(0)

        # save_image(x_labels, 'c_k_means.jpg', nrow=1)
        # save_image(y_labels, 's_k_means.jpg', nrow=1)

        return x_labels, y_labels


    def calc_mean_std(self, features):
        """
        :param features: shape of features -> [batch_size, c, h, w]
        :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
        """
        batch_size, c = features.size()[:2]
        features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
        features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
        return features_mean, features_std

    def adain(self, content_features, style_features):
        """
        Adaptive Instance Normalization

        :param content_features: shape -> [batch_size, c, h, w]
        :param style_features: shape -> [batch_size, c, h, w]
        :return: normalized_features shape -> [batch_size, c, h, w]
        """
        content_mean, content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
        return normalized_features

    def correspondence_soft_mask(self, x_features, y_features, x_semantic, y_semantic, patch_size):
        """
        warp y_features to the content of x_features with the guide of x_semantic and y_semantic

        :param x_features: shape -> [batch_size, c, h, w]
        :param y_features: shape -> [batch_size, c, h, w]
        :return: warped_features shape -> [batch_size, c, h, w]
        """
        b, c, h, w = x_features.shape  # 1, 512, 64, 64 for image with shape 1, 3, 512, 512
        patch_size = patch_size
        x_stride = patch_size
        y_stride = patch_size

        # correspondence of x_features and y_features
        y_patches = self.patches_sampling(y_features, patch_size=patch_size, stride=y_stride)  # the tensor shape of y_patches is 4096, 512, 1, 1
        eps = 1e-6
        y_patches_norm = self.cal_patches_norm(y_patches)+eps
        y_patches_norm = y_patches_norm.view(-1, 1, 1)

        exemplar_nums = y_patches.shape[0]  # 4096

        response = F.conv2d(x_features, y_patches, stride=x_stride)  # 1, 512, 64, 64 convoluted by 4096, 512, 1, 1 return 1, 4096, 64, 64
        response = response.div(y_patches_norm) 
        response_height, response_width = response.shape[2:]
        response = torch.reshape(response.permute(0,2,3,1), (-1,exemplar_nums))

        # correspondence of x_semantic and y_semantic
        y_semantic_patches = self.patches_sampling(y_semantic, patch_size=patch_size, stride=y_stride)
        y_semantic_patches_norm = self.cal_patches_norm(y_semantic_patches) + eps
        y_semantic_patches_norm = y_semantic_patches_norm.view(-1, 1, 1)

        semantic_response = F.conv2d(x_semantic, y_semantic_patches, stride=x_stride)
        semantic_response = semantic_response.div(y_semantic_patches_norm)
        semantic_response = torch.reshape(semantic_response.permute(0,2,3,1), (-1,exemplar_nums))

        warp_weight = F.softmax(response * semantic_response, 1)
        
        correspondence = torch.reshape(torch.mm(warp_weight, y_patches.view(exemplar_nums,-1)), (b, response_height, response_width, c, patch_size, patch_size))#([1,32,32,512,1,1])

        warped_features = torch.zeros(x_features.shape).to(x_features.device)


        r = [x for x in range(0, h - patch_size + 1, x_stride)]
        c = [x for x in range(0, w - patch_size + 1, x_stride)]
        
        for i in range(response_height):
            for j in range(response_width):
                for batch in range(b):
                    warped_features[batch, :, r[i]:r[i] + patch_size, c[j]:c[j] + patch_size] = correspondence[batch, i, j]
        
        return warped_features
     

    def correspondence_soft(self, x_features, y_features, patch_size):
        """
        warp y_features to the content of x_features with the guide of x_semantic and y_semantic

        :param x_features: shape -> [batch_size, c, h, w]
        :param y_features: shape -> [batch_size, c, h, w]
        :return: warped_features shape -> [batch_size, c, h, w]
        """
        b, c, h, w = x_features.shape  # 1, 512, 64, 64 for image with shape 1, 3, 512, 512
        patch_size = patch_size
        x_stride = patch_size
        y_stride = patch_size

        # correspondence of x_features and y_features
        y_patches = self.patches_sampling(y_features, patch_size=patch_size, stride=y_stride)  # the tensor shape of y_patches is 4096, 512, 1, 1
        y_patches_norm = self.cal_patches_norm(y_patches)
        y_patches_norm = y_patches_norm.view(-1, 1, 1)

        exemplar_nums = y_patches.shape[0]  # 4096

        response = F.conv2d(x_features, y_patches, stride=x_stride)  # 1, 512, 64, 64 convoluted by 4096, 512, 1, 1 return 1, 4096, 64, 64
        response = response.div(y_patches_norm)
        response_height, response_width = response.shape[2:]
        response = torch.reshape(response.permute(0,2,3,1), (-1,exemplar_nums))
          
        warp_weight = F.softmax(response, 1)
        
        correspondence = torch.reshape(torch.mm(warp_weight, y_patches.view(exemplar_nums,-1)), (b, response_height, response_width, c, patch_size, patch_size))

        warped_features = torch.zeros(x_features.shape).to(x_features.device)
        
        r = [x for x in range(0, h - patch_size + 1, x_stride)]
        c = [x for x in range(0, w - patch_size + 1, x_stride)]
        
        for i in range(response_height):
            for j in range(response_width):
                for batch in range(b):
                    warped_features[batch, :, r[i]:r[i] + patch_size, c[j]:c[j] + patch_size] = correspondence[batch, i, j]
        
        return warped_features

    def correspondence(self, x_features, y_features, patch_size):
        """
        warp y_features to the content of x_features with the guide of x_semantic and y_semantic

        :param x_features: shape -> [batch_size, c, h, w]
        :param y_features: shape -> [batch_size, c, h, w]
        :return: warped_features shape -> [batch_size, c, h, w]
        """
        b, c, h, w = x_features.shape  # 1, 512, 64, 64 for image with shape 1, 3, 512, 512
        patch_size = patch_size
        x_stride = patch_size
        y_stride = patch_size

        # correspondence of x_features and y_features
        y_patches = self.patches_sampling(y_features, patch_size=patch_size, stride=y_stride)  # the tensor shape of y_patches is 4096, 512, 1, 1
        y_patches_norm = self.cal_patches_norm(y_patches)
        y_patches_norm = y_patches_norm.view(-1, 1, 1)

        exemplar_nums = y_patches.shape[0]  # 4096

        response = F.conv2d(x_features, y_patches, stride=x_stride)  # 1, 512, 64, 64 convoluted by 4096, 512, 1, 1 return 1, 4096, 64, 64
        response = response.div(y_patches_norm)
        response_height, response_width = response.shape[2:]
        response = torch.reshape(response.permute(0,2,3,1), (-1,exemplar_nums))
          
        max_response = torch.argmax(response, 1)
        
        correspondence = y_patches[max_response, :, :, :]

        warped_features = torch.zeros(x_features.shape).to(x_features.device)
        
        r = [x for x in range(0, h - patch_size + 1, x_stride)]
        c = [x for x in range(0, w - patch_size + 1, x_stride)]
        k = 0
        for i in range(response_height):
            for j in range(response_width):
                for batch in range(b):
                    warped_features[batch, :, r[i]:r[i] + patch_size, c[j]:c[j] + patch_size] = correspondence[k]
                k = k + 1
        return warped_features

    @staticmethod
    def patches_sampling(image, patch_size, stride):
        """
        sampling patches form a image
        :param image:
        :param patch_size:
        :return:
        """
        h, w = image.shape[2:4]
        patches = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patches.append(image[:, :, i:i + patch_size, j:j + patch_size])
        patches = torch.cat(patches, dim=0)
        return patches

    @staticmethod
    def cal_patches_norm(patches):
        """
        calculate norm of image patches
        :return:
        """
        norm_array = torch.zeros(patches.shape[0]).to(patches.device)
        for i in range(patches.shape[0]):
            norm_array[i] = torch.pow(torch.sum(torch.pow(patches[i], 2)), 0.5)
        return norm_array
