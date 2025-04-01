import torch
import torch.nn as nn
import sklearn
import sklearn.preprocessing
import numpy as np
import cvxpy as cp
import torch.nn.functional as F
from scipy import spatial


def cos_simi( emb_before_pasted, emb_target_img):
    """
    :param emb_before_pasted: feature embedding for the generated adv-makeup face images
    :param emb_target_img: feature embedding for the victim target image
    :return: cosine similarity between two face embeddings
    """
    return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                      / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))

def numpy_cos(a,b):
    # dot = a*b #对应原始相乘dot.sum(axis=1)得到内积
    # a_len = np.linalg.norm(a,axis=1)#向量模长
    # b_len = np.linalg.norm(b,axis=1)
    cos = np.sum(a*b)/((np.linalg.norm(a,axis=1))*(np.linalg.norm(b,axis=1)))
    return cos



def cos_sim_dis(fea1,fea2):
    assert fea1.shape[0] == fea2.shape[0]
    fea1 = sklearn.preprocessing.normalize(fea1)
    fea2 = sklearn.preprocessing.normalize(fea2)
    similarity = []
    for i in range(fea1.shape[0]):
        similarity.append(np.sqrt(np.sum((fea1[i]-fea2[i])*(fea1[i]-fea2[i]))))
    return similarity

class inverse_mse(nn.Module):
    def __init__(self):
        super(inverse_mse, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, fea1, fea2):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim = 1).view(fea1.shape[0],1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim = 1).view(fea2.shape[0],1)
        dis = - self.mse(nfea1, nfea2)
        return dis

class eachother_dot(nn.Module):
    def __init__(self):
        super(eachother_dot, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, fea1, fea2):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)
        dis = torch.mean(torch.mm(nfea1, torch.transpose(nfea2, 0, 1)))
        return dis





def cos_simi_distance(emb_creat_img, emb_other_img):
    return 1 - torch.mean(torch.sum(torch.mul(emb_other_img, emb_creat_img), dim=1)
                            / emb_other_img.norm(dim=1) / emb_creat_img.norm(dim=1))


triplet_loss = nn.TripletMarginWithDistanceLoss(reduction="mean", distance_function=cos_simi_distance, margin=1)






class DFANet_MFIM():
    def __init__(self, step = 1, epsilon = 0.05, alpha = 1, random_start = True, loss_type=3, nter = 5, upper = 1.0, lower = 0.0):

        self.loss_type = loss_type
        self.step = step
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_start = random_start
        self.lower = lower
        self.upper = upper
        self.nter = nter
        if loss_type == 0: # FI-UAP
            self.LossFunction = inverse_mse()
        elif loss_type == 2: # FI-UAP+
            self.LossFunction = eachother_dot()
        elif loss_type == 7:  # OPOM-ClassCenter
            self.LossFunction = convex_hull_cvx_dyn()
        elif loss_type == 8: # OPOM-AffineHull
            self.LossFunction = convex_hull_cvx_dyn()
        elif loss_type == 9:  # OPOM-ConvexHull
            self.LossFunction = convex_hull_cvx_dyn()


    def process(self, model, target_data_1,target_data_2, fake_data, source_data):
        #prepare the models
        model.eval()
        # target_data=target_data.detach().clone()

        # target_data_noise = target_data + torch.zeros_like(target_data).uniform_(-self.epsilon, self.epsilon)
        # target_data_noise=target_data_noise.detach()

        for i in range(self.step):
            target_feature_1 = model(target_data_1)           

            target_feature_2 = model(target_data_2)
            
            fake_feature = model(fake_data)

            source_feature=model(source_data)

            tri_loss=triplet_loss(fake_feature, target_feature_1, source_feature)

        return tri_loss  
            # print("step", i, "dis", dis)

        #     if self.loss_type == 9:
        #         if i > self.nter: # init several steps to push adv to the outside of the convexhull
        #             Loss = self.LossFunction(fake_feature, target_feature_1, target_feature_2, source_feature)
        #         else:
        #             Loss_1 = self.LossFunction(fake_feature, target_feature_1,target_feature_2, source_feature, 0, 1.0)                      
        # return Loss_1






class convex_hull_cvx_dyn(nn.Module):
    def __init__(self):
        super(convex_hull_cvx_dyn, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, fake_feature, target_feature_1, target_feature_2, source_feature, lower = 0.0, upper = 1.0):

        

        fake_feature = fake_feature / torch.linalg.norm(fake_feature, dim=1).view(fake_feature.shape[0], 1)
        target_feature_1 = target_feature_1 / torch.linalg.norm(target_feature_1, dim=1).view(target_feature_1.shape[0], 1)
        target_feature_2 = target_feature_2 / torch.linalg.norm(target_feature_2, dim=1).view(target_feature_2.shape[0], 1)
        source_feature = source_feature / torch.linalg.norm(source_feature, dim=1).view(source_feature.shape[0], 1)
        # nfea2 --> A, nfea1 --> y, caculate x.
        # Using cvx to calculate variable x
        lowerbound = lower
        upperbound = upper
        A = target_feature_1.detach().cpu().numpy()
        XX = torch.tensor(np.zeros((target_feature_2.shape[0],target_feature_2.shape[0])), dtype=torch.float32, device=torch.device("cuda:0"))
        
        for i in range(target_feature_2.shape[0]):
            y = target_feature_2[i].detach().cpu().numpy()
            
            x = cp.Variable(target_feature_1.shape[0])
            # print(x.value)
            # x_A=x@A
            #embed()
            objective = cp.Minimize(cp.sum_squares(x @ A - y)) #欧式距离
            # objective = cp.Maximize((x_A@y)/((cp.norm(x_A))*(cp.norm(y))))        #余弦相似度
            constraints = [sum(x)==1, lowerbound <= x, x <= upperbound]  #凸包/类中心  
            # constraints = [sum(x)==1]     #仿射包
    
            prob = cp.Problem(objective, constraints)

            
            prob.solve()
            # print(i, "loss", prob.solve(), sum(x.value))
            #embed()
            # print(i, "x:", x.value)
            x_tensor = torch.tensor(x.value, dtype=torch.float32, device=torch.device("cuda:0"))
            XX[i]= x_tensor#XX.shape=[10,10]
        #embed()
        # DIS = - self.mse(torch.mm(XX.detach().to(fake_feature.device), target_nfea), fake_nfea)

        target_feature_space=torch.mm(XX.detach().to(fake_feature.device), target_feature_1)

        tri_loss=triplet_loss(fake_feature, target_feature_space, source_feature)

        # similarlity_respective=torch.cosine_similarity(torch.mm(XX.detach().to(fake_feature.device), target_nfea),fake_nfea,dim=1)
        # similarlity_all=cos_simi(torch.mm(XX.detach().to(fake_feature.device), target_nfea),fake_nfea)
        #embed()
        return tri_loss