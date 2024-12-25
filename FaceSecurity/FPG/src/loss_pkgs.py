import torch
from torch import nn
import torch.nn.functional as F

class MLLoss(nn.Module):
    def __init__(self, sim_threshold, invert = False):
        super(MLLoss, self).__init__()
        self.threshold = sim_threshold
        self.invert = invert
        
    def forward(self, input, target, eps=1e-6):
        # 0 - real; 1 - fake.
            
        loss = torch.tensor(0., device=target.device)
        batch_size = target.shape[0]
        mat_1 = torch.hstack([target.unsqueeze(-1)] * batch_size)
        mat_2 = torch.vstack([target] * batch_size)
        diff_mat = torch.logical_xor(mat_1, mat_2).float()
        or_mat = torch.logical_or(mat_1, mat_2)
        eye = torch.eye(batch_size, device=target.device)
        or_mat = torch.logical_or(or_mat, eye).float()
  
        for _ in input:
            mask = _ >= self.threshold
            sim = torch.sum(_ * diff_mat * mask, dim=[0, 1]) / (torch.sum(diff_mat * mask, dim=[0, 1]) + eps)
            partial_loss = sim - self.threshold
            loss += max(partial_loss, torch.zeros_like(partial_loss))
        
        return loss


class MLLoss_sim(nn.Module):
    def __init__(self):
        super(MLLoss_sim, self).__init__()
        
    def forward(self, input):
        # 0 - real; 1 - fake.
        b = input.shape[0]
        
        real_feat, fake_feat = input[:b//2], input[b//2:]
        sim = real_feat @ fake_feat.t() # b/2 x b/2
        sim = torch.diag(sim)

        loss = (sim + 1) / 2

        loss = torch.mean(loss)
        
        return loss


class CLoss_inout(nn.Module):
    def __init__(self):
        super(CLoss_inout, self).__init__()
        
    def forward(self, input_in, input_out, flip=False):
        # 0 - real; 1 - fake.
        sim = input_in @ input_out.t() # b/2 x b/2
        sim = torch.diag(sim)

        if not flip:
            loss = 1 - sim
            loss = torch.mean(loss)
        else:
            loss = sim + 1
            loss = torch.mean(loss)
            
        return loss


class CLoss_local(nn.Module):
    def __init__(self):
        super(CLoss_local, self).__init__()
        self.mse = nn.MSELoss().train()
        
    def forward(self, input_in, mask_local):
        # 0 - real; 1 - fake.
        R = mask_local.shape[1]
        H, W = input_in.shape[2], input_in.shape[3]
        mask_local = F.interpolate(mask_local, size=(H, W))
        mask_local[mask_local > 0] = 1
        
        # BRHW BCHW
        local_feats =  mask_local[:,:,None] * input_in[:,None]
        feat_0_in = torch.sum(local_feats, dim=(3, 4)) / (torch.sum(mask_local, dim=(2, 3), keepdim=True).squeeze(-1) + 1e-10)
        feat_0_in_mean = torch.mean(feat_0_in, dim = 1, keepdim=True)
        
        loss = self.mse(feat_0_in, feat_0_in_mean.expand(-1, R, -1))

        return loss


class CLoss_local_v2(nn.Module):
    def __init__(self):
        super(CLoss_local_v2, self).__init__()
        self.mse = nn.MSELoss().train()
        
    def forward(self, input_in, mask_local):
        # 0 - real; 1 - fake.
        R = mask_local.shape[1]
        H, W = input_in.shape[2], input_in.shape[3]
        mask_local = F.interpolate(mask_local, size=(H, W))
        mask_local[mask_local > 0] = 1
        
        # BRHW BCHW
        local_feats =  mask_local[:,:,None] * input_in[:,None]
        feat_0_in = torch.sum(local_feats, dim=(3, 4)) / (torch.sum(mask_local, dim=(2, 3), keepdim=True).squeeze(-1) + 1e-10)

        feat_0_in_locals = feat_0_in[:, :-1]
        feat_0_in_comp = feat_0_in[:, -1].unsqueeze(1)

        loss = self.mse(feat_0_in_locals, feat_0_in_comp.expand(-1, R - 1, -1))

        return loss