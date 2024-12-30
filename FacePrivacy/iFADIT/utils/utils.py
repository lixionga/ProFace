import torch
import numpy as np
import random

def loading_pretrianed(pretrained_dict, net):
    net_dict = net.state_dict()
    keys = []
    for k, v in pretrained_dict.items():
        keys.append(k)
    i = 0
    # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
    for k, v in net_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            net_dict[k] = pretrained_dict[keys[i]]
            # print(model_dict[k])
            i = i + 1
    net.load_state_dict(net_dict)


def get_concat_vec(id_images, attr_images, id_encoder, attr_encoder, G_net, fuse_mlp, passwords, mode):
    with torch.no_grad():
        if mode == 'forward':
            id_vec = id_encoder.extract_feats((id_images*2.)-1.)
            # print(f'id_vec shape is {id_vec.shape}, passwords shape is {passwords.shape}')
            id_fake, _ = G_net(id_vec, c=[passwords])
        if mode == 'backward':
            id_vec = id_encoder.extract_feats((id_images*2.)-1.)

            # compensated MLP
            id_vec = fuse_mlp(id_vec)
            # print(f'id_vec shape is {id_vec.shape}, passwords shape is {passwords.shape}')
            id_fake, _ = G_net(id_vec, c=[passwords], rev=True)
        attr_vec = attr_encoder(attr_images)


        # print(f'id_fake shape = {id_fake.shape}')
        # random_indices = torch.randperm(id_fake.size(0), device="cuda:0")
        # id_fake = id_fake[random_indices]
        # print(f'random index is {random_indices}')
        # attr_vec = fuse_attr_mlp(attr_vec)
        # print(f'_________attr_vec {attr_vec.shape}')          
        id_fake_s = id_fake.unsqueeze(1)
        id_fake_s, _ = torch.broadcast_tensors(id_fake_s, attr_vec)
        # print(f'{id_fake_s.shape, attr_vec.shape}')
        test_id_vec = torch.cat((id_fake_s, attr_vec), dim=2)

    return test_id_vec,id_fake_s, attr_vec, torch.broadcast_tensors(id_vec.unsqueeze(1), attr_vec)[0]

def inversion_encoder(id_images, attr_images, id_encoder, attr_encoder):
    with torch.no_grad():
        id_vec = id_encoder.extract_feats((id_images*2.)-1.).to(Global_Config.device)
        attr_vec = attr_encoder(attr_images)       # *torch.Tensor([0.000000001]).to(Global_Config.device)
        id_fake_s = id_vec.unsqueeze(1)
        id_fake_s, _ = torch.broadcast_tensors(id_fake_s, attr_vec)
        test_id_vec = torch.cat((id_fake_s, attr_vec), dim=2)

    return test_id_vec

def de_attr_encoder(id_images, attr_images, id_encoder, attr_encoder):
    with torch.no_grad():
        id_vec = id_encoder.extract_feats((id_images*2.)-1.)
        attr_vec = attr_encoder(attr_images)*torch.Tensor([0]).to(Global_Config.device)
        id_fake_s = id_vec.unsqueeze(1)
        id_fake_s, _ = torch.broadcast_tensors(id_fake_s, attr_vec)
        test_id_vec = torch.cat((id_fake_s, attr_vec), dim=2)

    return test_id_vec
def de_id_encoder(id_images, attr_images, id_encoder, attr_encoder):
    with torch.no_grad():
        id_vec = id_encoder.extract_feats((id_images*2.)-1.)*torch.Tensor([0]).to(Global_Config.device)
        attr_vec = attr_encoder(attr_images)
        id_fake_s = id_vec.unsqueeze(1)
        id_fake_s, _ = torch.broadcast_tensors(id_fake_s, attr_vec)
        test_id_vec = torch.cat((id_fake_s, attr_vec), dim=2)


    return test_id_vec

def normalize(x: torch.Tensor, adaptive=False):
    _min, _max = -1, 1
    if adaptive:
        _min, _max = x.min(), x.max()
    x_norm = (x - _min) / (_max - _min)
    return x_norm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True