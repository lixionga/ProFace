from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import os.path as osp
import shutil

import config
# from face_identity_transformer_master import models
from network.noise_layers.face_identity_transformer_master import models


def infoGAN_input(img1, img2):
    return torch.cat((img1, img2), 1)


def get_feat_loss(feat1, feat2, target, feature_loss_type, criterion_dict):
    if feature_loss_type == 'cos':
        loss_feat = criterion_dict['Feat'](feat1, feat2, target=target)
    else:
        loss_feat = -criterion_dict['Feat'](feat1, feat2)
    return loss_feat


def unsigned_long_to_binary_repr(unsigned_long, passwd_length):
    batch_size = unsigned_long.shape[0]
    target_size = passwd_length // 4

    binary = np.empty((batch_size, passwd_length), dtype=np.float32)
    for idx in range(batch_size):
        binary[idx, :] = np.array([int(item) for item in bin(unsigned_long[idx])[2:].zfill(passwd_length)])

    # dis_target = np.empty((batch_size, target_size), dtype=np.long)
    dis_target = np.empty((batch_size, target_size))
    for idx in range(batch_size):
        tmp = unsigned_long[idx]
        for byte_idx in range(target_size):
            dis_target[idx, target_size - 1 - byte_idx] = tmp % 16
            tmp //= 16
    return binary, dis_target


def generate_code(passwd_length, batch_size, device, inv, use_minus_one, gen_random_WR):
    unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
    binary, dis_target = unsigned_long_to_binary_repr(unsigned_long, passwd_length)
    z = torch.from_numpy(binary).to(device)
    dis_target = torch.from_numpy(dis_target).to(device)

    repeated = True
    while repeated:
        rand_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
        repeated = np.any(unsigned_long - rand_unsigned_long == 0)
    rand_binary, rand_dis_target = unsigned_long_to_binary_repr(rand_unsigned_long, passwd_length)
    rand_z = torch.from_numpy(rand_binary).to(device)
    rand_dis_target = torch.from_numpy(rand_dis_target).to(device)

    if not inv:
        if use_minus_one is True:
            z = (z - 0.5) * 2
            rand_z = (rand_z - 0.5) * 2
        elif use_minus_one == 'half':
            z -= 0.5
            rand_z -= 0.5
        elif use_minus_one == 'one_fourth':
            z = (z - 0.5) / 2
            rand_z = (z - 0.5) / 2
        return z, dis_target, rand_z, rand_dis_target
    else:
        inv_unsigned_long = 2 ** passwd_length - 1 - unsigned_long
        inv_binary, inv_dis_target = unsigned_long_to_binary_repr(inv_unsigned_long, passwd_length)

        inv_z = torch.from_numpy(inv_binary).to(device)
        inv_dis_target = torch.from_numpy(inv_dis_target).to(device)

        repeated = True
        while repeated:
            rand_inv_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
            repeated = np.any(inv_unsigned_long - rand_inv_unsigned_long == 0)
        rand_inv_binary, rand_inv_dis_target = unsigned_long_to_binary_repr(rand_inv_unsigned_long, passwd_length)
        rand_inv_z = torch.from_numpy(rand_inv_binary).to(device)
        rand_inv_dis_target = torch.from_numpy(rand_inv_dis_target).to(device)

        if gen_random_WR:
            repeated = True
            while repeated:
                rand_inv_2nd_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,),
                                                               dtype=np.uint64)
                repeated = np.any(inv_unsigned_long - rand_inv_2nd_unsigned_long == 0) or np.any(
                    rand_inv_unsigned_long - rand_inv_2nd_unsigned_long == 0)
            rand_inv_2nd_binary, rand_inv_2nd_dis_target = unsigned_long_to_binary_repr(
                rand_inv_2nd_unsigned_long,
                passwd_length)
            rand_inv_2nd_z = torch.from_numpy(rand_inv_2nd_binary).to(device)
            rand_inv_2nd_dis_target = torch.from_numpy(rand_inv_2nd_dis_target).to(device)

        if use_minus_one is True:
            z = (z - 0.5) * 2
            rand_z = (rand_z - 0.5) * 2
            inv_z = z * -1.
            rand_inv_z = (rand_inv_z - 0.5) * 2
            if gen_random_WR:
                rand_inv_2nd_z = (rand_inv_2nd_z - 0.5) * 2

        elif use_minus_one == 'half':
            z -= 0.5
            rand_z -= 0.5
            inv_z -= 0.5
            rand_inv_z -= 0.5
            if gen_random_WR:
                rand_inv_2nd_z -= 0.5

        elif use_minus_one == 'one_fourth':
            z = (z - 0.5) / 2 # 0 -> -0.25, 1-> 0.25
            rand_z = (rand_z - 0.5) / 2
            inv_z = (inv_z - 0.5) / 2
            rand_inv_z = (rand_inv_z - 0.5) / 2
            if gen_random_WR:
                rand_inv_2nd_z = (rand_inv_2nd_z - 0.5) / 2

        if gen_random_WR:
            return z, dis_target, rand_z, rand_dis_target, \
                   inv_z, inv_dis_target, rand_inv_z, rand_inv_dis_target, rand_inv_2nd_z, rand_inv_2nd_dis_target
        return z, dis_target, rand_z, rand_dis_target, \
               inv_z, inv_dis_target, rand_inv_z, rand_inv_dis_target, None, None


def alignment(src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    # crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    s = s / 125. - 1.
    r[:, 0] = r[:, 0] / 48. - 1
    r[:, 1] = r[:, 1] / 56. - 1

    all_tfms = np.empty((s.shape[0], 2, 3), dtype=np.float32)
    for idx in range(s.shape[0]):
        all_tfms[idx, :, :] = models.get_similarity_transform_for_cv2(r, s[idx, ...])
    all_tfms = torch.from_numpy(all_tfms).to(torch.device("cuda:0"))
    return all_tfms


def save_model(epoch, model_dict, optimizer_dict, args, iter=0, save_sep=True):
    save_dict = {
        'epoch': epoch,
        'iter': iter,
        # 'arch': args.arch,
    }

    for name, net in model_dict.items():
        if isinstance(net, list):
            continue
        save_dict['state_dict_' + name] = net.module.state_dict()

    for name, optimizer in optimizer_dict.items():
        if optimizer is not None: save_dict['optimizer_' + name] = optimizer.state_dict()
    save_checkpoint(save_dict, args.ckpt_dir, save_sep)


def save_checkpoint(state, ckpt_dir, save_sep, filename='checkpoints.pth.tar'):
    # TODO: if too much models are saved, delete some of them
    torch.save(state, osp.join(ckpt_dir, filename))
    # torch.save(state, osp.join(ckpt_dir, 'checkpoint_' + str(state['epoch']) + '_iter' + str(state['iter']) + '.pth.tar'))
    if save_sep:
        shutil.copyfile(
            osp.join(ckpt_dir, filename),
            osp.join(ckpt_dir, 'checkpoint_' + str(state['epoch']) + '_iter' + str(state['iter']) + '.pth.tar'),
        )


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    """
    ONLY SAVES the 1st image of the batch
    :param input_image:
    :param imtype:
    :return:
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # it can be deducted that the original range is [-1, 1]
    return image_numpy.astype(imtype)


def tensor2im_all(input_image, imtype=np.uint8):
    """
    SAVES ALL the images of the batch
    :param input_image:
    :param imtype:
    :return:
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[1] == 1:
        image_numpy = np.tile(image_numpy, (1, 3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    # it can be deducted that the original range is [-1, 1]
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
