import os
import numpy as np
import torch
import torchvision


def get_random_images(cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val):
    selected_id = np.random.randint(1, steg.shape[0]) if steg.shape[0] > 1 else 1
    cover = cover.cpu()[selected_id - 1:selected_id, :, :, :]
    secret = secret.cpu()[selected_id - 1:selected_id, :, :, :]
    steg = steg.cpu()[selected_id - 1:selected_id, :, :, :]
    steg_mp_val = steg_mp_val.cpu()[selected_id - 1:selected_id, :, :, :]
    secret_rev_val = secret_rev_val.cpu()[selected_id - 1:selected_id, :, :, :]
    secret_rev_mp_val = secret_rev_mp_val.cpu()[selected_id - 1:selected_id, :, :, :]

    return [cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val]


def concatenate_images(saved_all, cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val):
    saved = get_random_images(cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val)
    if saved_all[2].shape[2] != saved[2].shape[2]:
        return saved_all
    saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
    saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
    saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
    saved_all[3] = torch.cat((saved_all[3], saved[3]), 0)
    saved_all[4] = torch.cat((saved_all[4], saved[4]), 0)
    saved_all[5] = torch.cat((saved_all[5], saved[5]), 0)
    return saved_all


def save_images(saved_all, folder, description):
    cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val = saved_all

    stacked_images = None

    for i in range(steg.size(0)):
        tmp = torch.cat(
            [cover[i].unsqueeze(0), secret[i].unsqueeze(0),
             steg[i].unsqueeze(0), steg_mp_val[i].unsqueeze(0),
             secret_rev_val[i].unsqueeze(0), secret_rev_mp_val[i].unsqueeze(0)], dim=3)
        if stacked_images == None:
            stacked_images = tmp
        else:
            stacked_images = torch.cat([stacked_images, tmp], dim=2)

    filename = os.path.join(folder, 'Fig_' + description + '.png')
    torchvision.utils.save_image(stacked_images, filename)
