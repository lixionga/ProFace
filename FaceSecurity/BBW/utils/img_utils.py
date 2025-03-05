import os
import numpy as np
import torch
import torchvision


def get_random_images(cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val, diff_nm_secret_val,
                      diff_mp_secret_val):
    selected_id = np.random.randint(1, steg.shape[0]) if steg.shape[0] > 1 else 1
    cover = cover.cpu()[selected_id - 1:selected_id, :, :, :]
    secret = secret.cpu()[selected_id - 1:selected_id, :, :, :]
    steg = steg.cpu()[selected_id - 1:selected_id, :, :, :]
    steg_mp_val = steg_mp_val.cpu()[selected_id - 1:selected_id, :, :, :]
    secret_rev_val = secret_rev_val.cpu()[selected_id - 1:selected_id, :, :, :]
    secret_rev_mp_val = secret_rev_mp_val.cpu()[selected_id - 1:selected_id, :, :, :]
    diff_nm_secret_val = diff_nm_secret_val.cpu()[selected_id - 1:selected_id, :, :, :]
    diff_mp_secret_val = diff_mp_secret_val.cpu()[selected_id - 1:selected_id, :, :, :]

    return [cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val, diff_nm_secret_val, diff_mp_secret_val]


def concatenate_images(saved_all, cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val,
                       diff_nm_secret_val,
                       diff_mp_secret_val):
    saved = get_random_images(cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val, diff_nm_secret_val,
                              diff_mp_secret_val)
    if saved_all[2].shape[2] != saved[2].shape[2]:
        return saved_all
    saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
    saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
    saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
    saved_all[3] = torch.cat((saved_all[3], saved[3]), 0)
    saved_all[4] = torch.cat((saved_all[4], saved[4]), 0)
    saved_all[5] = torch.cat((saved_all[5], saved[5]), 0)
    saved_all[6] = torch.cat((saved_all[6], saved[6]), 0)
    saved_all[7] = torch.cat((saved_all[7], saved[7]), 0)
    return saved_all


def save_images(saved_all, folder, description):
    cover, secret, steg, steg_mp_val, secret_rev_val, secret_rev_mp_val, diff_nm_secret_val, diff_mp_secret_val = saved_all

    stacked_images = None

    for i in range(steg.size(0)):
        tmp = torch.cat(
            [cover[i].unsqueeze(0), secret[i].unsqueeze(0),
             steg[i].unsqueeze(0), steg_mp_val[i].unsqueeze(0),
             secret_rev_val[i].unsqueeze(0), secret_rev_mp_val[i].unsqueeze(0),
             diff_nm_secret_val[i].unsqueeze(0), diff_mp_secret_val[i].unsqueeze(0)], dim=3)
        if stacked_images == None:
            stacked_images = tmp
        else:
            stacked_images = torch.cat([stacked_images, tmp], dim=2)

    filename = os.path.join(folder, 'Fig4_all_' + description + '.png')
    torchvision.utils.save_image(stacked_images, filename)

# def get_random_images(cover, steg, diff_steg):
#     selected_id = np.random.randint(1, cover.shape[0]) if cover.shape[0] > 1 else 1
#     cover = cover.cpu()[selected_id - 1:selected_id, :, :, :]
#     steg = steg.cpu()[selected_id - 1:selected_id, :, :, :]
#     diff_steg = diff_steg.cpu()[selected_id - 1:selected_id, :, :, :]
#
#     return [cover, steg, diff_steg]
#
#
# def concatenate_images(saved_all, cover, steg, diff_steg):
#     saved = get_random_images(cover, steg, diff_steg)
#     if saved_all[2].shape[2] != saved[2].shape[2]:
#         return saved_all
#     saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
#     saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
#     saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
#     return saved_all
#
# def save_images(saved_all, folder, description):
#     cover, steg, diff_steg = saved_all
#
#     stacked_images = []
#
#     for i in range(cover.size(0)):
#         # Horizontally concatenate cover, steg, and diff_steg for this row
#         row = torch.cat(
#             [cover[i].unsqueeze(0), steg[i].unsqueeze(0), diff_steg[i].unsqueeze(0)], dim=2)  # Concatenate horizontally
#         stacked_images.append(row)
#
#     # Stack all rows vertically
#     stacked_images = torch.cat(stacked_images, dim=0)  # Concatenate rows vertically
#
#     # Ensure the tensor has the right shape: (N, C, H, W)
#     if stacked_images.dim() == 3:  # Check if it's (C, H, W) format
#         stacked_images = stacked_images.unsqueeze(0)  # Add a batch dimension if necessary
#
#     # Save the image
#     filename = os.path.join(folder, description + '_save_image.png')
#     torchvision.utils.save_image(stacked_images, filename)
