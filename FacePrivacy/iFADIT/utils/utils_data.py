import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torchvision.transforms import Resize
import numpy
to_tensor_transform = transforms.ToTensor()
resize_128 = Resize([128,128])

def plot_single_w_image(w, generator):
    w = w.unsqueeze(0).to('cuda:0')
    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True
    )
    new_image = sample.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    new_image = (new_image + 1) / 2
    plt.axis('off')
    plt.imshow(new_image)
    plt.show()


def get_w_image(w, generator):
    w = w.unsqueeze(0).to('cuda:0')

    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True, randomize_noise=True
    )
    new_image = sample.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    new_image = (new_image + 1) / 2

    return new_image

def get_w_image_injection(w1, w2, generator):
    w1 = w1.unsqueeze(0).to('cuda:0')
    w2 = w2.unsqueeze(0).to('cuda:0')
    sample, latents = generator(
        [w1, w2], input_is_latent=True, return_latents=True, randomize_noise=True, inject_index= 5
    )
    new_image = sample.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    new_image = (new_image + 1) / 2

    return new_image

def get_data_by_index(idx, root_dir, postfix):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    dir_idx = idx // 1000

    path = os.path.join(root_dir, str(dir_idx), str(idx) + postfix)
    if postfix == ".npy":
        data = torch.tensor(np.load(path))

    elif postfix == ".png":
        data = to_tensor_transform(Image.open(path))

    else:
        return None

    return data


class Image_W_Dataset(Dataset):
    def __init__(self, w_dir, image_dir):
        self.w_dir = w_dir
        self.image_dir = image_dir


    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.w_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        w = get_data_by_index(idx, self.w_dir, ".npy")
        image = get_data_by_index(idx, self.image_dir, ".png")
        return w, image



class Id_W_Dataset(Dataset):
    def __init__(self, w_dir, image_dir):
        self.w_dir = w_dir
        self.image_dir = image_dir

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.w_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        image = get_data_by_index(idx, self.image_dir, ".png")
        label = idx
        # if idx < 10:
        #     label = str(idx).zfill(4)
        # elif idx <100:
        #     label = str(idx).zfill(3)
        # elif idx <1000:
        #     label = str(idx).zfill(2)
        # elif idx <10000:
        #     label = str(idx).zfill(1)

        return image, label


def cycle_images_to_create_diff_order(images):
    batch_size = len(images)
    different_images = torch.empty_like(images, device='cuda:0')
    different_images[0] = images[batch_size - 1]
    different_images[1:] = images[:batch_size - 1]
    return different_images

def recycle_images_to_create_diff_order(images):
    batch_size = len(images)
    different_images = torch.empty_like(images, device='cuda:0')
    different_images[batch_size-1] = images[0]
    different_images[:batch_size-1] = images[1:]
    return different_images

def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    # hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    print()
    return hull_mask
def get_w_image_tensor(w, generator):
    w = w.to('cuda:0')

    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True, randomize_noise=True
    )
    new_image = (sample + 1) / 2

    return new_image

def get_w_image_tensor_batch(w, generator):
    w = w.to('cuda:0')

    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True, randomize_noise=True
    )
    new_image = (sample + 1) / 2

    return new_image


class Image_Mask_Dataset(Dataset):
    def __init__(self, w_dir, image_dir):
        self.w_dir = w_dir
        self.image_dir = image_dir


    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.w_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        image = get_data_by_index(idx, self.image_dir, ".png")

        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_idx = idx // 1000

        save_path = os.path.join(str(dir_idx), str(idx))
        return image, save_path
    

def tensor_erode(bin_img, ksize=5):
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=1)

    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    # 取每个 patch 中最小的值，i.e., 0
    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


class ImageWithMaskDataset(Dataset):
    #输出的mask和img
    def __init__(self, w_dir, image_dir, mask_dir):
        self.w_dir = w_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir


    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.w_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        w = get_data_by_index(idx, self.w_dir, ".npy")
        image = get_data_by_index(idx, self.image_dir, ".png")
        mask = get_data_by_index(idx, self.mask_dir, ".png")
        return w, image, mask

def get_masked_imgs(fg_imgs,bg_imgs, masks):
    masks = 1-tensor_erode(1-masks)
    b,c,h,w = fg_imgs.shape
    fg_imgs = fg_imgs.cpu().permute(0,2,3,1).numpy()
    bg_imgs = bg_imgs.cpu().permute(0,2,3,1).numpy()
    masks = masks.cpu().permute(0,2,3,1).numpy()

#########################
    for idx in range(b):
        fg_img = cv2.cvtColor(fg_imgs[idx], cv2.COLOR_RGB2BGR)
        bg_img = cv2.cvtColor(bg_imgs[idx], cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(masks[idx], cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f'/home/yl/lk/code/ID-dise4/metrics/{str(idx)}_erode.png', mask*255)
        mask =cv2.GaussianBlur(mask, (15, 15), -1)
        # cv2.imwrite(f'/home/yl/lk/code/ID-dise4/metrics/{str(idx)}_gBlur.png', mask*255)
        fina_img = fg_img*mask + (1-mask)*bg_img
        fina_img = cv2.cvtColor(fina_img, cv2.COLOR_BGR2RGB)
        fina_img = np.expand_dims(fina_img, axis=0)
        
        if idx == 0:
            fina_imgs = fina_img
        else:
            fina_imgs = np.concatenate((fina_imgs,fina_img),axis=0)
    # print(fina_imgs.shape)
    fina_imgs = torch.from_numpy(fina_imgs).type(torch.float32).permute(0,3,1,2).to('cuda:0')
    
# ######################### 
#     fg_imgs = np.array(fg_imgs*255,np.uint8)
#     bg_imgs = np.array(bg_imgs*255,np.uint8)
#     masks = np.array(masks*255,np.uint8)
#     # if fg_imgs.any()>255:
#     #     fg_imgs = 255
#     # elif fg_imgs.any()<0:
#     #     fg_imgs = 0

#     # if bg_imgs.any()>255:
#     #     bg_imgs = 255
#     # elif bg_imgs.any()<0:
#     #     bg_imgs = 0

#     # if masks.any()>255:
#     #     masks = 255
#     # elif masks.any()<0:
#     #     masks = 0

    
#     # print(fg_imgs.shape, type(fg_imgs))
#     # print(bg_imgs.shape, type(bg_imgs))
#     # print(masks.shape, type(masks))

#     for idx in range(b):

#         # print(fg_imgs[idx])
#         fg_img = cv2.cvtColor(fg_imgs[idx], cv2.COLOR_RGB2BGR)
#         bg_img = cv2.cvtColor(bg_imgs[idx], cv2.COLOR_RGB2BGR)
#         mask = cv2.cvtColor(masks[idx], cv2.COLOR_RGB2BGR)


#         # print(f'input image shape  is  {bg_img.shape, fg_imgs.shape, mask.shape}')
#         # fg_img = np.squeeze(fg_img, axis=0)
#         # fg_img = resize_128(fg_img)
#         # mask = resize_128(mask)
#         # print(f'input image shape  is  {bg_img.shape, fg_imgs.shape, mask.shape}')
#         # print(f'idx = {idx}')
#         # mask =cv2.GaussianBlur(mask, (7, 7), -1)
#         # mask =cv2.GaussianBlur(mask, (15, 15), -1)
#         fina_img = cv2.seamlessClone(fg_img, bg_img, mask, (128,128), cv2.NORMAL_CLONE)#MIXED_CLONE,NORMAL_CLONE
#         fina_img = cv2.resize(fina_img,(256,256))

#         # fina_img = resize_128(fina_img)
#         # print(f'input image shape  is  {bg_img.shape, fg_imgs.shape, mask.shape}')
#         # a
        
#         # fina_img = cv2.resize(fina_img,(128,128))




#         # detail_MultiBandBlender
#         # mask = mask.astype(np.uint8)
#         # print(f'input image shape  is  {bg_img.shape, fg_imgs.shape, mask.shape} , and type is {type(bg_img), type(fg_img), type(mask)}')
#         # mask = np.uint8(mask)
#         # blender=cv2.detail_MultiBandBlender()
#         # roiRect=(0, 0, fg_img.shape[1], fg_img.shape[0])
#         # blender.prepare(roiRect)
#         # blender.feed(fg_img, mask, (0,0))
#         # blender.feed(bg_img, 255-mask, (0,0))
#         # imgBlendAO=fg_img.copy()
#         # fina_img,dst_mask=blender.blend(imgBlendAO, mask)
#         # fina_img = cv2.resize(fina_img,(128,128))


#         fina_img = cv2.cvtColor(fina_img, cv2.COLOR_BGR2RGB)
#         fina_img = np.expand_dims(fina_img, axis=0)
        
#         if idx == 0:
#             fina_imgs = fina_img
#         else:
#             fina_imgs = np.concatenate((fina_imgs,fina_img),axis=0)

#     fina_imgs = torch.from_numpy(fina_imgs/255).type(torch.float32).permute(0,3,1,2).to('cuda:0')
#     # print(f'fina_imgs.shape = {fina_imgs.shape, type(fina_imgs), }')
#     # ##########################
        
    return fina_imgs



class Image_Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.image_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, str(self.images[idx]))
        data = to_tensor_transform(Image.open(path))

        save_path = str(self.images[idx])
        return data, save_path
    
class ImageAndMask_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(self.image_dir)
        self.images.sort()
        self.images = self.images
        # print(f'self.images = {self.images}')

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.image_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, str(self.images[idx]))
        mask_path = os.path.join(self.mask_dir, str(self.images[idx]))
        # print(f'path = {path}')
        data = to_tensor_transform(Image.open(path))
        mask = to_tensor_transform(Image.open(mask_path))

        return data, mask

class Change_Face_Dataset(Dataset):
    """
    同时导入前景、后景、掩膜
    """
    def __init__(self, image_dir, bg_img_dir, mask_dir):
        self.image_dir = image_dir
        self.bg_img_dir = bg_img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(self.image_dir)
        self.images = self.images

        # print(f'self.images = {self.images}')

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.image_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, str(self.images[idx]))
        bg_img = os.path.join(self.bg_img_dir, str(self.images[idx]))
        mask_path = os.path.join(self.mask_dir, str(self.images[idx]))

        # print(f'path = {path}')
        data = to_tensor_transform(Image.open(path))
        bg_img = to_tensor_transform(Image.open(bg_img))
        mask = to_tensor_transform(Image.open(mask_path))


        return data, bg_img, mask



# if __name__ == '__main__':

#     import torchvision
#     from torchvision.transforms import Resize
#     totensor = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])

#     raw_img = cv2.imread('/userHOME/yl/lk_data/face_disc/small_mask/0/0.png')
#     raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
#     raw_img = cv2.resize(raw_img,(256,256),interpolation=cv2.INTER_CUBIC)
#     raw_img = totensor(raw_img)
#     bg_img = raw_img.unsqueeze(0).to('cuda:0')    


#     manual_img = cv2.imread(fg_path+str(test_idx)+'.png')
#     manual_img = cv2.cvtColor(manual_img, cv2.COLOR_BGR2RGB)
#     manual_img = cv2.resize(manual_img,(256,256),interpolation=cv2.INTER_CUBIC)
#     manual_img = totensor(manual_img)
#     fg_img = manual_img.unsqueeze(0).to('cuda:0')

#     mask = cv2.imread(fg_path+str(test_idx)+'.png')
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#     mask = cv2.resize(mask,(256,256),interpolation=cv2.INTER_CUBIC)
#     mask = totensor(mask)
#     fg_img = mask.unsqueeze(0).to('cuda:0')

import random
class ImagePool():
    """
    To make the discriminator learn the distribution
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images, batch_size=None):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # change image pool [random_id] to image, return old image to be replaced
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # image pool is not changed, return image
                    return_images.append(image)

        if batch_size is not None:
            cur_len = len(return_images)
            indices = np.random.choice(cur_len, batch_size, replace=False)
            return_images = [item for idx,item in enumerate(return_images) if idx in indices]
        return_images = torch.cat(return_images, 0)
        return return_images
