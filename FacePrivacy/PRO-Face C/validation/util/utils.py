import random
import os
from torch.utils.data import Dataset, DataLoader

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from datetime import datetime
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io
from validation.util.verification import evaluate

random.seed(56)


# Gaussian
class Blur(torch.nn.Module):
    def __init__(self, kernel_size, sigma_min, sigma_max):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_avg = (sigma_min + sigma_max) / 2

    def forward(self, img, _=None):
        sigma = self.sigma_avg
        img_blurred = F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img_blurred


# pixelate
class Pixelate(torch.nn.Module):
    def __init__(self, block_size_avg):
        super().__init__()
        if not isinstance(block_size_avg, int):
            raise ValueError("block_size_avg must be int")
        self.block_size_avg = block_size_avg
        self.block_size_min = block_size_avg - 4
        self.block_size_max = block_size_avg + 4

    def forward(self, img, _=None):
        img_size = img.shape[-1]
        block_size = self.block_size_avg
        pixelated_size = img_size // block_size
        img_pixelated = F.resize(F.resize(img, pixelated_size), img_size, F.InterpolationMode.NEAREST)
        return img_pixelated


# median
class MedianBlur(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel = kernel_size
        self.size_min = kernel_size - 7
        self.size_max = kernel_size + 7
        self.number = [7, 9, 11, 13, 15, 17, 19]

    def forward(self, img, _=None):
        kernel_size = self.kernel
        img_blurred = kornia.filters.median_blur(img, (kernel_size, kernel_size))
        return img_blurred

# hybrid
class MyBlurWay(object):
    def __init__(self, patch, size):
        import math
        patch_r = int(math.sqrt(patch))
        assert (patch_r * patch_r) == patch, "error!"
        self.patch = patch
        self.patch_r = patch_r
        self.size = size

    def __call__(self, img):
        img = img.cuda()
        p = int(self.size / self.patch_r)
        from torchvision import transforms
        if p * self.patch_r != self.size:
            resize_img = transforms.Resize((p + 1) * self.patch_r)
            img = resize_img(img).cuda()
            p += 1

        medianBlur = MedianBlur(13)
        med_image = medianBlur(img)

        GausBlur = Blur(31, 2, 8)
        gaus_image = GausBlur(img)

        PixBlur = Pixelate(7)
        pix_image = PixBlur(img)

        from einops.layers.torch import Rearrange

        to_patch_embedding = Rearrange('b c (h p1) (w p2) ->b (h w) c p1 p2', p1=p, p2=p)
        med_img_list = []
        gaus_img_list = []
        pix_img_list = []
        med_img_patch = to_patch_embedding(med_image)
        gaus_img_patch = to_patch_embedding(gaus_image)
        pix_img_patch = to_patch_embedding(pix_image)

        for i in range(0, self.patch):
            indices = torch.tensor([i]).cuda()
            tmp_med = torch.index_select(med_img_patch, 1, indices)
            tmp_gaus = torch.index_select(gaus_img_patch, 1, indices)
            tmp_pix = torch.index_select(pix_img_patch, 1, indices)
            med_img_list.append(tmp_med)
            gaus_img_list.append(tmp_gaus)
            pix_img_list.append(tmp_pix)

        index_list = random.sample(range(0, self.patch), self.patch)

        blur_img_list = self.patch * [None]
        index_tmp = int(self.patch / 3)
        for i in range(0, self.patch):
            tmp = index_list[i]
            if i < index_tmp:
                blur_img_list[tmp] = med_img_list[tmp]
            elif index_tmp <= i < index_tmp * 2:
                blur_img_list[tmp] = gaus_img_list[tmp]
            else:
                blur_img_list[tmp] = pix_img_list[tmp]

        truple_image = tuple(blur_img_list)
        tmp_blur_img = torch.cat(truple_image, dim=1)
        to_patch_embedding = Rearrange('b (h w) c p1 p2  -> b c (h p1) (w p2)', h=self.patch_r, w=self.patch_r)
        blur_image = to_patch_embedding(tmp_blur_img)

        resize_img_two = transforms.Resize(self.size)
        blur_image = resize_img_two(blur_image).cuda()

        return blur_image


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')

    return lfw, cfp_fp, agedb_30, calfw, cplfw, lfw_issame, cfp_fp_issame, agedb_30_issame, calfw_issame, cplfw_issame

def get_test_celeA(data_path, pairs_file):
    img_list = []
    issame = []

    pairs_file_buf = open(pairs_file)
    line = pairs_file_buf.readline()
    line = pairs_file_buf.readline().strip()
    while line:
        line_strs = line.split('\t')
        if len(line_strs) == 3:
            person_name = line_strs[0]
            image_index1 = line_strs[1]
            image_index2 = line_strs[2]
            image_name1 = data_path + '/' + image_index1
            image_name2 = data_path + '/' + image_index2
            label = 1
        elif len(line_strs) == 4:
            person_name1 = line_strs[0]
            image_index1 = line_strs[1]
            person_name2 = line_strs[2]
            image_index2 = line_strs[3]
            image_name1 = data_path + '/' + image_index1
            image_name2 = data_path + '/' + image_index2
            label = 0
        else:
            raise Exception('Line error: %s.' % line)

        img_list.append(image_name1)
        img_list.append(image_name2)
        if label == 1:
            issame.append(True)
        else:
            issame.append(False)

        line = pairs_file_buf.readline().strip()

    issame = np.array(issame)
    return img_list, issame


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


def hflip_batch(imgs_tensor):
    """ bacth data Horizontally flip
    """
    hflip = transforms.Compose([
        de_preprocess,
        transforms.ToPILImage(),
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


# --------------crrop-------------------------------
def ccrop_batch(imgs_tensor):
    """crop image tensor
    """
    ccrop = transforms.Compose([
        de_preprocess,
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


# --------------blurred-------------------------------
import torchvision.transforms.functional as F
import kornia


def Gaus_blur(image, kernel_size=31, sigma=5):
    trans_blur = transforms.GaussianBlur(kernel_size, sigma)
    img_blurred = trans_blur(image)
    return img_blurred


def Pixel_blur(image, block_size=7):
    img_size = image.shape[-1]
    pixelated_size = img_size // block_size
    img_pixelated = F.resize(F.resize(image, pixelated_size), img_size, F.InterpolationMode.NEAREST)
    return img_pixelated


def Median_blur(image, kernel_size=13):
    img_blurred = kornia.filters.median_blur(image, (kernel_size, kernel_size))
    return img_blurred


def blur_batch(imgs_tensor, blur_type=0, hflip=False):
    if hflip:
        ccrop = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        ccrop = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    blured_imgs = torch.empty_like(imgs_tensor)

    for i, img_ten in enumerate(imgs_tensor):
        tmp = ccrop(img_ten)
        # C H W -> B C H W
        tmp = tmp.unsqueeze(0)
        tmp = tmp.cuda()

        if blur_type == 0:
            img = Gaus_blur(tmp)
        elif blur_type == 1:
            img = Median_blur(tmp)
        elif blur_type == 2:
            img = Pixel_blur(tmp)
        elif blur_type == 3:
            patch = 16
            Blur = MyBlurWay(patch, 112)
            img = Blur(tmp)

        # B C H W -> C H W
        blured_imgs[i] = img.squeeze(dim=0)

    return blured_imgs
# ---------------------------------------------

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()

    return buf


def perform_val(embedding_size, batch_size, backbone, carray, issame,
                nrof_folds=10, aux_model=None, blur_type=0):

    backbone.eval()

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            ccropped = ccrop_batch(batch)
            ccropped = ccropped.cuda()
            blured = blur_batch(batch, blur_type, hflip=False)
            blured = blured.cuda()

            res_imge = ccropped - blured

            aux_feature, aux_list, key_list = backbone(res_imge.cuda())
            privacy_feature, _ = aux_model(blured.cuda(), aux_list, key_list)

            emb_batch = privacy_feature.cpu()

            embeddings[idx:idx + batch_size] = l2_norm(emb_batch.cpu())
            idx += batch_size

        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            ccropped = ccrop_batch(batch)
            ccropped = ccropped.cuda()
            blured = blur_batch(batch, blur_type, hflip=False)
            blured = blured.cuda()

            res_imge = ccropped - blured
            aux_feature, aux_list, key_list = backbone(res_imge.cuda())
            privacy_feature, _ = aux_model(blured.cuda(), aux_list, key_list)

            emb_batch = privacy_feature.cpu()

        embeddings[idx:] = l2_norm(emb_batch.cpu())

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)

    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


# CelebA Data
class getCelebAtest(Dataset):
    def __init__(self, img_path_list, transform=None):
        self.transform = transform
        self.img_path_list = img_path_list

    def __getitem__(self, idx):
        img_item_path = self.img_path_list[idx]

        img = Image.open(img_item_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path_list)


# CelebA_test
def perform_test_celebA(embedding_size, batch_size, backbone, img_list, issame,
                        nrof_folds=10, aux_model=None, blur_type=0):

    embeddings = np.zeros([len(img_list), embedding_size])
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        datasets = getCelebAtest(img_list, transform)
        test_dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=16, shuffle=False)

        idx = 0
        for i, imgs in enumerate(test_dataloader):
            if blur_type == 0:
                blur_imgs = Gaus_blur(imgs)
            elif blur_type == 1:
                blur_imgs = Median_blur(imgs)
            elif blur_type == 2:
                blur_imgs = Pixel_blur(imgs)
            elif blur_type == 3:
                patch = 16
                Blur = MyBlurWay(patch, 112)
                blur_imgs = Blur(imgs)
                blur_imgs = blur_imgs.cuda()
                imgs = imgs.cuda()

            if idx + batch_size <= len(img_list):
                res_imge = imgs - blur_imgs
                aux_feature, aux_list, key_list = backbone(res_imge.cuda())
                privacy_feature, _ = aux_model(blur_imgs.cuda(), aux_list, key_list)

                emb_batch = privacy_feature.cpu()

                embeddings[idx:idx + batch_size] = l2_norm(emb_batch.cpu())
                idx += batch_size
            else:
                res_imge = imgs - blur_imgs
                aux_feature, aux_list, key_list = backbone(res_imge.cuda())
                privacy_feature, _ = aux_model(blur_imgs.cuda(), aux_list, key_list)

                emb_batch = privacy_feature.cpu()

                embeddings[idx:] = l2_norm(emb_batch.cpu())

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch):
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
    writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
