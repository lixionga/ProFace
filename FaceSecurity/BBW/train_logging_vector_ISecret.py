#!/usr/bin/env python
import datetime
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score
from utils import *
import random
import string
import kornia
import numpy as np
import torch.nn
import torch.optim
from lpips import lpips
from torchvision.transforms import transforms
import torch.nn.functional as F
from network.discriminator import Discriminator
from network.Vector import vector_var
from network.distortions.benign.diff_jpeg.jpeg import DiffJPEGCoding
import network.Unet_common as common
from tensorboardX import SummaryWriter
import logging
from torchvision import datasets
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import ROC, F1Score, Recall
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from network.model import *
from network.distortions.deepfakes.selfblended.SelfBlendedImage.src.utils.sbi import SBI

from network.distortions.deepfakes.FaceShifter.face_modules.model import Backbone
from network.distortions.deepfakes.FaceShifter.face_modules.mtcnn import MTCNN
from network.distortions.deepfakes.FaceShifter.network.AEI_Net import AEI_Net


device = torch.device(c.device if torch.cuda.is_available() else "cpu")


def seed_torch(seed=25):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)
    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduction="sum")
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduction="sum")
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduction="sum")
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    discriminator.load_state_dict(state_dicts['discriminator'])
    template_init.load_state_dict(state_dicts['template_init'])
    try:
        optim.load_state_dict(state_dicts['opt'])
        optim_d.load_state_dict(state_dicts['optim_d'])
        optim_template.load_state_dict(state_dicts['optim_template_init'])
    except:
        print('Cannot load optimizer for some reason or other')

def differentiable_JPEG(img, quality):
    img = img * 255
    quality = torch.tensor(quality).repeat(img.size()[0]).to(device)
    img_jpeg = jpge_model.forward(img, jpeg_quality=quality).to(device)  # 输入转换为[0-255]
    return img_jpeg / 255

def image_brightness(steg_img, brightness):
    return torchvision.transforms.functional.adjust_brightness(steg_img, brightness)

def image_saturation(steg_img, saturation):
    return  torchvision.transforms.functional.adjust_saturation(steg_img, saturation)

def image_blur(steg_img, blur_sigma):
    blur_transform = kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(blur_sigma, blur_sigma), p=1)
    return blur_transform(steg_img)


def image_resize(steg_img, down_scale=0.5):
    image = steg_img
    down = F.interpolate(image, size=(int(down_scale * image.shape[2]), int(down_scale * image.shape[3])),
                         mode='nearest')
    up = F.interpolate(down, size=(image.shape[2], image.shape[3]), mode='nearest')
    return up

def image_gaussnoise(image, std=0.1):
    transform_gauss = kornia.augmentation.RandomGaussianNoise(mean=0, std=std, p=1)
    return transform_gauss(image)

def image_jpeg(img, quality):
    B, _, _, _ = img.shape
    img = img.mul(255).add_(0.5).clamp_(0, 255)
    quality = torch.tensor(quality).repeat(B, 1).to(device)

    # Init list to store compressed images
    image_rgb_jpeg = []
    # Code each image
    for index in range(B):
        # Make encoding parameter
        encode_parameters = (int(cv2.IMWRITE_JPEG_QUALITY), int(quality[index].item()))
        # Encode image note CV2 is using [B, G, R]
        _, encoding = cv2.imencode(".jpeg", img[index].flip(0).permute(1, 2, 0).cpu().numpy(), encode_parameters)
        image_rgb_jpeg.append(torch.from_numpy(cv2.imdecode(encoding, 1)).permute(2, 0, 1).flip(0))

    # Stack images
    image_rgb_jpeg = torch.stack(image_rgb_jpeg, dim=0).float() / 255

    return image_rgb_jpeg.to(device)


def rand_distortion_train(steg_img):
    p = random.uniform(0, 1)
    dis_type = random.randint(0, 5)
    res = steg_img

    if p <= 0.85:  # 85%的概率发生变换
        if dis_type == 0:
            arg = random.uniform(1, 2)
            res = image_blur(steg_img, arg)
        elif dis_type == 1:
            arg = random.uniform(0.0, 0.05)
            res = image_gaussnoise(steg_img, arg)
        elif dis_type == 2:
            arg = random.uniform(0.5, 1)
            res = image_resize(steg_img, arg)
        elif dis_type == 3:
            arg = random.choice([-1, 1]) * random.uniform(0.0, 0.2) + 1
            res = image_saturation(steg_img, arg)
        elif dis_type == 4:
            arg = random.choice([-1, 1]) * random.uniform(0.0, 0.2) + 1
            res = image_brightness(steg_img, arg)
        else:
            arg = [50, 75, 95]
            res = differentiable_JPEG(steg_img, random.choice(arg))
    return res


def image_identity(image, param=None):
    return image


def val_distortion(img, step):
    transformations = [
        (image_blur, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
        (image_gaussnoise, [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]),
        (image_resize, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
        (image_jpeg, [50, 75, 95]),
        (image_brightness, [0.8, 0.85, 0.9, 0.95, 1.05, 1.10, 1.15, 1.20]),
        (image_saturation, [0.8, 0.85, 0.9, 0.95, 1.05, 1.10, 1.15, 1.20]),
        (image_identity, [None])
    ]

    flat_transformations = [(func, param) for func, params in transformations for param in params]

    total_transforms = len(flat_transformations)
    index = step % total_transforms
    func, param = flat_transformations[index]

    transformed_img = func(img, param)

    return transformed_img


# 实现tensor滚动
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    # print(axis,n,f_idx,b_idx)
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift(real, imag):
    for dim in range(1, len(real.size())):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return real, imag

def trans_label(pre):
    return (pre > 0.5).float()

def set_requires_grad(nets, requires_grad=False):
    for param in nets.parameters():
        param.requires_grad = requires_grad

def get_target(input_dataset, classes=0, size=8):
    index = [i for i, t in enumerate(input_dataset.targets) if t == classes]
    rand_inds = np.random.choice(index, size=size, replace=False)
    imgs = [input_dataset[i][0] for i in rand_inds]
    return torch.stack(imgs).to(device)

def get_image_by_class_and_index(target_dataset, classes, idx):
    # 找到所有属于指定类别的图片索引
    indices = [i for i, target in enumerate(target_dataset.targets) if target == classes]
    # 使用模运算获取有效的索引
    valid_idx = idx % len(indices)
    target_index = indices[valid_idx]  # 防止 valid_idx 超出当前类别的图片数量
    # 提取并返回对应的图片
    image, label = target_dataset[target_index]
    return torch.tensor(image).unsqueeze(0).to(device)


def batch_selfBlended(batch_img, landmark):
    res = batch_img
    for i in range(batch_img.size()[0]):

        tmp = SBI_model.forward(batch_img[i], landmark[i])

        trans_t = data_transform(tmp)
        if trans_t.size() != (3, c.cropsize_val, c.cropsize_val):
            trans_t = F.interpolate(trans_t.unsqueeze(0), (256, 256), mode='nearest', align_corners=True)
            res[i] = trans_t.squeeze(0)
        else:
            res[i] = trans_t
    return res


def face_shifter(xt, xs):
    xt = face_shifter_transform(xt)
    xs = face_shifter_transform(xs)
    _, _, w, h = xt.size()
    with torch.no_grad():
        embeds = arcface(F.interpolate(xs, (112, 112), mode='bilinear', align_corners=True))
        yt, _ = G(xt, embeds)
        return (yt + 1) / 2


def nonLinearTrans(input_image):
    input_image = input_image + 0.5
    input_image = torch.pow(input_image, 2.0)
    return input_image.clamp(0, 1)


# Model initialize: #
seed_torch(25)

net = Model().to(device)
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)

para = get_parameter_number(net)
print(para)

# self blend model init
SBI_model = SBI(image_size=c.cropsize).to(device)

# faceshifter model init
detector = MTCNN()
G = AEI_Net(c_id=512)
G.eval()
G.load_state_dict(torch.load('./network/distortions/deepfakes/FaceShifter/saved_models/G_latest.pth', map_location=device))
G = G.to(device)
arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(
    torch.load('./network/distortions/deepfakes/FaceShifter/face_modules/model_ir_se50.pth', map_location=device),
    strict=False)
face_shifter_transform = transforms.Compose([  # 传入faceshifter之前将数据从[0,1]->[-1,1]
    transforms.Normalize(mean=0.5, std=0.5)
])

# JPEG simulator
jpge_model = DiffJPEGCoding(ste=True).to(device)

discriminator = Discriminator(input_shape=(c.channels_in, c.cropsize, c.cropsize)).to(device)

template_init = vector_var(size=c.template_size).to(device)
template = template_init()
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

optim_template = torch.optim.Adam(template_init.parameters(), lr=c.template_lr, weight_decay=c.weight_decay)
optim_d = torch.optim.Adam(discriminator.parameters(), lr=c.discriminator_lr, weight_decay=c.weight_decay)
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

sig = str(datetime.datetime.now())

acc = BinaryAccuracy().to(device)
roc = ROC(task="binary", thresholds=[0.5]).to(device)
F1 = F1Score(task="binary", threshold=0.5).to(device)
recall = Recall(task="binary").to(device)
SSIM = StructuralSimilarityIndexMeasure().to(device)

# loss
BCELoss = torch.nn.BCELoss(reduction="sum").to(device)
LPIPSLoss = lpips.LPIPS(net='vgg').to(device)

data_transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])

# deepfake target datasets
target_dir = c.TARGET_PATH
target_dataset = datasets.ImageFolder(root=target_dir, transform=data_transform)

dwt = common.DWT()
iwt = common.IWT()

if c.train_next:
    load(c.MODEL_PATH_FRE + c.suffix_fre)

logger_train = logging.getLogger('train')
logger_train.info(net)

try:
    writer = SummaryWriter(comment=c.MODEL_DESC, filename_suffix="steg")

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1

        loss_history = []
        g_loss_history = []
        r_loss_history = []
        l_secret_loss_history = []
        discriminator_loss_history = []

        acc.reset()
        F1.reset()
        recall.reset()

        #################
        #     train:    #
        #################
        for i_batch, data in enumerate(basedatasets.trainloader):
            image = data[0].to(device)
            landmark = data[1]

            cover = image
            secret = template.repeat([cover.size()[0], 1, 1, 1])

            cover_input = dwt(cover)
            secret_input = dwt(secret)

            input = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input)

            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            steg_img = iwt(output_steg)

            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)

            steg_nm = rand_distortion_train(steg_img)
            steg_nm_input = dwt(steg_nm)

            target = get_target(target_dataset, classes=1, size=cover.size()[0])

            steg_mp = batch_selfBlended(steg_img.clamp(0, 1), landmark)
            steg_mp = rand_distortion_train(steg_mp)

            steg_mp_input = dwt(steg_mp)

            #################
            #   backward:   #
            #################
            output_z_guass = gauss_noise(output_z.shape)

            output_rev = torch.cat((steg_nm_input, output_z_guass), 1)
            output_rev_mp = torch.cat((steg_mp_input, output_z_guass), 1)

            output_image = net(output_rev, rev=True)
            output_image_mp = net(output_rev_mp, rev=True)

            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)

            secret_rev_mp = output_image_mp.narrow(1, 4 * c.channels_in, output_image_mp.shape[1] - 4 * c.channels_in)
            secret_rev_mp = iwt(secret_rev_mp)

            diff_nm_secret = secret - secret_rev
            diff_mp_secret = secret - secret_rev_mp

            #################
            #     loss:     #
            #################

            #################
            # discriminator #
            #################
            pred_secret_rev = discriminator.forward(diff_nm_secret).to(device).reshape(-1)
            pred_secret_fake = discriminator.forward(diff_mp_secret).to(device).reshape(-1)

            valid_label_l = torch.ones(cover.size()[0]).to(device)  # 真实标签，都是1
            fake_label_l = torch.zeros(cover.size()[0]).to(device)  # 真实标签，都是0

            target_label = torch.cat((valid_label_l, fake_label_l), dim=0)
            pred_label = torch.cat((pred_secret_rev, pred_secret_fake), dim=0)

            d_loss = BCELoss(pred_label, target_label)

            #################
            #     HiNet     #
            #################
            g_loss = guide_loss(steg_img.to(device), cover.to(device))

            cover_low = cover_input.narrow(1, 0, c.channels_in)
            secret_low = secret_input.narrow(1, 0, c.channels_in)
            l_secret_loss = low_frequency_loss(secret_low, cover_low)

            r_loss = reconstruction_loss(secret, secret_rev)

            total_loss = c.lamda_guide * g_loss + c.lamda_reconstruction * r_loss + \
                         c.lamda_discriminator * d_loss + c.lamda_low_frequency_secret * l_secret_loss

            optim.zero_grad()
            optim_template.zero_grad()
            optim_d.zero_grad()

            # 反向传播
            total_loss.backward()

            optim.step()
            optim_template.step()
            optim_d.step()

            g_loss_history.append(g_loss.detach().cpu().numpy())
            r_loss_history.append(r_loss.detach().cpu().numpy())
            l_secret_loss_history.append(l_secret_loss.detach().cpu().numpy())
            discriminator_loss_history.append(d_loss.detach().cpu().numpy())

        r_epoch_losses = np.mean(r_loss_history)
        g_epoch_losses = np.mean(g_loss_history)
        l_epoch_secret_losses = np.mean(l_secret_loss_history)
        discriminator_epoch_losses = np.mean(discriminator_loss_history)

        #################
        #      val      #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                psnr_s_f = []
                psnr_s = []
                psnr_c = []
                aucAll = []
                ssim_c = []
                lpips_c = []

                net.eval()
                discriminator.eval()
                template_init.eval()

                acc.reset()
                recall.reset()
                F1.reset()

                saved_iterations = np.random.choice(np.arange(1, len(basedatasets.valloader) + 1), size=4,
                                                    replace=False)
                saved_all = None

                for i, x in enumerate(basedatasets.valloader):
                    cover = x.to(device)
                    secret = template.repeat([cover.size()[0], 1, 1, 1])

                    cover_input = dwt(cover)
                    secret_input = dwt(secret)

                    input_img = torch.cat((cover_input, secret_input), 1)

                    #################
                    #    forward:   #
                    #################
                    output = net(input_img)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)

                    steg = iwt(output_steg)

                    steg_nm = val_distortion(steg, step=i)
                    steg_nm_input_val = dwt(steg_nm)

                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)

                    # classes = 0->test,1->val
                    target = get_image_by_class_and_index(target_dataset, classes=1, idx=i)

                    steg_mp_val = face_shifter(steg, target)
                    steg_mp_val = val_distortion(steg_mp_val, step=i)

                    steg_mp_input_val = dwt(steg_mp_val)

                    #################
                    #   backward:   #
                    #################
                    input_rev = torch.cat((steg_nm_input_val, output_z), 1)
                    input_rev_mp_val = torch.cat((steg_mp_input_val, output_z), 1)

                    output_image = net(input_rev, rev=True)
                    output_image_mp = net(input_rev_mp_val, rev=True)

                    secret_rev_val = output_image.narrow(1, 4 * c.channels_in,
                                                         output_image.shape[1] - 4 * c.channels_in)
                    secret_rev_val = iwt(secret_rev_val)

                    secret_rev_mp_val = output_image_mp.narrow(1, 4 * c.channels_in,
                                                               output_image_mp.shape[1] - 4 * c.channels_in)
                    secret_rev_mp_val = iwt(secret_rev_mp_val)
                    ######################
                    # backward Forensics #
                    ######################
                    diff_nm_secret_val = secret - secret_rev_val
                    diff_mp_secret_val = secret - secret_rev_mp_val

                    valid_label_val = torch.ones(cover.size()[0], dtype=torch.long).to(device)  # 真实标签，都是1
                    fake_label_val = torch.zeros(cover.size()[0], dtype=torch.long).to(device)  # 假标签，都是0

                    all_inputs = torch.cat((diff_nm_secret_val, diff_mp_secret_val), dim=0)
                    target_labels = torch.cat((valid_label_val, fake_label_val), dim=0)

                    preds = discriminator(all_inputs).reshape(-1)

                    fpr, tpr, _ = roc(preds, target_labels)

                    acc.update(preds, target_labels)
                    F1.update(preds, target_labels)
                    recall.update(preds, target_labels)

                    aucAll.append(roc_auc_score(target_labels.cpu().numpy(), preds.cpu().numpy()))

                    psnr_temp_c = - kornia.losses.psnr_loss(cover.detach(), steg.detach(), 2)
                    psnr_c.append(psnr_temp_c.cpu().numpy())

                    psnr_temp_s = - kornia.losses.psnr_loss(secret.detach(), secret_rev_val.detach(), 2)
                    psnr_s.append(psnr_temp_s.cpu().numpy())

                    psnr_temp_s_f = - kornia.losses.psnr_loss(secret.detach(), secret_rev_mp_val.detach(),
                                                              2)
                    psnr_s_f.append(psnr_temp_s_f.cpu().numpy())

                    ssim_temp = 1 - 2 * kornia.losses.ssim_loss(cover.detach(), steg, window_size=11, reduction="mean")
                    ssim_c.append(ssim_temp.cpu().numpy())

                    lpips_temp = LPIPSLoss(cover.detach().to(device), steg.detach().to(device))
                    lpips_c.append(lpips_temp.cpu().numpy())

                    if i in saved_iterations:
                        if saved_all is None:
                            saved_all = get_random_images(cover, secret, steg, steg_mp_val,
                                                          nonLinearTrans(diff_nm_secret_val),
                                                          nonLinearTrans(diff_mp_secret_val))
                        else:
                            saved_all = concatenate_images(saved_all, cover, secret, steg, steg_mp_val,
                                                          nonLinearTrans(diff_nm_secret_val),
                                                          nonLinearTrans(diff_mp_secret_val))

                # 保存图片
                save_images(saved_all, c.MODEL_PATH_FRE, "epoch: " + str(i_epoch))

                print(
                    "epoch={},r_loss={:.4f},g_loss={:.4f},l_loss={:.4f},discriminator_loss={}".format(
                        i_epoch,
                        r_epoch_losses,
                        g_epoch_losses,
                        l_epoch_secret_losses,
                        discriminator_epoch_losses
                    ))

                logger_train.info(
                    f"TEST:   "
                    f'ACC: {acc.compute():.4f} | '
                    f'F1: {F1.compute():.4f} | '
                    f'AUC: {np.mean(aucAll):.4f} | '
                    f'PSNR_C: {np.mean(psnr_c):.4f} | '
                    f'SSIM_C: {np.mean(ssim_c):.4f} | '
                    f'LPIPS_C: {np.mean(lpips_c):.4f} | '
                    f'PSNR_S: {np.mean(psnr_s):.4f} | '
                    f'PSNR_S_F: {np.mean(psnr_s_f):.4f} | '
                )

                # TenosrBoard
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_S_F", {"average psnr": np.mean(psnr_s_f)}, i_epoch)
                writer.add_scalars("SSIM_C", {"average ssim": np.mean(ssim_c)}, i_epoch)
                writer.add_scalars("LPIPS_C", {"average lpips": np.mean(lpips_c)}, i_epoch)
                writer.add_scalars("ACC", {"ACC": acc.compute()}, i_epoch)
                writer.add_scalars("F1", {"F1": F1.compute()}, i_epoch)
                writer.add_scalars("AUC", {"AUC": np.mean(aucAll)}, i_epoch)

        # save model
        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'optim_template_init': optim_template.state_dict(),
                        'template_init': template_init.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'net': net.state_dict()},
                       c.MODEL_PATH_FRE + 'HiNet_patchGAN_model_checkpoint_%.5i' % i_epoch + '.pt')
        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'optim_d': optim_d.state_dict(),
                'optim_template_init': optim_template.state_dict(),
                'template_init': template_init.state_dict(),
                'discriminator': discriminator.state_dict(),
                'net': net.state_dict()}, c.MODEL_PATH_FRE + 'model' + '.pt')

    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'optim_template_init': optim_template.state_dict(),
                    'template_init': template_init.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH_FRE + 'model_ABORT' + '.pt')
    raise
