from embedder import dwt, iwt, ModelDWT
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from utils.utils_train import normalize, gauss_noise
from utils.image_processing import Obfuscator, input_trans, rgba_image_loader
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError, PeakSignalNoiseRatio
from utils.loss_functions import lpips_loss
from utils.utils_func import get_parameter_number
import config.config as c
import random, string
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA512
import sys
import json
from tqdm import tqdm
import time

sys.path.append(os.path.join(c.DIR_PROJECT, 'SimSwap'))

DIR_HOME = os.path.expanduser("~")
DIR_PROJ = os.path.dirname(os.path.realpath(__file__))
DIR_EVAL_OUT = os.path.join(DIR_PROJ, 'eval_out')

print("Hello")
device = c.GPU0
adaptive_norm = True

def random_password(length=16):
   return ''.join(random.choice(string.printable) for i in range(length))


def generate_key(password,  bs, w, h):
    '''
    Function to generate a secret key with length nbits, based on an input password
    :param password: string password
    :param bs, w, h: batch_size, weight, height
    :return: tensor of 1 and -1 in shape(bs, 1, w, h)
    '''
    salt = 1
    key = PBKDF2(password, salt, int(w * h / 8), count=10, hmac_hash_module=SHA512)
    list_int = list(key)
    array_uint8 = np.array(list_int, dtype=np.uint8)
    array_bits = np.unpackbits(array_uint8).astype(int) * 2 - 1
    array_bits_2d = array_bits.reshape((w, h))
    skey_tensor = torch.tensor(array_bits_2d).repeat(bs, 1, 1, 1)
    return skey_tensor


dwt.to(device)
lpips_loss.to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
mse = MeanSquaredError().to(device)
psnr = PeakSignalNoiseRatio().to(device)


def test_epoch_mm23(embedder, obfuscator, dataloader, swap_target_set=(), cartoon_set=(), typeWR='',
                    dir_image='./images'):

    pro_ssim_list = []
    pro_psnr_list = []
    pro_lpips_list = []

    rec_ssim_list = []
    rec_psnr_list = []
    rec_lpips_list = []

    wrec_ssim_list = []
    wrec_psnr_list = []
    wrec_lpips_list = []

    pro_abs_ssim_list = []
    pro_abs_psnr_list = []
    pro_abs_lpips_list = []

    swap_target_set_len = len(swap_target_set)
    cartoon_set_len = len(cartoon_set)

    time_diff_list = []

    for i_batch, data_batch in tqdm(enumerate(dataloader)):
        i_batch += 1
        # if i_batch > 10:
        #     break

        xa, _ = data_batch

        _bs, _c, _w, _h = xa.shape
        xa = xa.to(device)

        targ_img = None
        obf_name = obfuscator.func.__class__.__name__
        if obf_name in ['FaceShifter', 'SimSwap']:
            targ_img, _ = swap_target_set[i_batch % swap_target_set_len]
            # targ_img, _ = swap_target_set[i_batch]
        elif obf_name == 'Mask':
            targ_img, _ = cartoon_set[(i_batch - 1) % cartoon_set_len]


        t1 = time.time()

        xa_obfs = obfuscator(xa, targ_img)

        # t2 = time.time()

        xa_obfs.detach()

        ## Create password from protection
        # password = random_password()
        password = 0
        skey1 = generate_key(password, _bs, _w, _h).to(device)
        skey1_dwt = dwt(skey1)



        xa_out_z, xa_proc = embedder(xa, xa_obfs, skey1_dwt)

        t3 = time.time()

        time_diff_list.append(t3 - t1)

        img_z = iwt(xa_out_z)

        # Correct recovery
        key_rec = skey1_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
            gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
        xa_rev, xa_rev_2 = embedder(key_rec, xa_proc, skey1_dwt, rev=True)  # Recovery using noisy image

        password = random_password()
        skey2 = generate_key(password, _bs, _w, _h).to(device)
        skey2_dwt = dwt(skey2)
        key_rec = skey2_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
            gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
        xa_rev_wrong, xa_rev_wrong_2 = embedder(key_rec, xa_proc, skey2_dwt, rev=True)


        xa_norm = normalize(xa)

        #### Privacy metrics
        xa_proc_norm, xa_obfs_norm = normalize(xa_proc), normalize(xa_obfs)
        # SSIM
        pro_ssim_score = ssim(xa_proc_norm, xa_obfs_norm).detach().cpu()
        pro_ssim_list.append(pro_ssim_score)
        # PSNR
        pro_psnr = psnr(xa_proc_norm, xa_obfs_norm).detach().cpu()
        pro_psnr_list.append(pro_psnr)
        # LPIPS
        pro_lpips = lpips_loss(xa_proc_norm, xa_obfs_norm).detach().cpu()
        pro_lpips_list.append(pro_lpips)

        #### Abstract Privacy metrics
        # SSIM
        pro_ssim_abs = ssim(xa_proc_norm, xa_norm).detach().cpu()
        pro_abs_ssim_list.append(pro_ssim_abs)
        # PSNR
        pro_psnr_abs = psnr(xa_proc_norm, xa_norm).detach().cpu()
        pro_abs_psnr_list.append(pro_psnr_abs)
        # LPIPS
        pro_lpips_abs = lpips_loss(xa_proc_norm, xa_norm).detach().cpu()
        pro_abs_lpips_list.append(pro_lpips_abs)

        #### Recovery metrics
        # SSIM
        xa_rev_norm = normalize(xa_rev)
        rec_ssim_score = ssim(xa_rev_norm, xa_norm).detach().cpu()
        rec_ssim_list.append(rec_ssim_score)
        # PSNR
        rec_psnr = psnr(xa_rev_norm, xa_norm).detach().cpu()
        rec_psnr_list.append(rec_psnr)
        # LPIPS
        rec_lpips = lpips_loss(xa_rev_norm, xa_norm).detach().cpu()
        rec_lpips_list.append(rec_lpips)

        #### Wrong Recovery metrics
        xa_rev_wrong_norm = normalize(xa_rev_wrong, adaptive=True)

        if typeWR == 'RandWR':
            # SSIM
            wrec_ssim_score = ssim(xa_rev_wrong_norm, xa_norm).detach().cpu()
            wrec_ssim_list.append(wrec_ssim_score)
            # PSNR
            wrec_psnr = psnr(xa_rev_wrong_norm, xa_norm).detach().cpu()
            wrec_psnr_list.append(wrec_psnr)
            # LPIPS
            wrec_lpips = lpips_loss(xa_rev_wrong_norm, xa_norm).detach().cpu()
            wrec_lpips_list.append(wrec_lpips)
        else:
            # SSIM
            wrec_ssim_score = ssim(xa_rev_wrong_norm, xa_obfs_norm).detach().cpu()
            wrec_ssim_list.append(wrec_ssim_score)
            # PSNR
            wrec_psnr = psnr(xa_rev_wrong_norm, xa_obfs_norm).detach().cpu()
            wrec_psnr_list.append(wrec_psnr)
            # LPIPS
            wrec_lpips = lpips_loss(xa_rev_wrong_norm, xa_obfs_norm).detach().cpu()
            wrec_lpips_list.append(wrec_lpips)

        if i_batch <= 30:
            save_image(normalize(xa), f"{dir_image}/batch{i_batch}_orig.jpg", nrow=4)
            save_image(normalize(xa_proc), f"{dir_image}/batch{i_batch}_{obf_name}_proc.jpg", nrow=4)
            save_image(normalize(img_z, adaptive_norm), f"{dir_image}/batch{i_batch}_{obf_name}_proc_byproduct.jpg",
                       nrow=4)
            save_image(normalize(xa_obfs), f"{dir_image}/batch{i_batch}_{obf_name}.jpg", nrow=4)
            save_image(normalize(xa_rev), f"{dir_image}/batch{i_batch}_{obf_name}_rev.jpg", nrow=4)
            save_image(normalize(xa_rev_2, adaptive_norm), f"{dir_image}/batch{i_batch}_{obf_name}_rev_byproduct.jpg", nrow=4)
            save_image(normalize(xa_rev_wrong, adaptive_norm), f"{dir_image}/batch{i_batch}_{obf_name}_rev_wrong.jpg",
                       nrow=4)
            save_image(normalize(xa_rev_wrong_2, adaptive_norm), f"{dir_image}/batch{i_batch}"
                                                              f"_{obf_name}_rev_wrong_byproduct.jpg", nrow=4)

            ########### Recover the image using pre-obfuscated directly ###########
            key_rec = skey1_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
                gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
            xa_revFromObfs, xa_revFromObfs_2 = embedder(key_rec, xa_obfs, skey1_dwt, rev=True)  # Recovery using noisy image
            save_image(normalize(xa_revFromObfs, adaptive_norm),
                       f"{dir_image}/batch{i_batch}_{obf_name}_revFromObfs.jpg", nrow=4)
            save_image(normalize(xa_revFromObfs_2, adaptive_norm),
                       f"{dir_image}/batch{i_batch}_{obf_name}_revFromObfs_byproduct.jpg", nrow=4)

            ########### Recovery image using wrong password with only 1 bit difference ###########
            password = 1
            skey_1bit = generate_key(password, _bs, _w, _h).to(device)
            skey_1bit_dwt = dwt(skey_1bit)
            key_rec = skey_1bit_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
                gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
            xa_rev_wrong1bit, xa_rev_wrong1bit_2 = embedder(key_rec, xa_proc, skey_1bit_dwt, rev=True)
            save_image(normalize(xa_rev_wrong1bit, adaptive_norm),
                       f"{dir_image}/batch{i_batch}_{obf_name}_rev_wrong1bit.jpg", nrow=4)

            ########### Noise Test: Recovery with noised add on image ##########
            for m in range(0, 11):
                key_rec = skey1_dwt.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else \
                    gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
                img_nose = gauss_noise(xa_proc.shape).to(device) * 0.001 * m
                xa_revNoise, xa_revNoise_2 = embedder(key_rec, xa_proc + img_nose, skey1_dwt, rev=True)  # Recovery using
                # noisy image
                save_image(normalize(xa_revNoise, adaptive_norm), f"{dir_image}/batch{i_batch}_{obf_name}_rev_noise{m}.jpg", nrow=4)


    metric_results = {
        'pSSIM': float(np.mean(pro_ssim_list)),
        'pLPIPS': float(np.mean(pro_lpips_list)),
        'pPSNR': float(np.mean(pro_psnr_list)),
        'rSSIM': float(np.mean(rec_ssim_list)),
        'rLPIPS': float(np.mean(rec_lpips_list)),
        'rPSNR': float(np.mean(rec_psnr_list)),
        'wrSSIM': float(np.mean(wrec_ssim_list)),
        'wrLPIPS': float(np.mean(wrec_lpips_list)),
        'wrPSNR': float(np.mean(wrec_psnr_list)),
        'pSSIM_abs': float(np.mean(pro_abs_ssim_list)),
        'pLPIPS_abs': float(np.mean(pro_abs_lpips_list)),
        'pPSNR_abs': float(np.mean(pro_abs_psnr_list)),
    }


    return metric_results


def main(inv_nblocks, embedder_path, dataset_path, test_session_dir, typeWR, batch_size):

    workers = 0 if os.name == 'nt' else 8

    # Determine if an nvidia GPU is available
    print('Running on device: {}'.format(device))

    #### Define the models
    embedder = ModelDWT(n_blocks=inv_nblocks)
    state_dict = torch.load(embedder_path)
    embedder.load_state_dict(state_dict)
    embedder.to(device)

    # Target images used for face swapping
    test_frontal_set = datasets.ImageFolder(c.target_img_dir_test)
    test_frontal_nums = len(test_frontal_set)
    target_set_train_nums = int(test_frontal_nums * 0.9)
    target_set_test_nums = test_frontal_nums - target_set_train_nums
    torch.manual_seed(0)
    target_set_train, target_set_test = \
        torch.utils.data.random_split(test_frontal_set, [target_set_train_nums, target_set_test_nums])

    # Sticker images used for face masking in train and test
    cartoon_set = datasets.ImageFolder(c.cartoon_face_path, loader=rgba_image_loader)
    cartoon_num = len(cartoon_set)
    _train_num = int(cartoon_num * 0.9)
    _test_num = cartoon_num - _train_num
    torch.manual_seed(1)
    cartoon_set_train, cartoon_set_test = torch.utils.data.random_split(cartoon_set, [_train_num, _test_num])

    # Try run validation first
    embedder.eval()

    # Create obfuscator
    # Blur(61, 9, 21), MedianBlur(23), Pixelate(20), FaceShifter(self.device), SimSwap(), Mask()
    obf_options = ['simswap', 'mask', 'blur_61_9_21', 'medianblur_23', 'pixelate_20', 'faceshifter']
    # obf_options = ['simswap']

    ####
    #### Complexity measurement
    ####
    # Create train dataloader
    dir_test = os.path.join(dataset_path)
    dataset_test = datasets.ImageFolder(dir_test, transform=input_trans)
    print(dataset_test.samples[:5])
    loader_test = DataLoader(dataset_test, num_workers=workers, batch_size=batch_size, shuffle=False)

    # Compute complexity
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    tensor = (torch.rand(1, 3, 112, 112).to(device), torch.rand(1, 3, 112, 112).to(device), torch.rand(1, 4, 56, 56).to(device))
    print('Number of parameters: {}'.format(get_parameter_number(embedder)))  # Number of parameters: 1073040
    print("FLOPs", FlopCountAnalysis(embedder, tensor).total())
    print("# params", parameter_count_table(embedder))



    results = {}
    for obf_opt in obf_options:
        obfuscator = Obfuscator(obf_opt, device)
        print('{} Number of parameters: {}'.format(obf_opt, get_parameter_number(obfuscator)))
        obfuscator.eval()
        obfuscator_metrics = test_epoch_mm23(
            embedder, obfuscator, loader_test, target_set_test, cartoon_set_test, typeWR, dir_image=test_session_dir
        )

        print('{}: {}'.format(obf_opt, obfuscator_metrics))
        results[obf_opt] = obfuscator_metrics

    return results


if __name__ == '__main__':
    embedder_configs = [
        [3, os.path.join(DIR_PROJ,
                         c.DIR_INN), 'RandWR'],
    ]

    # Path to original datasets
    datasets1k = (
        ('CelebA', os.path.join(c.DIR_PROJECT, 'experiments/test_data/CelebA')),
        ('LFW', os.path.join(c.DIR_PROJECT, 'experiments/test_data/LFW')),
        ('FFHQ', os.path.join(c.DIR_PROJECT, 'experiments/test_data/FFHQ')),
    )

    for inv_nblocks, embedder_path, typeWR in embedder_configs:
        print(f"******************* {inv_nblocks} inv blocks **********************")
        for dataset_name, dataset_path in datasets1k:
            test_session = f"{dataset_name}_{inv_nblocks}InvBlocks_{typeWR}_256_TripMargin1.2"
            if adaptive_norm:
                test_session += '_adaNorm'
            test_session_dir = os.path.join(DIR_EVAL_OUT, test_session)
            os.makedirs(test_session_dir, exist_ok=True)
            result_file = os.path.join(DIR_EVAL_OUT, f"{test_session}.json")
            result_dict = main(inv_nblocks, embedder_path, dataset_path, test_session_dir, typeWR, batch_size=1)
            with open(result_file, 'w') as f:
                json.dump(result_dict, f)
