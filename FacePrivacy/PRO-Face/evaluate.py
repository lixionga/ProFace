# LFW evaluation
import json
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import os
from torchvision.utils import save_image
from utils.utils_eval import read_pairs, get_paths, evaluate
from embedder import ProFaceEmbedder
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import logging

dir_home = os.path.expanduser("~")
dir_facenet = os.path.dirname(os.path.realpath(__file__))

from face.face_recognizer import get_recognizer
from utils.loss_functions import triplet_loss, lpips_loss
from torch.nn import TripletMarginWithDistanceLoss
from utils.image_processing import Obfuscator, input_trans, normalize
import config.config as c
import sys
sys.path.append(os.path.join(c.PROJECT_DIR, 'SimSwap'))


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
lpips_loss.to(device)
perc_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y : lpips_loss(x, y), margin=0.5)


input_trans = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

input_trans_simswap = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


def adaptive_normalize(x: torch.Tensor):
    _min = x.min()
    _max = x.max()
    x_norm = (x - _min) / (_max - _min)
    return x_norm


def proc_for_rec(img_batch, zero_mean=False, resize=0, grayscale=False):
    _res = img_batch
    if zero_mean:
        _res = img_batch.sub(0.5).mul(2.0)
    if resize and resize != img_batch.shape[-1]:
        _res = F.resize(_res, size=[resize, resize])
    if grayscale:
        _res = F.rgb_to_grayscale(_res)
    return _res


def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)
    return noise


def run_eval(embedder, recognizer, obfuscator, dataloader, path_list, issame_list, target_set, out_dir, model_name):
    file_paths = []
    classes = []
    embeddings_list_orig = []
    embeddings_list_proc = []
    embeddings_list_obfs = []

    obf_name = obfuscator.name

    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(dataloader):
            batch_idx += 1

            xb, (paths, yb) = batch

            _bs, _, _w, _h = xb.shape
            xb = xb.to(device)

            file_paths.extend(paths)

            if obf_name in ['blur', 'pixelate']:
                xb_obfs = obfuscator(xb)
                xb_proc = embedder(xb, xb_obfs)
            else:
                # Sample a target image and apply face swapping
                targ_img, _ = target_set[batch_idx]
                targ_img_batch = targ_img.repeat(xb.shape[0], 1, 1, 1).to(device)
                xb_obfs = obfuscator.swap(xb, targ_img_batch)
                # Attention: Our pretrained models for SimSwap were done in 224x224 image resolution
                if obf_name == 'simswap':
                    xb_obfs = F.resize(xb_obfs, [224, 224], F.InterpolationMode.BICUBIC)
                xb_proc = embedder(xb, xb_obfs)

            # Clamp output image into normal dynamic range
            xb_proc_clamp = torch.clamp(xb_proc, 0, 1) if obf_name == 'simswap' else torch.clamp(xb_proc, -1, 1)
            min_val = 0 if obf_name == 'simswap' else -1

            if batch_idx % 100 == 0:
                save_image(normalize(xb, lower=min_val, upper=1), f"{out_dir}/{model_name}_{batch_idx}_orig.jpg", nrow=4)
                save_image(normalize(xb_obfs, lower=min_val, upper=1), f"{out_dir}/{model_name}_{batch_idx}_{obf_name}.jpg", nrow=4)
                save_image(normalize(xb_proc_clamp, lower=min_val, upper=1), f"{out_dir}/{model_name}_{batch_idx}_proc.jpg", nrow=4)

            orig_embeddings = recognizer(recognizer.resize(xb))
            proc_embeddings = recognizer(recognizer.resize(xb_proc_clamp))
            obfs_embeddings = recognizer(recognizer.resize(xb_obfs))

            orig_embeddings = orig_embeddings.to('cpu').numpy()
            proc_embeddings = proc_embeddings.to('cpu').numpy()
            obfs_embeddings = obfs_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings_list_orig.extend(orig_embeddings)
            embeddings_list_proc.extend(proc_embeddings)
            embeddings_list_obfs.extend(obfs_embeddings)

    embeddings_dict_orig = dict(zip(file_paths, embeddings_list_orig))
    embeddings_dict_proc = dict(zip(file_paths, embeddings_list_proc))
    embeddings_dict_obfs = dict(zip(file_paths, embeddings_list_obfs))

    embeddings_list_p2o_ordered = []
    embeddings_list_obfs_ordered = []
    for path_a, path_b in zip(path_list[0::2], path_list[1::2]):
        embeddings_list_p2o_ordered.append(embeddings_dict_proc[path_a])
        embeddings_list_p2o_ordered.append(embeddings_dict_orig[path_b])
        embeddings_list_obfs_ordered.append(embeddings_dict_obfs[path_a])
        embeddings_list_obfs_ordered.append(embeddings_dict_orig[path_b])
    embeddings_list_p2o_ordered = np.array(embeddings_list_p2o_ordered)
    embeddings_list_orig_ordered = np.array([embeddings_dict_orig[path] for path in path_list])
    embeddings_list_proc_ordered = np.array([embeddings_dict_proc[path] for path in path_list])
    embeddings_list_obfs_ordered = np.array(embeddings_list_obfs_ordered)

    test_cases = [
        ('Orig.', embeddings_list_orig_ordered, 'r-'),
        ('ADR', embeddings_list_proc_ordered, 'g--'),
        ('XDR', embeddings_list_p2o_ordered, 'b-.'),
        ('Obf.', embeddings_list_obfs_ordered, 'k:'),
    ]

    plt.clf()
    plt.figure(figsize=(4, 4))
    for case, embedding_list, line_style in test_cases:
        tpr, fpr, roc_auc, eer, accuracy, precision, recall, tars, tar_std, fars, bts = \
            evaluate(embedding_list, issame_list, distance_metric=1)

        result_dict = dict(
            tpr=list(tpr),
            fpr=list(fpr),
            roc_auc=roc_auc,
            eer=eer,
            acc=list(accuracy),
            precision=list(precision),
            recall=list(recall),
            tars=list(tars),
            tar_std=list(tar_std),
            fars=list(fars),
            bts=list(bts),
        )
        json_file = f'{out_dir}/json_{model_name}_{case}.json'
        with open(json_file, "w") as f:
            json.dump(result_dict, f)

        acc, thres = np.mean(accuracy), np.mean(bts)
        result_msg = '{}：\n' \
                     '    ACC: {:.4f} | THRES: {:.4f} | AUC: {:.4f} | EER: {:.4f} | TARs: {} | FARs: {}'. \
            format(case, acc, thres, roc_auc, eer,
                   '/'.join([str(round(i, 3)) for i in tars]),
                   '/'.join([str(round(i, 3)) for i in fars]))
        logging.info(result_msg)
        print(result_msg)
        plt.plot(fpr, tpr, line_style, label=case)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(model_name)
    # plt.legend(fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(f'{out_dir}/roc_{model_name}.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.show()


def prepare_eval_data(data_dir, data_pairs, transform, dataset_name='lfw', batch_size=8):
    workers = 0 if os.name == 'nt' else 8
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # overwrites class labels in dataset with path so path can be used for saving output
    dataset.samples = [
        (p, (p, idx))
        for p, idx in dataset.samples
    ]
    pairs = read_pairs(data_pairs)
    path_list, issame_list = get_paths(data_dir, pairs, dataset_name)
    test_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    return test_loader, path_list, issame_list


def main(embedder_path, rec_name, data_dir, data_pairs, obfuscator, out_dir, targ_img_path=None, dataset_name='lfw'):
    filename = rec_name + '_' + dataset_name

    # Load pretrained embedder and recognizer model
    embedder = ProFaceEmbedder()
    embedder.to(device)
    embedder_state_dict = torch.load(embedder_path)
    embedder.load_state_dict(embedder_state_dict)
    embedder.eval()

    recognizer = get_recognizer(rec_name)
    recognizer.to(device).eval()

    # Test config:
    trans = input_trans_simswap if obfuscator.name == 'simswap' else input_trans
    test_loader, path_list, issame_list = prepare_eval_data(data_dir, data_pairs, trans, dataset_name)

    if obfuscator.name in ['blur', 'pixelate']:
        target_set = None
    else:
        target_set = datasets.ImageFolder(targ_img_path, transform=obfuscator.targ_img_trans)

    # dataset_target = datasets.ImageFolder(targ_img_path, transform=input_trans)
    run_eval(embedder, recognizer, obfuscator, test_loader, path_list, issame_list, target_set, out_dir, filename)


if __name__ == '__main__':

    if not os.path.exists('eval/'):
        os.makedirs('eval/')
    logging.basicConfig(level=logging.INFO, filename="eval/eval.log")

    # Path to saved protection checkpoint
    embedder_path = 'checkpoints/pixelate_7_IResNet100_ep12_BEST.pth'

    # Name of the pretrained face recognizer, which should fit the protection model,
    # selected from ['MobileFaceNet', 'InceptionResNet', 'IResNet50', 'IResNet100', 'SEResNet50']
    recognizer_name = 'IResNet100'

    # Name of the obfuscator, which should fit the protection model,
    # selected from ['blur_31_2_8', 'pixelate_7', 'faceshifter', 'simswap']
    obfuscator_ops = 'pixelate_7'

    # Test dataset
    test_dataset = {
        'name': 'lfw',
        'dir': os.path.join(dir_home, 'Datasets/LFW/LFW_align_crop_224_test_pairs'),
        'pairs': os.path.join(dir_home, 'Datasets/LFW/pairs.txt')
    }

    # Path to target images (for face swap only)
    targ_img_path = c.target_img_dir_test

    # Dir to save generated files
    output_dir = 'eval'

    # Obfuscator
    obfuscator = Obfuscator(obfuscator_ops)
    obfuscator.eval()

    main(embedder_path, recognizer_name, test_dataset['dir'], test_dataset['pairs'], obfuscator, output_dir,
         targ_img_path, test_dataset['name'])
