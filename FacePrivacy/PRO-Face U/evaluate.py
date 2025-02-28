# LFW evaluation
import argparse
import torch
import random
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import embedder as eb
import os
from torchvision.utils import save_image
from utils.utils_eval import read_pairs, get_paths, evaluate
from face_embedder import PrivFaceEmbedder
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import logging
# from training_triplet_logits import get_batch_negative_index
from utils.utils_train import get_batch_negative_index

dir_home = os.path.expanduser("~")
dir_facenet = os.path.dirname(os.path.realpath(__file__))

from face.face_recognizer import get_recognizer
from utils.loss_functions import triplet_loss, lpips_loss
from torch.nn import TripletMarginWithDistanceLoss
from utils.image_processing import Obfuscator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
lpips_loss.to(device)
perc_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y : lpips_loss(x, y), margin=0.5)


def normalize(x: torch.Tensor):
    x_norm = x.add(1.0).mul(0.5)
    return x_norm


def proc_for_rec(img_batch, zero_mean=False, resize=0, grayscale=False):
    _res = img_batch
    if zero_mean:
        _res = img_batch.sub(0.5).mul(2.0)
    if zero_mean:
        _res = img_batch.sub(0.5).mul(2.0)
    if resize and resize != img_batch.shape[-1]:
        _res = F.resize(_res, size=[resize, resize])
    if grayscale:
        _res = F.rgb_to_grayscale(_res)
        
    return _res
def run_eval(embedder, recognizer, obfuscator, dataloader, path_list, issame_list, target_set, out_dir, model_name):
    file_paths = []
    classes = []
    embeddings_list_orig = []
    embeddings_list_proc = []
    embeddings_list_obfs = []

    triplet_losses = []
    privacy_scores = []

    obf_name = obfuscator.name

    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(dataloader):
            batch_idx += 1
            xb, (paths, yb) = batch
            xb = xb.to(device)
            file_paths.extend(paths)

            if obf_name in ['faceshifter']:
                # Randomly sample a target image and apply face swapping
                num_targ_imgs = len(target_set)
                targ_img_idx = random.randint(0, num_targ_imgs - 1)
                targ_img, _ = target_set[targ_img_idx]
                targ_img_batch = targ_img.repeat(xb.shape[0], 1, 1, 1).to(device)
                xb_obfs = obfuscator.swap(xb, targ_img_batch)
                xb_proc = embedder(xb, xb_obfs)
            else:
                xb_obfs = obfuscator(xb)
                xb_proc = embedder(xb, xb_obfs)
            xb_proc_clamp = torch.clamp(xb_proc, -1, 1)

            # privacy_cost = perc_triplet_loss(xb_targ, xb_proc_clamp, xb).to('cpu')
            dist_protcted = lpips_loss(xb_obfs, xb_proc_clamp).to('cpu')
            dist_original = lpips_loss(xb_obfs, xb).to('cpu')
            privacy_scores.append(dist_protcted / dist_original)

            obfs = obfuscator.name
            if batch_idx % 100 == 0:
                save_image(normalize(xb), f"{out_dir}/Eval_{model_name}_batch{batch_idx}_orig.jpg", nrow=4)
                save_image(normalize(xb_obfs), f"{out_dir}/Eval_{model_name}_batch{batch_idx}_{obfs}.jpg", nrow=4)
                save_image(normalize(xb_proc_clamp), f"{out_dir}/Eval_{model_name}_batch{batch_idx}_proc.jpg", nrow=4)

            orig_embeddings = recognizer(recognizer.resize(xb))
            proc_embeddings = recognizer(recognizer.resize(xb_proc_clamp))
            obfs_embeddings = recognizer(recognizer.resize(xb_obfs))

            # negative_indexes = get_batch_negative_index(yb.tolist())
            # anchor = proc_embeddings
            # positive = orig_embeddings
            # negative = proc_embeddings[negative_indexes]
            # loss_triplet = triplet_loss(anchor, positive, negative)
            # triplet_losses.append(float(loss_triplet.to('cpu')))

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
        # embeddings_list_o2p.append(embeddings_dict_orig[path_a])
        # embeddings_list_o2p.append(embeddings_dict_proc[path_b])
        embeddings_list_p2o_ordered.append(embeddings_dict_proc[path_a])
        embeddings_list_p2o_ordered.append(embeddings_dict_orig[path_b])
        embeddings_list_obfs_ordered.append(embeddings_dict_obfs[path_a])
        embeddings_list_obfs_ordered.append(embeddings_dict_orig[path_b])
    # embeddings_list_o2p = np.array(embeddings_list_o2p)
    embeddings_list_p2o_ordered = np.array(embeddings_list_p2o_ordered)
    embeddings_list_orig_ordered = np.array([embeddings_dict_orig[path] for path in path_list])
    embeddings_list_proc_ordered = np.array([embeddings_dict_proc[path] for path in path_list])
    embeddings_list_obfs_ordered = np.array(embeddings_list_obfs_ordered)

    test_cases = [
        ('Original', embeddings_list_orig_ordered, 'k-'),
        ('Protected', embeddings_list_proc_ordered, 'k--'),
        ('Prot-Orig', embeddings_list_p2o_ordered, 'k-.'),
        ('Obfuscated', embeddings_list_obfs_ordered, 'k:'),
    ]

    plt.clf()

    for case, embedding_list, line_style in test_cases:
        tpr, fpr, roc_auc, eer, accuracy, precision, recall, tars, tar_std, fars, bts = \
            evaluate(embedding_list, issame_list, distance_metric=1)
        acc, thres = np.mean(accuracy), np.mean(bts)
        result_msg = '{}：\n' \
                     '    ACC: {:.4f} | THRES: {:.4f} | AUC: {:.4f} | EER: {:.4f} | TARs: {} | FARs: {} | PS: {:.4f}'. \
            format(case, acc, thres, roc_auc, eer, [round(i, 4) for i in tars], [round(i, 4) for i in fars],
                   np.mean(privacy_scores))
        logging.info(result_msg)
        print(result_msg)
        plt.plot(fpr, tpr, line_style, label='{} | Acc. {:.4f} | Thres. {:.4f}'.format(case, acc, thres))

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(model_name)
    plt.legend()
    plt.grid()
    plt.savefig(f'{out_dir}/roc_{model_name}.pdf', bbox_inches='tight')
    plt.show()

    # print('AVG. Triplet Loss:', np.mean(triplet_losses))
    print('AVG. Privacy Score:', np.mean(privacy_scores))
