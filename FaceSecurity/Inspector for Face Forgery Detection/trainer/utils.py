import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Tracking the path to the definition of the model.


def exp_recons_loss(recons, x):
    x, y = x
    loss = torch.tensor(0., device=y.device)
    real_index = torch.where(1 - y)[0]
    for r in recons:
        if real_index.numel() > 0:
            real_x = torch.index_select(x, dim=0, index=real_index)
            real_rec = torch.index_select(r, dim=0, index=real_index)
            real_rec = F.interpolate(real_rec, size=x.shape[-2:], mode='bilinear', align_corners=True)
            loss += torch.mean(torch.abs(real_rec - real_x))
    return loss


def center_print(content, around='*', repeat_around=10):
    num = repeat_around
    s = around
    print(num * s + ' %s ' % content + num * s)


def reduce_tensor(t):
    rt = t.clone()
    dist.all_reduce(rt)
    rt /= float(dist.get_world_size())
    return rt


def tensor2image(tensor):
    image = tensor.permute([1, 2, 0]).cpu().detach().numpy()
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def state_dict(state_dict):
    """ Remove 'module' keyword in state dictionary. """
    weights = OrderedDict()
    for k, v in state_dict.items():
        weights.update({k.replace("module.", ""): v})
    return weights


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class Timer(object):
    """The class for timer."""

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)



class AccMeter(object):
    def __init__(self):
        self.nums = 0
        self.acc = 0

    def reset(self):
        self.nums = 0
        self.acc = 0

    def update(self, pred, target, use_bce=False):
        if use_bce:
            pred = (pred >= 0.5).int()
        else:
            pred = pred.argmax(1)
        
        self.nums += target.shape[0]
        self.acc += torch.sum(pred == target)

    def mean_acc(self):
        return self.acc / self.nums


class AUCMeter(object):
    def __init__(self):
        self.score = None
        self.true = None

    def reset(self):
        self.score = None
        self.true = None

    def update(self, score, true, use_bce=False):
        if use_bce:
            score = score.detach().cpu().numpy()
        else:
            score = torch.softmax(score.detach(), dim=-1)
            score = torch.select(score, 1, 1).cpu().numpy()
        true = true.flatten().cpu().numpy()
        self.score = score if self.score is None else np.concatenate([self.score, score])
        self.true = true if self.true is None else np.concatenate([self.true, true])

    def mean_auc(self):
        return roc_auc_score(self.true, self.score)

    def curve(self, prefix):
        fpr, tpr, thresholds = roc_curve(self.true, self.score, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        print(f"# EER: {eer:.4f}(thresh: {thresh:.4f})")
        torch.save([fpr, tpr, thresholds], os.path.join(prefix, "roc_curve.pickle"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
