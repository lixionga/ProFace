import torch
import time
import random
import numpy as np
import os
import logging
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from utils.loss_functions import vgg_loss, l1_loss, triplet_loss, lpips_loss
from torch.nn import TripletMarginWithDistanceLoss
from utils.image_processing import normalize
import config.config as c


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Logger(object):
    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i):
        track_str = '{} | {:5d}/{:<5d}| '.format(self.mode, i, self.length)
        loss_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in loss.items())
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        logging.info(track_str + loss_str + '| ' + metric_str)
        print('\r' + track_str + loss_str + '| ' + metric_str, end='')
        if i + 1 == self.length:
            logging.info('')
            print('')


class BatchTimer(object):
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.

    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=False):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred=(), y=()):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


def save_model(embedder, dir_checkpoint, session, epoch, i_batch):
    model_name = f'{session}_ep{epoch}_iter{i_batch}'
    saved_path = f'{dir_checkpoint}/{model_name}.pth'
    torch.save(embedder.state_dict(), saved_path)
    return saved_path


def pass_epoch(embedder, recognizer, obfuscator, dataloader, target_set_train, session='',
               dir_image='./images', dir_checkpoint='./checkpoints', optimizer=None, scheduler=None, show_running=True,
               device='cpu', writer=None, epoch=0, max_batch=np.inf, logger_train=None):
    """Train or evaluate over a data epoch.

    Arguments:
        model {torch.nn.Module} -- Pytorch model.
        loss_fn {callable} -- A function to compute (scalar) loss.
        loader {torch.utils.data.DataLoader} -- A pytorch data loader.

    Keyword Arguments:
        optimizer {torch.optim.Optimizer} -- A pytorch optimizer.
        scheduler {torch.optim.lr_scheduler._LRScheduler} -- LR scheduler (default: {None})
        batch_metrics {dict} -- Dictionary of metric functions to call on each batch. The default
            is a simple timer. A progressive average of these metrics, along with the average
            loss, is printed every batch. (default: {{'time': iter_timer()}})
        show_running {bool} -- Whether or not to print losses and metrics for the current batch
            or rolling averages. (default: {False})
        device {str or torch.device} -- Device for pytorch to use. (default: {'cpu'})
        writer {torch.utils.tensorboard.SummaryWriter} -- Tensorboard SummaryWriter. (default: {None})

    Returns:
        tuple(torch.Tensor, dict) -- A tuple of the average loss and a dictionary of average
            metric values across the epoch.
    """

    mode = 'Train' if embedder.training else 'Valid'
    logger = Logger(mode, length=len(dataloader), calculate_mean=show_running)
    loss_image_vgg_total = 0
    loss_triplet_p2p_total = 0
    loss_triplet_p2o_total = 0
    loss_image_l1_total = 0
    loss_rec_total = 0
    loss_rec_wrong_total = 0
    loss_lf = 0
    loss_lf_total = 0
    loss_batch_total = 0
    metrics = {}
    batch_metrics = {
        'fps': BatchTimer(),
    }

    triplet_loss.to(device)
    lpips_loss.to(device)
    l1_loss.to(device)
    percep_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y: lpips_loss(x, y), margin=1.0)

    models_saved = []
    i_batch = 0
    obf_name = obfuscator.name
    for i_batch, data_batch in enumerate(dataloader):

        if i_batch > max_batch:
            break

        a, n, p = data_batch
        xa, label_a = a
        xn, label_n = n
        xp, label_p = p

        _bs, _, _w, _h = xa.shape

        xa = xa.to(device)
        xn = xn.to(device)
        xp = xp.to(device)

        if obf_name in ['blur', 'pixelate', 'medianblur']:
            ## Perform image processing as target image
            xa_obfs = obfuscator(xa)
            xn_obfs = obfuscator(xn)
            xp_obfs = obfuscator(xp)
        else:
            num_targ_imgs = len(target_set_train)
            targ_img_idx = random.randint(0, num_targ_imgs - 1)
            targ_img, _ = target_set_train[targ_img_idx]
            targ_img_batch = targ_img.repeat(xa.shape[0], 1, 1, 1).to(device)
            xa_obfs = obfuscator.swap(xa, targ_img_batch)
            xn_obfs = obfuscator.swap(xn, targ_img_batch)
            xp_obfs = obfuscator.swap(xp, targ_img_batch)
            targ_img_batch.detach()
            xa_obfs.detach()
            xn_obfs.detach()
            xp_obfs.detach()
            if i_batch % c.SAVE_IMAGE_INTERVAL == 0:
                save_image(obfuscator.targ_img_trans_inv(targ_img),
                           f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_targ.jpg")

        ## Perform image protection
        xa_proc = embedder(xa, xa_obfs)
        xn_proc = embedder(xn, xn_obfs)
        xp_proc = embedder(xp, xp_obfs)

        # Compute face embedding
        embed_orig_a = recognizer(recognizer.resize(xa))
        embed_proc_a = recognizer(recognizer.resize(xa_proc))
        embed_proc_n = recognizer(recognizer.resize(xn_proc))
        embed_proc_p = recognizer(recognizer.resize(xp_proc))

        # Compute two types of triplet losses for identity
        loss_triplet_p2p = triplet_loss(embed_proc_a, embed_proc_p, embed_proc_n)
        loss_triplet_p2o = triplet_loss(embed_proc_a, embed_orig_a, embed_proc_n)

        ## Three kinds of perceptual losses
        # loss_image_vgg = percep_triplet_loss(xa_obfs, xa_proc, xa)
        loss_image_visual = lpips_loss(xa_obfs, xa_proc)
        loss_image_l1 = l1_loss(xa_obfs, xa_proc)

        loss_batch = 2 * loss_image_visual + loss_image_l1 + 0.5 * loss_triplet_p2p + 0.1 * loss_triplet_p2o

        # Save images
        if i_batch % c.SAVE_IMAGE_INTERVAL == 0:
            save_image(normalize(xa),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_orig.jpg", nrow=4)
            save_image(normalize(xa_proc),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_proc.jpg", nrow=4)
            save_image(normalize(xa_obfs),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_{obf_name}.jpg", nrow=4)

        if embedder.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn().detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

        if writer is not None and embedder.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss_image_vgg', {mode: loss_image_visual.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_triplet_p2p', {mode: loss_triplet_p2p.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_triplet_p2o', {mode: loss_triplet_p2o.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_image_l1', {mode: loss_image_l1.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_batch', {mode: loss_batch.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_image_visual = loss_image_visual.detach().cpu()
        loss_image_vgg_total += loss_image_visual
        loss_triplet_p2p = loss_triplet_p2p.detach().cpu()
        loss_triplet_p2p_total += loss_triplet_p2p
        loss_triplet_p2o = loss_triplet_p2o.detach().cpu()
        loss_triplet_p2o_total += loss_triplet_p2o
        loss_image_l1 = loss_image_l1.detach().cpu()
        loss_image_l1_total += loss_image_l1
        loss_batch = loss_batch.detach().cpu()
        loss_batch_total += loss_batch


        if show_running:
            loss_log = {
                'loss_image_vgg': loss_image_vgg_total,
                'loss_image_l1': loss_image_l1_total,
                'loss_triplet_p2p': loss_triplet_p2p_total,
                'loss_triplet_p2o': loss_triplet_p2o_total,
            }
            logger(loss_log, metrics, i_batch)
        else:
            loss_log = {
                'loss_image_vgg': loss_image_visual,
                'loss_image_l1': loss_image_l1,
                'loss_triplet_p2p': loss_triplet_p2p,
                'loss_triplet_p2o': loss_triplet_p2o,
            }
            logger(loss_log, metrics_batch, i_batch)

        # Save model every 5000 iteration
        if (i_batch > 0) and (i_batch % c.SAVE_MODEL_INTERVAL == 0) and (mode == 'Train'):
            saved_path = save_model(embedder, dir_checkpoint, session, epoch, i_batch)
            models_saved.append(saved_path)

    print('\n')
    if embedder.training and scheduler is not None:
        scheduler.step()

    loss_image_vgg_total = loss_image_vgg_total / (i_batch + 1)
    loss_triplet_p2p_total = loss_triplet_p2p_total / (i_batch + 1)
    loss_triplet_p2o_total = loss_triplet_p2o_total / (i_batch + 1)
    loss_image_l1_total = loss_image_l1_total / (i_batch + 1)
    loss_batch_total = loss_batch_total / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

    if writer is not None and not embedder.training:
        writer.add_scalars('loss_image_vgg', {mode: loss_image_vgg_total.detach()}, writer.iteration)
        writer.add_scalars('loss_triplet_p2p_total', {mode: loss_triplet_p2p_total.detach()}, writer.iteration)
        writer.add_scalars('loss_triplet_p2o_total', {mode: loss_triplet_p2o_total.detach()}, writer.iteration)
        writer.add_scalars('loss_image_l1', {mode: loss_image_l1_total.detach()}, writer.iteration)
        writer.add_scalars('loss_batch', {mode: loss_batch_total.detach()}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return loss_batch_total, metrics, models_saved
