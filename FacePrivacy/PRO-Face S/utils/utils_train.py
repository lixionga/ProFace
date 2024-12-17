import torch
import time
import random
import numpy as np
import os
import logging
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from utils.loss_functions import vgg_loss, l1_loss, triplet_loss, lpips_loss, logits_loss, percep_triplet_loss
from torch.nn import TripletMarginWithDistanceLoss
import modules.Unet_common as common
from utils.image_processing import normalize, clamp_normalize
import config.config as c
from PIL import Image
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = c.DEVICE

dwt = common.DWT()
iwt = common.IWT()
# percep_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y: lpips_loss(x, y), margin=1.0)
triplet_loss.to(device)


class Logger(object):
    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / i
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i_batch):
        track_str = '{} | {:5d}/{:<5d}| '.format(self.mode, i_batch, self.length)
        loss_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i_batch)) for k, v in loss.items())
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i_batch)) for k, v in metrics.items())
        logging.info(track_str + loss_str + '| ' + metric_str)
        print('\r' + track_str + loss_str + '| ' + metric_str, end='')
        if i_batch == self.length:
            logging.info('\n')
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


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


def get_random_sample(lst, execlude):
    '''
    Randomly select a sample (from an input list) that is different from execlude
    :param lst: input list
    :param execlude: the value to execlude
    :return: the selected number if there exist, otherwise none
    '''
    for i in range(len(lst)):
        element = random.sample(lst, 1)[0]
        if element != execlude:
            return element
    return None


def get_batch_negative_index(label_list):
    negative_indexes = []
    for i, label in enumerate(label_list):
        other_elements = list(np.delete(label_list, i))
        neg_label = get_random_sample(other_elements, label)
        neg_index = label_list.index(neg_label) if neg_label is not None else i
        negative_indexes.append(neg_index)
    return negative_indexes


def get_batch_triplet_index(batch_labels):
    import itertools
    batch_size = batch_labels.shape[0]
    all_pairs = itertools.permutations(range(batch_size), 2)
    pos_idx = []
    neg_idx = []
    for i, j in all_pairs:
        if not batch_labels.tolist()[i] == batch_labels.tolist()[j]:
            pos_idx.append(i)
            neg_idx.append(j)
    return pos_idx, neg_idx



def save_model(embedder, optimizer, dir_checkpoint, session, epoch, i_batch):
    model_name = f'{session}_ep{epoch}_iter{i_batch}'
    saved_path = f'{dir_checkpoint}/{model_name}.pth'
    # torch.save({'opt': optimizer.state_dict(), 'net': embedder.state_dict()}, saved_path)
    torch.save(embedder.state_dict(), saved_path)
    return saved_path



def gauss_noise(shape):
    # noise = torch.zeros(shape).cuda()
    noise = torch.zeros(shape)
    for i in range(noise.shape[0]):
        # noise[i] = torch.randn(noise[i].shape).cuda()
        noise[i] = torch.randn(noise[i].shape)

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)



# def unsigned_long_to_binary_repr(unsigned_long, passwd_length):
#     batch_size = unsigned_long.shape[0]
#     target_size = passwd_length // 4
#
#     binary = np.empty((batch_size, passwd_length), dtype=np.float32)
#     for idx in range(batch_size):
#         binary[idx, :] = np.array([int(item) for item in bin(unsigned_long[idx])[2:].zfill(passwd_length)])
#
#     dis_target = np.empty((batch_size, target_size), dtype=np.long)
#     for idx in range(batch_size):
#         tmp = unsigned_long[idx]
#         for byte_idx in range(target_size):
#             dis_target[idx, target_size - 1 - byte_idx] = tmp % 16
#             tmp //= 16
#     return binary, dis_target


def pass_epoch(embedder, recognizer, obfuscator, classifier, dataloader, dataloader_nonface, swap_target_set=(),
               cartoon_set=(), session='', dir_image='./images', dir_checkpoint='./checkpoints', optimizer=None,
               scheduler=None, show_running=True, writer=None, epoch=0, debug=False):
    """Train or evaluate over a data epoch.

    Arguments:
        face_detection {torch.nn.Module} -- Pytorch face_detection.
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
    debug_max_batches = 10
    mode = 'Train' if embedder.training else 'Valid'
    logger = Logger(mode, length=debug_max_batches if debug else len(dataloader), calculate_mean=show_running)
    loss_img_perc_total = 0
    loss_triplet_p2p_total = 0
    loss_triplet_p2o_total = 0
    loss_img_l1_total = 0
    loss_rec_total = 0
    loss_rec_wrong_total = 0
    loss_utility_total = 0
    loss_batch_total = 0
    metrics = {}
    metric_functions = {'FPS': BatchTimer()}
    if c.utility_level == c.Utility.FACE or c.utility_level == c.Utility.GENDER:
        metric_functions['Acc'] = accuracy
    # num_targ_imgs = len(swap_target_set)
    # num_cartoon_imgs = len(cartoon_set)
    target_set_dict = {'FaceShifter': swap_target_set, 'SimSwap': swap_target_set, 'Mask': cartoon_set}

    triplet_loss.to(device)
    lpips_loss.to(device)
    l1_loss.to(device)
    # percep_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y: lpips_loss(x, y), margin=1.0)

    models_saved = []
    i_batch = 1
    obf_name = obfuscator.name

    loader_nonface_iter = None
    if c.utility_level == c.Utility.FACE and dataloader_nonface:
        loader_nonface_iter = iter(dataloader_nonface)

    for i_batch, data_batch in enumerate(dataloader):
        i_batch += 1
        # Only run 2 batches in debug mode
        if debug and i_batch > debug_max_batches:
            break

        a, n, p = data_batch
        xa, name_a, gender_a = a
        xn, name_n, gender_n = n
        xp, name_p, gender_p = p

        _bs, _, _w, _h = xa.shape
        xa = xa.to(device)
        xn = xn.to(device)
        xp = xp.to(device)
        gender_a = gender_a.to(device)
        gender_n = gender_n.to(device)
        gender_p = gender_p.to(device)

        # Obtain the target image for swapping obfuscations
        targ_img = None
        obf_func = obfuscator.func
        # if obf_name in ['faceshifter', 'simswap']:
        #     targ_img_idx = random.randint(0, num_targ_imgs - 1)
        #     targ_img, _ = target_set_train[targ_img_idx]
        #
        #     if i_batch % c.SAVE_IMAGE_INTERVAL == 0: # Save target image
        #         targ_img.save(f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_targ.jpg")

        if obf_name in ['hybrid', 'hybridMorph', 'hybridAll']:
            obfuscator.func = random.choice(obfuscator.functions)

        obf_type = obfuscator.func.__class__.__name__
        if obf_type in ['FaceShifter', 'SimSwap', 'Mask']:
            target_set = target_set_dict[obf_type]
            num_targ_imgs = len(target_set)
            targ_img_idx = random.randint(0, num_targ_imgs - 1)
            targ_img, _ = target_set[targ_img_idx]
            targ_extension = 'png' if obf_type == 'Mask' else 'jpg'
            if (i_batch % c.SAVE_IMAGE_INTERVAL == 0) or (i_batch == 1):  # Save target image
                targ_img.save(f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_targ.{targ_extension}")

        xa_obfs = obfuscator(xa, targ_img)
        xn_obfs = obfuscator(xn, targ_img)
        xp_obfs = obfuscator(xp, targ_img)
        xa_obfs.detach()
        xn_obfs.detach()
        xp_obfs.detach()

################################################################################################################
        ## Create password from protection
        password_a = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))
        password_n = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))
        password_p = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))

        # password = torch.randint(0, 2, (_bs, 32, 1, 1)).mul(4).sub(2).repeat(1, 1, _w // 2, _h // 2).to(device)
        xa_out_z, xa_proc = embedder(xa, xa_obfs, password_a)
        xn_out_z, xn_proc = embedder(xn, xn_obfs, password_n)
        xp_out_z, xp_proc = embedder(xp, xp_obfs, password_p)

        xa_rev, xa_obfs_rev = embedder(password_a.repeat(1, 4, 1, 1), xa_proc, password_a, rev=True)
        xn_rev, xn_obfs_rev = embedder(password_n.repeat(1, 4, 1, 1), xn_proc, password_n, rev=True)
        xp_rev, xp_obfs_rev = embedder(password_p.repeat(1, 4, 1, 1), xp_proc, password_p, rev=True)

        # Compute face embedding
        embed_orig_a = recognizer(recognizer.resize(xa))
        embed_proc_a = recognizer(recognizer.resize(xa_proc))
        embed_proc_n = recognizer(recognizer.resize(xn_proc))
        embed_proc_p = recognizer(recognizer.resize(xp_proc))

        loss_utility = 0
        attr_pred, attr_label = (), ()
        loss_triplet_p2p, loss_triplet_p2o = 0, 0
        if c.utility_level == c.Utility.FACE and dataloader_nonface:
            try:
                batch_nonface, (_, label_nonface) = next(loader_nonface_iter)
            except StopIteration:
                loader_nonface_iter = iter(dataloader_nonface)
                batch_nonface, (_, label_nonface) = next(loader_nonface_iter)
            _embed_nonface = recognizer(recognizer.resize(batch_nonface.to(device)))
            _embed_face = torch.cat((embed_proc_a, embed_proc_n, embed_proc_p), dim=0)
            _embeds_all = torch.cat((_embed_face, _embed_nonface), dim=0)
            attr_pred = classifier(_embeds_all)
            label_face = torch.zeros(_bs * 3)
            attr_label = torch.cat((label_face.to(device), label_nonface.to(device)), dim=0).long()
            loss_utility = logits_loss(attr_pred, attr_label)
        elif c.utility_level == c.Utility.GENDER:
            embed_proc = torch.cat((embed_proc_a, embed_proc_n, embed_proc_p), dim=0)
            attr_pred = classifier(embed_proc)
            attr_label = torch.cat((gender_a, gender_n, gender_p), dim=0)
            loss_utility = logits_loss(attr_pred, attr_label)
        elif c.utility_level == c.Utility.IDENTITY:
            loss_triplet_p2p = triplet_loss(embed_proc_a, embed_proc_p, embed_proc_n)
            loss_triplet_p2o = triplet_loss(embed_proc_a, embed_orig_a, embed_proc_n)
            _id_w1, _id_w2 = c.identity_weights[recognizer.name][obf_name]
            loss_utility = _id_w1 * loss_triplet_p2p + _id_w2 * loss_triplet_p2o


        ## Three kinds of perceptual losses
        loss_img_perc = lpips_loss(xa_obfs, xa_proc) \
                        + lpips_loss(xn_obfs, xn_proc) \
                        + lpips_loss(xp_obfs, xp_proc)
        loss_img_l1 = l1_loss(xa_obfs, xa_proc) \
                      + l1_loss(xn_obfs, xn_proc) \
                      + l1_loss(xp_obfs, xp_proc)
        loss_image = 5 * loss_img_perc + loss_img_l1
        # loss_id = 0.5 * loss_triplet_p2p + 0.1 * loss_triplet_p2o
        # loss_id =  2.5 * loss_triplet_p2p + 0.5 * loss_triplet_p2o # AdaFace
        # loss_id =  5 * loss_triplet_p2p + loss_triplet_p2o # AdaFace

        ## Make correctly recovered image closer to original while further to wrong recovered image
        password_wrong = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))
        xa_rev_wrong, _ = embedder(password_wrong.repeat(1, 4, 1, 1), xa_proc, password_wrong, rev=True)
        xn_rev_wrong, _ = embedder(password_wrong.repeat(1, 4, 1, 1), xn_proc, password_wrong, rev=True)
        xp_rev_wrong, _ = embedder(password_wrong.repeat(1, 4, 1, 1), xp_proc, password_wrong, rev=True)

        # # recovery loss v1
        # loss_rec = l1_loss(xa_rev, xa) \
        #            + l1_loss(xn_rev, xn) \
        #            + l1_loss(xp_rev, xp)
        #
        # loss_rec_wrong = percep_triplet_loss(xa, xa_rev, xa_rev_wrong) \
        #                  + percep_triplet_loss(xn, xn_rev, xp_rev_wrong) \
        #                  + percep_triplet_loss(xp, xp_rev, xp_rev_wrong)

        # recovery loss v2
        loss_rec = l1_loss(xa_rev, xa) \
                   + l1_loss(xn_rev, xn) \
                   + l1_loss(xp_rev, xp) \
                   + l1_loss(xa_rev_wrong, xa_obfs) \
                   + l1_loss(xn_rev_wrong, xn_obfs) \
                   + l1_loss(xp_rev_wrong, xp_obfs)
        loss_rec_wrong = percep_triplet_loss(xa_rev, xa, xa_obfs) \
                         + percep_triplet_loss(xn_rev, xn, xn_obfs) \
                         + percep_triplet_loss(xp_rev, xp, xp_obfs) \
                         + percep_triplet_loss(xa_rev_wrong, xa_obfs, xa) \
                         + percep_triplet_loss(xn_rev_wrong, xn_obfs, xn) \
                         + percep_triplet_loss(xp_rev_wrong, xp_obfs, xp)

        # Gender utility preserved
        loss_batch = loss_image + loss_rec + loss_rec_wrong
        if c.utility_level > c.Utility.NONE:
            loss_batch += (loss_utility * c.utility_weights[c.utility_level])
        # ID utility preserved
        # loss_batch = loss_image + loss_id + loss_rec + loss_rec_wronghtop


        # Save images
        if (i_batch % c.SAVE_IMAGE_INTERVAL == 0) or (i_batch == 1):
            save_image(normalize(xa),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_orig.jpg", nrow=4)
            save_image(normalize(xa_proc),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_proc.jpg", nrow=4)
            save_image(normalize(xa_obfs),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_{obf_type}.jpg", nrow=4)
            save_image(normalize(xa_rev),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_rev.jpg", nrow=4)
            save_image(normalize(xa_obfs_rev),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_obfs_rev.jpg", nrow=4)
            save_image(normalize(xa_rev_wrong, adaptive=True),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_rev_wrong.jpg", nrow=4)

        if embedder.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        # loss_history.append([loss_batch.item(), 0.])
        # image_loss_history.append([loss_image.item(), 0.])
        # rec_loss_history.append([loss_rec.item(), 0.])
        # id_loss_history.append([loss_id.item(), 0.])

        metrics_batch = {}
        for metric_name, metric_fn in metric_functions.items():
            metrics_batch[metric_name] = metric_fn(attr_pred, attr_label).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

        if writer is not None and embedder.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss_img_perc', {mode: loss_img_perc.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_img_l1', {mode: loss_img_l1.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_rec', {mode: loss_rec.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_rec_wrong', {mode: loss_rec_wrong.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_batch', {mode: loss_batch.detach().cpu()}, writer.iteration)
                if c.utility_level > c.Utility.NONE:
                    writer.add_scalars('loss_utility', {mode: loss_utility.detach().cpu()}, writer.iteration)
                if c.utility_level == c.Utility.IDENTITY:
                    writer.add_scalars('loss_triplet_p2p', {mode: loss_triplet_p2p.detach().cpu()}, writer.iteration)
                    writer.add_scalars('loss_triplet_p2o', {mode: loss_triplet_p2o.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_img_perc = loss_img_perc.detach().cpu()
        loss_img_perc_total += loss_img_perc
        loss_img_l1 = loss_img_l1.detach().cpu()
        loss_img_l1_total += loss_img_l1
        loss_rec = loss_rec.detach().cpu()
        loss_rec_total += loss_rec
        loss_rec_wrong = loss_rec_wrong.detach().cpu()
        loss_rec_wrong_total += loss_rec_wrong
        loss_batch = loss_batch.detach().cpu()
        loss_batch_total += loss_batch
        if c.utility_level > c.Utility.NONE:
            loss_utility = loss_utility.detach().cpu()
            loss_utility_total += loss_utility
        if c.utility_level == c.Utility.IDENTITY:
            loss_triplet_p2p = loss_triplet_p2p.detach().cpu()
            loss_triplet_p2p_total += loss_triplet_p2p
            loss_triplet_p2o = loss_triplet_p2o.detach().cpu()
            loss_triplet_p2o_total += loss_triplet_p2o
        if show_running:
            loss_log = {
                'L_visual': loss_img_perc_total,
                'L_l1': loss_img_l1_total,
                'L_p2p': loss_triplet_p2p_total,
                'L_p2o': loss_triplet_p2o_total,
                'L_rec': loss_rec_total,
                'L_recx': loss_rec_wrong_total,
                'L_utility': loss_utility_total
            }
            logger(loss_log, metrics, i_batch)
        else:
            loss_log = {
                'L_visual': loss_img_perc,
                'L_l1': loss_img_l1,
                'L_p2p': loss_triplet_p2p,
                'L_p2o': loss_triplet_p2o,
                'L_rec': loss_rec,
                'L_recx': loss_rec_wrong,
                'L_utility': loss_utility
            }
            logger(loss_log, metrics_batch, i_batch)

        # Save face_detection every 5000 iteration
        if (i_batch % c.SAVE_MODEL_INTERVAL == 0) and (mode == 'Train'):
            saved_path = save_model(embedder, optimizer, dir_checkpoint, session, epoch, i_batch)
            models_saved.append(saved_path)


    # print('\n')
    if embedder.training and scheduler is not None:
        scheduler.step()

    loss_img_perc_total = loss_img_perc_total / i_batch
    loss_img_l1_total = loss_img_l1_total / i_batch
    loss_triplet_p2p_total = loss_triplet_p2p_total / i_batch
    loss_triplet_p2o_total = loss_triplet_p2o_total / i_batch
    loss_rec_total = loss_rec_total / i_batch
    loss_rec_wrong_total = loss_rec_wrong_total / i_batch
    loss_utility_total = loss_utility_total / i_batch
    loss_batch_total = loss_batch_total / i_batch
    metrics = {k: v / i_batch for k, v in metrics.items()}

    if writer is not None and not embedder.training:
        writer.add_scalars('loss_img_perc', {mode: loss_img_perc_total.detach()}, writer.iteration)
        writer.add_scalars('loss_img_l1', {mode: loss_img_l1_total.detach()}, writer.iteration)
        writer.add_scalars('loss_rec_total', {mode: loss_rec_total.detach()}, writer.iteration)
        writer.add_scalars('loss_rec_wrong_total', {mode: loss_rec_wrong_total.detach()}, writer.iteration)
        writer.add_scalars('loss_batch', {mode: loss_batch_total.detach()}, writer.iteration)
        if c.utility_level > c.Utility.NONE:
            writer.add_scalars('loss_utility', {mode: loss_utility_total.detach()}, writer.iteration)
        if c.utility_level == c.Utility.IDENTITY:
            writer.add_scalars('loss_triplet_p2p_total', {mode: loss_triplet_p2p_total.detach()}, writer.iteration)
            writer.add_scalars('loss_triplet_p2o_total', {mode: loss_triplet_p2o_total.detach()}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return loss_batch_total, metrics, models_saved


def pass_epoch_mm23(embedder, obfuscator, dataloader, swap_target_set=(),
               cartoon_set=(), session='', dir_image='./images', dir_checkpoint='./checkpoints', optimizer=None,
               scheduler=None, show_running=True, writer=None, epoch=0, debug=False):
    """Train or evaluate over a data epoch.

    Arguments:
        face_detection {torch.nn.Module} -- Pytorch face_detection.
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
    debug_max_batches = 10
    mode = 'Train' if embedder.training else 'Valid'
    logger = Logger(mode, length=debug_max_batches if debug else len(dataloader), calculate_mean=show_running)
    loss_img_perc_total = 0
    loss_triplet_p2p_total = 0
    loss_triplet_p2o_total = 0
    loss_img_l1_total = 0
    loss_rec_total = 0
    loss_rec_wrong_total = 0
    loss_utility_total = 0
    loss_batch_total = 0
    metrics = {}
    metric_functions = {'FPS': BatchTimer()}
    if c.utility_level == c.Utility.FACE or c.utility_level == c.Utility.GENDER:
        metric_functions['Acc'] = accuracy
    # num_targ_imgs = len(swap_target_set)
    # num_cartoon_imgs = len(cartoon_set)
    target_set_dict = {'FaceShifter': swap_target_set, 'SimSwap': swap_target_set, 'Mask': cartoon_set}

    triplet_loss.to(device)
    lpips_loss.to(device)
    l1_loss.to(device)

    models_saved = []
    i_batch = 1
    obf_name = obfuscator.name

    # loader_nonface_iter = None
    # if c.utility_level == c.Utility.FACE and dataloader_nonface:
    #     loader_nonface_iter = iter(dataloader_nonface)

    for i_batch, data_batch in enumerate(dataloader):
        i_batch += 1
        # Only run 2 batches in debug mode
        if debug and i_batch > debug_max_batches:
            break

        # xa, _ = data_batch
        xa = data_batch

        _bs, _c, _w, _h = xa.shape
        xa = xa.to(device)

        # Obtain the target image for swapping obfuscations
        targ_img = None

        if obf_name in ['hybrid', 'hybridMorph', 'hybridAll']:
            obfuscator.func = random.choice(obfuscator.functions)

        obf_type = obfuscator.func.__class__.__name__
        if obf_type in ['FaceShifter', 'SimSwap', 'Mask']:
            target_set = target_set_dict[obf_type]
            num_targ_imgs = len(target_set)
            targ_img_idx = random.randint(0, num_targ_imgs - 1)
            # targ_img, _ = target_set[targ_img_idx]
            targ_img, _ = target_set[targ_img_idx]
            targ_extension = 'png' if obf_type == 'Mask' else 'jpg'
            if (i_batch % c.SAVE_IMAGE_INTERVAL == 0) or (i_batch == 1):  # Save target image
                targ_img.save(f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_targ.{targ_extension}")

        xa_obfs = obfuscator(xa, targ_img)
        xa_obfs.detach()

################################################################################################################
        ## Create password from protection
        password_a = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))

        xa_out_z, xa_proc = embedder(xa, xa_obfs, password_a)

        # xa_rev, xa_obfs_rev = embedder(password_a.repeat(1, 4, 1, 1), xa_proc, password_a, rev=True)

        # Feed random noise as input in recovery process
        noise = password_a.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
        xa_rev, xa_obfs_rev = embedder(noise, xa_proc, password_a, rev=True)

        loss_utility = 0
        attr_pred, attr_label = (), ()
        loss_triplet_p2p, loss_triplet_p2o = 0, 0

        ## Three kinds of perceptual losses
        loss_img_perc = lpips_loss(xa_obfs, xa_proc)
        loss_img_l1 = l1_loss(xa_obfs, xa_proc)
        loss_image = 5 * loss_img_perc + loss_img_l1

        ## Make correctly recovered image closer to original while further to wrong recovered image
        password_wrong = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))
        noise2 = password_wrong.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
        xa_rev_wrong, _ = embedder(noise2, xa_proc, password_wrong, rev=True)

        if c.WRONG_RECOVER_TYPE == 'Random':
            loss_rec = l1_loss(xa_rev, xa)
            loss_rec_wrong = percep_triplet_loss(xa, xa_rev, xa_rev_wrong) + triplet_loss(xa, xa_rev, xa_rev_wrong)
            loss_batch = 0.3 * loss_image + 0.5 * loss_rec + 0.2 * loss_rec_wrong
        else:
            loss_rec = l1_loss(xa_rev, xa) + l1_loss(xa_rev_wrong, xa_obfs)
            loss_rec_wrong = percep_triplet_loss(xa_rev, xa, xa_obfs) + percep_triplet_loss(xa_rev_wrong, xa_obfs, xa)
            loss_batch = 0.2 * loss_image + 0.3 * loss_rec + 0.5 * loss_rec_wrong

        if c.utility_level > c.Utility.NONE:
            loss_batch += (loss_utility * c.utility_weights[c.utility_level])

        # Save images
        if (i_batch % c.SAVE_IMAGE_INTERVAL == 0) or (i_batch == 1):
            save_image(normalize(xa),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_orig.jpg", nrow=4)
            save_image(normalize(xa_proc),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_proc.jpg", nrow=4)
            save_image(normalize(xa_obfs),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_{obf_type}.jpg", nrow=4)
            save_image(normalize(xa_rev),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_rev.jpg", nrow=4)
            save_image(normalize(xa_obfs_rev),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_obfs_rev.jpg", nrow=4)
            save_image(clamp_normalize(xa_rev_wrong, lmin=-1, lmax=1),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_rev_wrong.jpg", nrow=4)

        if embedder.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        metrics_batch = {}
        for metric_name, metric_fn in metric_functions.items():
            metrics_batch[metric_name] = metric_fn(attr_pred, attr_label).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
        # metrics['privBudget'] = metrics.get('privBudget', .0) + privacy_budget

        if writer is not None and embedder.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss_img_perc', {mode: loss_img_perc.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_img_l1', {mode: loss_img_l1.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_rec', {mode: loss_rec.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_rec_wrong', {mode: loss_rec_wrong.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_batch', {mode: loss_batch.detach().cpu()}, writer.iteration)
                if c.utility_level > c.Utility.NONE:
                    writer.add_scalars('loss_utility', {mode: loss_utility.detach().cpu()}, writer.iteration)
                if c.utility_level == c.Utility.IDENTITY:
                    writer.add_scalars('loss_triplet_p2p', {mode: loss_triplet_p2p.detach().cpu()}, writer.iteration)
                    writer.add_scalars('loss_triplet_p2o', {mode: loss_triplet_p2o.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_img_perc = loss_img_perc.detach().cpu()
        loss_img_perc_total += loss_img_perc
        loss_img_l1 = loss_img_l1.detach().cpu()
        loss_img_l1_total += loss_img_l1
        loss_rec = loss_rec.detach().cpu()
        loss_rec_total += loss_rec
        loss_rec_wrong = loss_rec_wrong.detach().cpu()
        loss_rec_wrong_total += loss_rec_wrong
        loss_batch = loss_batch.detach().cpu()
        loss_batch_total += loss_batch
        if c.utility_level > c.Utility.NONE:
            loss_utility = loss_utility.detach().cpu()
            loss_utility_total += loss_utility
        if c.utility_level == c.Utility.IDENTITY:
            loss_triplet_p2p = loss_triplet_p2p.detach().cpu()
            loss_triplet_p2p_total += loss_triplet_p2p
            loss_triplet_p2o = loss_triplet_p2o.detach().cpu()
            loss_triplet_p2o_total += loss_triplet_p2o
        if show_running:
            loss_log = {
                'L_visual': loss_img_perc_total,
                'L_l1': loss_img_l1_total,
                'L_rec': loss_rec_total,
                'L_recx': loss_rec_wrong_total,
                'L_total': loss_batch_total,
            }
            logger(loss_log, metrics, i_batch)
        else:
            loss_log = {
                'L_visual': loss_img_perc,
                'L_l1': loss_img_l1,
                'L_rec': loss_rec,
                'L_recx': loss_rec_wrong,
                'L_total': loss_batch,
            }
            logger(loss_log, metrics_batch, i_batch)

        # Save face_detection every 5000 iteration
        if (i_batch % c.SAVE_MODEL_INTERVAL == 0) and (mode == 'Train'):
            saved_path = save_model(embedder, optimizer, dir_checkpoint, session, epoch, i_batch)
            models_saved.append(saved_path)


    # print('\n')
    if embedder.training and scheduler is not None:
        scheduler.step()

    loss_img_perc_total = loss_img_perc_total / i_batch
    loss_img_l1_total = loss_img_l1_total / i_batch
    loss_triplet_p2p_total = loss_triplet_p2p_total / i_batch
    loss_triplet_p2o_total = loss_triplet_p2o_total / i_batch
    loss_rec_total = loss_rec_total / i_batch
    loss_rec_wrong_total = loss_rec_wrong_total / i_batch
    loss_utility_total = loss_utility_total / i_batch
    loss_batch_total = loss_batch_total / i_batch
    metrics = {k: v / i_batch for k, v in metrics.items()}

    if writer is not None and not embedder.training:
        writer.add_scalars('loss_img_perc', {mode: loss_img_perc_total.detach()}, writer.iteration)
        writer.add_scalars('loss_img_l1', {mode: loss_img_l1_total.detach()}, writer.iteration)
        writer.add_scalars('loss_rec_total', {mode: loss_rec_total.detach()}, writer.iteration)
        writer.add_scalars('loss_rec_wrong_total', {mode: loss_rec_wrong_total.detach()}, writer.iteration)
        writer.add_scalars('loss_batch', {mode: loss_batch_total.detach()}, writer.iteration)
        if c.utility_level > c.Utility.NONE:
            writer.add_scalars('loss_utility', {mode: loss_utility_total.detach()}, writer.iteration)
        if c.utility_level == c.Utility.IDENTITY:
            writer.add_scalars('loss_triplet_p2p_total', {mode: loss_triplet_p2p_total.detach()}, writer.iteration)
            writer.add_scalars('loss_triplet_p2o_total', {mode: loss_triplet_p2o_total.detach()}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return loss_batch_total, metrics, models_saved


def pass_epoch_utility(embedder, utility_fc, obfuscator, recognizer, dataloader, swap_target_set=(),
               cartoon_set=(), session='', dir_image='./images', dir_checkpoint='./checkpoints', optimizer=None,
               scheduler=None, show_running=True, writer=None, epoch=0, debug=False):
    """Train or evaluate over a data epoch.

    Arguments:
        face_detection {torch.nn.Module} -- Pytorch face_detection.
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
    debug_max_batches = 10
    mode = 'Train' if embedder.training else 'Valid'
    logger = Logger(mode, length=debug_max_batches if debug else len(dataloader), calculate_mean=show_running)
    loss_img_perc_total = 0
    loss_triplet_p2p_total = 0
    loss_triplet_p2o_total = 0
    loss_img_l1_total = 0
    loss_rec_total = 0
    loss_rec_wrong_total = 0
    loss_utility_total = 0
    loss_batch_total = 0
    metrics = {}
    metric_functions = {'FPS': BatchTimer()}
    if c.utility_level == c.Utility.FACE or c.utility_level == c.Utility.GENDER:
        metric_functions['Acc'] = accuracy
    # num_targ_imgs = len(swap_target_set)
    # num_cartoon_imgs = len(cartoon_set)
    target_set_dict = {'FaceShifter': swap_target_set, 'SimSwap': swap_target_set, 'Mask': cartoon_set}

    triplet_loss.to(device)
    lpips_loss.to(device)
    l1_loss.to(device)

    models_saved = []
    i_batch = 1
    obf_name = obfuscator.name

    # loader_nonface_iter = None
    # if c.utility_level == c.Utility.FACE and dataloader_nonface:
    #     loader_nonface_iter = iter(dataloader_nonface)

    for i_batch, data_batch in enumerate(dataloader):
        i_batch += 1
        # Only run 2 batches in debug mode
        if debug and i_batch > debug_max_batches:
            break

        xa, _ = data_batch

        _bs, _c, _w, _h = xa.shape
        xa = xa.to(device)

        # Obtain the target image for swapping obfuscations
        targ_img = None

        if obf_name in ['hybrid', 'hybridMorph', 'hybridAll']:
            obfuscator.func = random.choice(obfuscator.functions)

        obf_type = obfuscator.func.__class__.__name__
        if obf_type in ['FaceShifter', 'SimSwap', 'Mask']:
            target_set = target_set_dict[obf_type]
            num_targ_imgs = len(target_set)
            targ_img_idx = random.randint(0, num_targ_imgs - 1)
            targ_img, _ = target_set[targ_img_idx]
            targ_extension = 'png' if obf_type == 'Mask' else 'jpg'
            if (i_batch % c.SAVE_IMAGE_INTERVAL == 0) or (i_batch == 1):  # Save target image
                targ_img.save(f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_targ.{targ_extension}")

        xa_obfs = obfuscator(xa, targ_img)
        xa_obfs.detach()

################################################################################################################
        ## Create password from protection
        password_a = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))
        # utility_factor = np.random.rand()
        utility_factor = random.randint(0, 1)
        # condition_utility = torch.full((_bs, 1, _w // 2, _h // 2), torch.tensor(utility_factor).float()).to(device)
        utility_cond_init = torch.tensor([float(utility_factor), 1 - float(utility_factor)]).repeat(_bs, 1).to(device)
        utility_condition = utility_fc(utility_cond_init).repeat(1, 4).reshape(_bs, 1, _w // 2, _h // 2)
        condition = torch.concat((password_a, utility_condition), dim=1)

        xa_out_z, xa_proc = embedder(xa, xa_obfs, condition)

        # xa_rev, xa_obfs_rev = embedder(password_a.repeat(1, 4, 1, 1), xa_proc, password_a, rev=True)

        # Feed random noise as input in recovery process
        noise = password_a.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
        xa_rev, xa_obfs_rev = embedder(noise, xa_proc, condition, rev=True)

        loss_utility = 0
        attr_pred, attr_label = (), ()
        loss_triplet_p2p, loss_triplet_p2o = 0, 0

        ## Three kinds of perceptual losses
        loss_img_perc = lpips_loss(xa_obfs, xa_proc)
        loss_img_l1 = l1_loss(xa_obfs, xa_proc)
        loss_image = 5 * loss_img_perc + loss_img_l1

        ## Make correctly recovered image closer to original while further to wrong recovered image
        password_wrong = dwt(torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device))
        condition_wrong = torch.concat((password_wrong, utility_condition), dim=1)
        noise2 = password_wrong.repeat(1, 3, 1, 1) if c.SECRET_KEY_AS_NOISE else gauss_noise((_bs, _c * 4, _w // 2, _h // 2)).to(device)
        xa_rev_wrong, _ = embedder(noise2, xa_proc, condition_wrong, rev=True)

        embedding_orig = recognizer(recognizer.resize(xa))
        embedding_obfs = recognizer(recognizer.resize(xa_obfs))
        embedding_proc = recognizer(recognizer.resize(xa_proc))

        loss_utility = triplet_loss(embedding_orig, embedding_proc, embedding_obfs) \
            if utility_factor else l1_loss(embedding_proc, embedding_obfs)

        # cosine_sim_orig = torch.nn.functional.cosine_similarity(embedding_orig, embedding_obfs)
        # cosine_sim_proc = torch.nn.functional.cosine_similarity(embedding_orig, embedding_proc)
        # cosine_sim_expected = 1 - torch.nn.functional.relu(1 - cosine_sim_orig * np.power(5, utility_factor))
        # loss_utility = l1_loss(cosine_sim_proc, cosine_sim_expected)

        if c.WRONG_RECOVER_TYPE == 'Random':
            loss_rec = l1_loss(xa_rev, xa)
            loss_rec_wrong = percep_triplet_loss(xa, xa_rev, xa_rev_wrong) + triplet_loss(xa, xa_rev, xa_rev_wrong)
            # loss_batch = 0.3 * loss_image + 0.3 * loss_rec + 0.2 * loss_rec_wrong + 0.2 * loss_utility
        else:
            loss_rec = l1_loss(xa_rev, xa) + l1_loss(xa_rev_wrong, xa_obfs)
            loss_rec_wrong = percep_triplet_loss(xa_rev, xa, xa_obfs) + percep_triplet_loss(xa_rev_wrong, xa_obfs, xa)
            # loss_batch = 0.2 * loss_image + 0.3 * loss_rec + 0.5 * loss_rec_wrong

        # loss_batch = 0.3 * loss_image + 0.1 * loss_rec + 0.2 * loss_rec_wrong + 0.4 * loss_utility
        loss_utility_param = 0.5 if utility_factor else 0.1
        loss_batch = 0.3 * loss_image + 0.1 * loss_rec + 0.2 * loss_rec_wrong + loss_utility_param * loss_utility


        # if c.utility_level > c.Utility.NONE:
        #     loss_batch += (loss_utility * c.utility_weights[c.utility_level])


        # Save images
        if (i_batch % c.SAVE_IMAGE_INTERVAL == 0) or (i_batch == 1):
            save_image(normalize(xa),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_orig.jpg", nrow=4)
            save_image(normalize(xa_proc),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_proc.jpg", nrow=4)
            save_image(normalize(xa_obfs),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_{obf_type}.jpg", nrow=4)
            save_image(normalize(xa_rev),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_rev.jpg", nrow=4)
            save_image(normalize(xa_obfs_rev),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_obfs_rev.jpg", nrow=4)
            save_image(clamp_normalize(xa_rev_wrong, lmin=-1, lmax=1),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_rev_wrong.jpg", nrow=4)

        if embedder.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        metrics_batch = {}
        for metric_name, metric_fn in metric_functions.items():
            metrics_batch[metric_name] = metric_fn(attr_pred, attr_label).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
        # metrics['privBudget'] = metrics.get('privBudget', .0) + privacy_budget

        if writer is not None and embedder.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss_img_perc', {mode: loss_img_perc.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_img_l1', {mode: loss_img_l1.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_rec', {mode: loss_rec.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_rec_wrong', {mode: loss_rec_wrong.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_batch', {mode: loss_batch.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_utility', {mode: loss_utility.detach().cpu()}, writer.iteration)
                # if c.utility_level > c.Utility.NONE:
                #     writer.add_scalars('loss_utility', {mode: loss_utility.detach().cpu()}, writer.iteration)
                # if c.utility_level == c.Utility.IDENTITY:
                #     writer.add_scalars('loss_triplet_p2p', {mode: loss_triplet_p2p.detach().cpu()}, writer.iteration)
                #     writer.add_scalars('loss_triplet_p2o', {mode: loss_triplet_p2o.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_img_perc = loss_img_perc.detach().cpu()
        loss_img_perc_total += loss_img_perc
        loss_img_l1 = loss_img_l1.detach().cpu()
        loss_img_l1_total += loss_img_l1
        loss_rec = loss_rec.detach().cpu()
        loss_rec_total += loss_rec
        loss_rec_wrong = loss_rec_wrong.detach().cpu()
        loss_rec_wrong_total += loss_rec_wrong
        loss_batch = loss_batch.detach().cpu()
        loss_batch_total += loss_batch
        loss_utility = loss_utility.detach().cpu()
        loss_utility_total += loss_utility
        # if c.utility_level > c.Utility.NONE:
        #     loss_utility = loss_utility.detach().cpu()
        #     loss_utility_total += loss_utility
        # if c.utility_level == c.Utility.IDENTITY:
        #     loss_triplet_p2p = loss_triplet_p2p.detach().cpu()
        #     loss_triplet_p2p_total += loss_triplet_p2p
        #     loss_triplet_p2o = loss_triplet_p2o.detach().cpu()
        #     loss_triplet_p2o_total += loss_triplet_p2o
        if show_running:
            loss_log = {
                'L_visual': loss_img_perc_total,
                'L_l1': loss_img_l1_total,
                # 'L_p2p': loss_triplet_p2p_total,
                # 'L_p2o': loss_triplet_p2o_total,
                'L_rec': loss_rec_total,
                'L_recx': loss_rec_wrong_total,
                'L_utility': loss_utility_total,
                'L_total': loss_batch_total,
            }
            logger(loss_log, metrics, i_batch)
        else:
            loss_log = {
                'L_visual': loss_img_perc,
                'L_l1': loss_img_l1,
                # 'L_p2p': loss_triplet_p2p,
                # 'L_p2o': loss_triplet_p2o,
                'L_rec': loss_rec,
                'L_recx': loss_rec_wrong,
                'L_utility': loss_utility_total,
                'L_total': loss_batch,
            }
            logger(loss_log, metrics_batch, i_batch)

        # Save face_detection every 5000 iteration
        if (i_batch % c.SAVE_MODEL_INTERVAL == 0) and (mode == 'Train'):
            saved_path = save_model(embedder, optimizer, dir_checkpoint, session, epoch, i_batch)
            models_saved.append(saved_path)


    # print('\n')
    if embedder.training and scheduler is not None:
        scheduler.step()

    loss_img_perc_total = loss_img_perc_total / i_batch
    loss_img_l1_total = loss_img_l1_total / i_batch
    # loss_triplet_p2p_total = loss_triplet_p2p_total / i_batch
    # loss_triplet_p2o_total = loss_triplet_p2o_total / i_batch
    loss_rec_total = loss_rec_total / i_batch
    loss_rec_wrong_total = loss_rec_wrong_total / i_batch
    loss_utility_total = loss_utility_total / i_batch
    loss_batch_total = loss_batch_total / i_batch
    metrics = {k: v / i_batch for k, v in metrics.items()}

    if writer is not None and not embedder.training:
        writer.add_scalars('loss_img_perc', {mode: loss_img_perc_total.detach()}, writer.iteration)
        writer.add_scalars('loss_img_l1', {mode: loss_img_l1_total.detach()}, writer.iteration)
        writer.add_scalars('loss_rec', {mode: loss_rec_total.detach()}, writer.iteration)
        writer.add_scalars('loss_rec_wrong', {mode: loss_rec_wrong_total.detach()}, writer.iteration)
        writer.add_scalars('loss_batch', {mode: loss_batch_total.detach()}, writer.iteration)
        writer.add_scalars('loss_utility', {mode: loss_utility_total.detach()}, writer.iteration)
        # if c.utility_level > c.Utility.NONE:
        #     writer.add_scalars('loss_utility', {mode: loss_utility_total.detach()}, writer.iteration)
        # if c.utility_level == c.Utility.IDENTITY:
        #     writer.add_scalars('loss_triplet_p2p_total', {mode: loss_triplet_p2p_total.detach()}, writer.iteration)
        #     writer.add_scalars('loss_triplet_p2o_total', {mode: loss_triplet_p2o_total.detach()}, writer.iteration)
        # for metric_name, metric in metrics.items():
        #     writer.add_scalars(metric_name, {mode: metric})

    return loss_batch_total, metrics, models_saved


