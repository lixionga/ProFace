import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import logging
import random
from torch.cuda import amp
import torch
import torch.fft
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
import net
from torchkit.util import AverageMeter, Timer
from torchkit.util import accuracy_dist
from torchkit.loss import get_loss
from torchkit.task import BaseTask
import torchvision.transforms.functional as F
import kornia

from validation.util.utils import get_val_data, perform_val

class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """

    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

        self.flag = True
        self.adaface = load_pretrained_model(architecture='ir_101')

    def loop_step(self, epoch):

        backbone, heads = self.backbone, list(self.heads.values())
        backbone.train()

        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes
        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        am_L2_losses = [AverageMeter() for _ in batch_sizes]

        t = Timer()

        # ------------------train start-----------------------------------
        for step, samples in enumerate(self.train_loader):
            list_patch = [4, 9, 16, 25]
            patch = random.choice(list_patch)
            self.Blur = MyBlurWay(patch, 112)

            self.call_hook("before_train_iter", step, epoch)
            backbone_opt, head_opts = self.opt['backbone'], list(self.opt['heads'].values())

            inputs = samples[0].cuda()
            labels = samples[1].cuda()
            assert not torch.any(torch.isnan(inputs))
            assert not torch.any(torch.isnan(labels))

            # blur image
            blur_a = self.Blur(inputs)
            res_input = inputs - blur_a

            aux_features, aux_list, key_list = backbone(res_input)
            assert torch.isnan(aux_features).sum() == 0, print(aux_features)

            features, _ = self.adaface(blur_a, aux_list, key_list)

            target_features, _ = self.adaface(inputs)

            # L2 loss
            L2_loss = nn.MSELoss()
            loss_l2 = L2_loss(features, target_features)

            outputs, labels, original_outputs = heads[0](features, labels)

            loss = self.loss(outputs, labels)

            for i in range(len(batch_sizes)):
                prec1, prec5 = accuracy_dist(self.cfg,
                                             original_outputs.data,
                                             labels,
                                             self.class_shards[i],
                                             topk=(1, 5))
                am_losses[i].update(loss.data.item(), features[i].size(0))
                am_L2_losses[i].update(loss_l2.data.item(), features[i].size(0))
                am_top1s[i].update(prec1.data.item(), features[i].size(0))
                am_top5s[i].update(prec5.data.item(), features[i].size(0))

            # compute loss
            total_loss = loss + loss_l2
            assert torch.isnan(total_loss).sum() == 0, print(total_loss)

            # compute gradient and do SGD
            backbone_opt.zero_grad()

            for head_opt in head_opts:
                head_opt.zero_grad()

            total_loss.backward()

            backbone_opt.step()
            for head_opt in head_opts:
                head_opt.step()

            scalars = {
                'train/loss': am_losses,
                'train/top1': am_top1s,
                'train/top5': am_top5s,
            }
            self.update_summary({'scalars': scalars})
            log = {
                'loss': am_losses,
                'prec@1': am_top1s,
                'prec@5': am_top5s,
            }
            self.update_log_buffer(log)

            cost = t.get_duration()

            self.update_log_buffer({'time_cost': cost})

            self.call_hook("after_train_iter", step, epoch)

        # -------------------------test start-------------------------------------
        backbone.eval()
        self.logging.info('Test Epoch: {} ...'.format(epoch + 1))

        # datas
        lfw, cfp_fp, agedb, calfw, cplfw, lfw_issame, cfp_fp_issame, \
        agedb_issame, calfw_issame, cplfw_issame = get_val_data(self.val_data_root)

        auxi_model = load_pretrained_model(architecture='ir_101')

        # Gaussian Blur
        accuracy_lfw_Gau, best_threshold_lfw_Gau, roc_curve_lfw_Gau = perform_val(embedding_size=512, batch_size=256,
                                                                                  backbone=backbone, carray=lfw,
                                                                                  issame=lfw_issame,
                                                                                  aux_model=auxi_model,
                                                                                  blur_type=0)
        self.logging.info("Gaussian：")
        self.logging.info('accuracy_lfw: {:.4f}, threshold_lfw: {:.4f}'
                          .format(accuracy_lfw_Gau * 100, best_threshold_lfw_Gau))

        # pixelate
        accuracy_lfw_Pix, best_threshold_lfw_Pix, roc_curve_lfw_Pix = perform_val(embedding_size=512, batch_size=256,
                                                                                  backbone=backbone, carray=lfw,
                                                                                  issame=lfw_issame,
                                                                                  aux_model=auxi_model,
                                                                                  blur_type=2)

        self.logging.info("Pixelate：")
        self.logging.info('accuracy_lfw: {:.4f}, threshold_lfw: {:.4f}'
                          .format(accuracy_lfw_Pix * 100, best_threshold_lfw_Pix))

        # median filter
        accuracy_lfw_Med, best_threshold_lfw_Med, roc_curve_lfw_Med = perform_val(embedding_size=512, batch_size=256,
                                                                                  backbone=backbone, carray=lfw,
                                                                                  issame=lfw_issame,
                                                                                  aux_model=auxi_model,
                                                                                  blur_type=1)

        self.logging.info("Median：")
        self.logging.info('accuracy_lfw: {:.4f}, threshold_lfw: {:.4f}'
                          .format(accuracy_lfw_Med * 100, best_threshold_lfw_Med))

        # hybrid
        accuracy_lfw_My, best_threshold_lfw_My, roc_curve_lfw_My = perform_val(embedding_size=512, batch_size=512,
                                                                               backbone=backbone, carray=lfw,
                                                                               issame=lfw_issame,
                                                                               aux_model=auxi_model,
                                                                               blur_type=3)

        self.logging.info("hybrid:")
        self.logging.info('accuracy_lfw: {:.4f}, threshold_lfw: {:.4f}'
                          .format(accuracy_lfw_My * 100, best_threshold_lfw_My))

    def prepare(self):
        """ common prepare task for training
        """
        for key in self.cfg:
            print(key, self.cfg[key])
        self.make_inputs()

        self.make_model()

        self.loss = get_loss('DistCrossEntropy').cuda()
        self.cross_entropy_loss = CrossEntropyLoss().cuda()
        self.opt = self.get_optimizer()
        self.scaler = amp.GradScaler()
        self.register_hooks()
        self.pfc = self.cfg['HEAD_NAME'] == 'PartialFC'

    def train(self):
        self.prepare()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            self.loop_step(epoch)
            self.call_hook("after_train_epoch", epoch)
        self.call_hook("after_run")


# Gaussian Blur
class Blur(torch.nn.Module):
    def __init__(self, kernel_size, sigma_min, sigma_max):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_avg = (sigma_min + sigma_max) / 2

    def forward(self, img, _=None):
        sigma = random.uniform(self.sigma_min, self.sigma_max) if self.training else self.sigma_avg
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
        block_size = random.randint(self.block_size_min, self.block_size_max) if self.training else self.block_size_avg

        pixelated_size = img_size // block_size
        img_pixelated = F.resize(F.resize(img, pixelated_size), img_size, F.InterpolationMode.NEAREST)
        return img_pixelated


# median filter
class MedianBlur(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel = kernel_size
        self.size_min = kernel_size - 7
        self.size_max = kernel_size + 7
        self.number = [7, 9, 11, 13, 15, 17, 19]

    def forward(self, img, _=None):
        kernel_size = random.sample(self.number, 1)[0] if self.training else self.kernel
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
        """
        Args:
            img : tensor (c h w)
        Returns:
            img : tensor (c h w)
        """
        img = img.cuda()

        p = int(self.size / self.patch_r)
        from torchvision import transforms
        if p * self.patch_r != self.size:
            resize_img = transforms.Resize((p + 1) * self.patch_r)
            img = resize_img(img).cuda()
            p += 1

        medianBlur = MedianBlur(13).train()
        med_image = medianBlur(img)

        GausBlur = Blur(31, 2, 8).train()
        gaus_image = GausBlur(img)

        PixBlur = Pixelate(7).train()
        pix_image = PixBlur(img)

        from einops.layers.torch import Rearrange
        # the image blocks of each kind of blurred image are acquired
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

        # get a random number
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

        # combined image block
        truple_image = tuple(blur_img_list)
        tmp_blur_img = torch.cat(truple_image, dim=1)
        to_patch_embedding = Rearrange('b (h w) c p1 p2  -> b c (h p1) (w p2)', h=self.patch_r, w=self.patch_r)
        blur_image = to_patch_embedding(tmp_blur_img)

        resize_img_two = transforms.Resize(self.size)
        blur_image = resize_img_two(blur_image).cuda()

        return blur_image

adaface_models = {
    'ir_50': "../../model/adaface_ir50_webface4m.ckpt",
    'ir_101': "../../model/adaface_ir101_webface12m.ckpt"
}

# AdaFace
def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture).cuda()
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)

    model = model.cuda()
    model.eval()
    logging.info("AdaFace load success!")
    return model


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train.yaml'))
    task.init_env()
    task.train()


if __name__ == "__main__":
    main()
