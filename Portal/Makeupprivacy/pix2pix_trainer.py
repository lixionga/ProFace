import torch
import torch.nn.functional as F
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
            self.clamp=self.pix2pix_model_on_one_gpu.WGAN_clamp()
    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated, a_label, b_label, warped_features = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated
        self.a_label = a_label
        self.b_label = b_label
        self.warped_features = warped_features

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.clamp
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated
    
    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    def get_visual_a_label(self):
        return self.get_color_label(self.a_label)

    def get_visual_b_label(self):
        return self.get_color_label(self.b_label)

    def get_visual_warped_features(self,layer):
        channel_size = self.warped_features[layer-1].shape[1]
        kernel = torch.Tensor(3, channel_size, 1, 1).fill_(1 / channel_size).to(self.warped_features[layer-1].device)
        return F.conv2d(self.warped_features[layer-1], kernel, stride=1)

    @staticmethod
    def get_color_label(label_tensor):
        device = label_tensor.device
        part_colors = torch.FloatTensor([[255, 0, 0], [255, 85, 0], [255, 170, 0],
                    [255, 0, 85], [255, 0, 170],
                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
                    [0, 255, 85], [0, 255, 170],
                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
                    [0, 85, 255], [0, 170, 255],
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]).to(device)
        color_label = torch.FloatTensor(label_tensor.shape[0], label_tensor.shape[2], label_tensor.shape[3], 3).to(device)
        label = torch.argmax(label_tensor, 1) + 1
        num_of_class = torch.max(label)
        for pi in range(1, num_of_class + 1):
            color_label[(label == pi)] = part_colors[pi]
        color_label = (color_label.permute(0, 3, 1, 2) / 255 - 0.5) * 2.0
        return color_label


    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
