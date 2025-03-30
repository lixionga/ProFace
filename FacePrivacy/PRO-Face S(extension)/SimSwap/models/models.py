import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from .config import device, num_classes
import functools
import gc
from torch.nn import init

def create_model(opt):
    from .fs_model import fsModel
    model = fsModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)

        return x


class ArcMarginModel(nn.Module):
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# 定义鉴别器
def define_Discriminator(input_nc, ndf, which_model_netD, n_layers_D,
                         normD='batch', init_type='normal', init_gain=0.02,
                   D_names=['M', 'WR', 'R'],
                   ):
    """
    三个鉴别器
    """
    netD = define_D(input_nc, ndf, which_model_netD,
                    n_layers_D=n_layers_D, norm=normD,
                    init_type=init_type, init_gain=init_gain, D_names=D_names,
                    )
    return netD


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02,
             D_names=['M', 'WR', 'R']):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'multiscale_separated':
        netD = MultiscaleDiscriminatorSharedFE(D_names, input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain)



class MultiscaleDiscriminatorSharedFE(nn.Module):
    """
    return a list of patchGAN results for different Ds
    """

    def __init__(self, D_names, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(MultiscaleDiscriminatorSharedFE, self).__init__()
        self.n_layers = n_layers

        netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
        self.FE = netD.model

        for D_name in D_names:
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            setattr(self, 'layer1_' + D_name, netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input, D_name):
        result = []
        input_downsampled = input

        model = getattr(self, 'layer1_' + D_name)
        result.append(self.singleD_forward(model, input_downsampled))

        input_downsampled = self.downsample(input_downsampled)
        result.append(self.singleD_forward(self.FE, input_downsampled))
        return result


# 可能不是passwordGAN原始论文的代码
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        # padw = int(np.ceil((kw - 1.0) / 2))  # 2
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]  # 0 -- 65 or floor(x / 2 + 1) ; 33

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                nn.Conv2d(nf_prev, nf,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(nf) if norm_layer is not None else nn.Sequential(),
                nn.LeakyReLU(0.2, True)
            ]
        # 1 -- 33, 2 -- 17; 17, 9

        nf_prev = nf
        nf = min(nf * 2, 512)
        kw = 4  # if n_layers == 3 else 3
        sequence += [
            nn.Conv2d(nf_prev, nf,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(nf) if norm_layer is not None else nn.Sequential(),
            nn.LeakyReLU(0.2, True)
        ]
        #  3 -- 18; 10
        # only diff: stride

        kw = 4  # if n_layers == 3 else 3
        sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw, bias=False)]
        #  4 -- 19; 11
        # rm bias=False

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_net(net, init_type='normal', init_gain=0.02):
    # net = torch.nn.DataParallel(net).cuda()
    net = net.cuda()
    # net.to("cuda:0")

    gc.collect()
    torch.cuda.empty_cache()
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# GAN loss
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
                Parameters:
                    gan_mode (string)-- the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
                    target_real_label (bool) -- label for a real image
                    target_fake_label (bool)-- label of a fake image
                Note: Do not use sigmoid as the last layer of Discriminator.
                LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def get_single_loss(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                loss += self.get_single_loss(input_i, target_is_real)
            return loss
        else:
            return self.get_single_loss(input, target_is_real)