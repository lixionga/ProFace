import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class SPADE(nn.Module):
    def __init__(self, cin, seg_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(seg_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.alpha = nn.Conv2d(128, cin,
            kernel_size=3, stride=1, padding=1)
        self.beta = nn.Conv2d(128, cin,
            kernel_size=3, stride=1, padding=1)
            
    @staticmethod
    def PN(x):
        '''
            positional normalization: normalize each positional vector along the channel dimension
        '''
        assert len(x.shape) == 4, 'Only works for 4D(image) tensor'
        x = x - x.mean(dim=1, keepdim=True)
        x_norm = x.norm(dim=1, keepdim=True) + 1e-6
        x = x / x_norm
        return x
        
    def DPN(self, x, s):
        h, w = x.shape[2], x.shape[3]
        s = F.interpolate(s, (h, w), mode='bilinear', align_corners = False)
        s = self.conv(s)
        a = self.alpha(s)
        b  = self.beta(s)
        return x * (1 + a) + b

    def forward(self, x, s):
        x_out = self.DPN(self.PN(x), s)
        return x_out
