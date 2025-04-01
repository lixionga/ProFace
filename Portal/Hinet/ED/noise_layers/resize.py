import torch.nn as nn
import torch.nn.functional as F


class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, down_scale=0.5):
        super(Resize, self).__init__()
        self.down_scale = down_scale

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        noised_down = F.interpolate(image,size=(int(self.down_scale * image.shape[2]), int(self.down_scale * image.shape[3])),mode='nearest')
        noised_up = F.interpolate(noised_down,size=(image.shape[2], image.shape[3]),mode='nearest')
        return noised_up


