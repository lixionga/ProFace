
import torch
from torch import nn
from torchkit.backbone import get_model


class ClientBackbone(nn.Module):

    def __init__(self, channels_in, channels_out, client_backbone_name='MobileFaceNet'):
        super(ClientBackbone, self).__init__()

        # feature map size for the cloud
        pri_shape = [torch.randn(torch.Size([1, 64, 56, 56])), torch.randn(torch.Size([1, 128, 28, 28])),
                     torch.randn(torch.Size([1, 256, 14, 14])), torch.randn(torch.Size([1, 512, 7, 7]))]

        # feature map size on the end side
        aux_shape = [torch.randn(torch.Size([1, 64, 28, 28])), torch.randn(torch.Size([1, 128, 14, 14])),
                     torch.randn(torch.Size([1, 128, 7, 7])), torch.randn(torch.Size([1, 512, 7, 7]))]

        self.weight1 = nn.Parameter(torch.ones(pri_shape[0].shape[1], pri_shape[0].shape[2], pri_shape[0].shape[2]))
        self.weight2 = nn.Parameter(torch.ones(pri_shape[1].shape[1], pri_shape[1].shape[2], pri_shape[1].shape[2]))
        self.weight3 = nn.Parameter(torch.ones(pri_shape[2].shape[1], pri_shape[2].shape[2], pri_shape[2].shape[2]))
        self.weight4 = nn.Parameter(torch.ones(pri_shape[3].shape[1], pri_shape[3].shape[2], pri_shape[3].shape[2]))

        self.interface_1 = nn.Sequential(
            nn.Upsample(size=(pri_shape[0].shape[2], pri_shape[0].shape[3]), mode='bilinear'),
            nn.Conv2d(aux_shape[0].shape[1], pri_shape[0].shape[1], (1, 1))
        )
        self.interface_2 = nn.Sequential(
            nn.Upsample(size=(pri_shape[1].shape[2], pri_shape[1].shape[3]), mode='bilinear'),
            nn.Conv2d(aux_shape[1].shape[1], pri_shape[1].shape[1], (1, 1))
        )
        self.interface_3 = nn.Sequential(
            nn.Upsample(size=(pri_shape[2].shape[2], pri_shape[2].shape[3]), mode='bilinear'),
            nn.Conv2d(aux_shape[2].shape[1], pri_shape[2].shape[1], (1, 1))
        )
        self.interface_4 = nn.Sequential(
            nn.Upsample(size=(pri_shape[3].shape[2], pri_shape[3].shape[3]), mode='bilinear'),
            nn.Conv2d(aux_shape[3].shape[1], pri_shape[3].shape[1], (1, 1))
        )

        # MobileFaceNet
        self.client_backbone = get_model(client_backbone_name)

        self.num_blocks = 28
        import random
        channel_permutations_list = []
        for _ in range(self.num_blocks * self.num_blocks):
            channel_permutation = list(range(0, pri_shape[0].shape[1]))
            random.shuffle(channel_permutation)
            channel_permutations_list.append(channel_permutation)

        my_list = torch.tensor(channel_permutations_list)
        self.channel_permutations = nn.Parameter(my_list, requires_grad=False)

        self.client_backbone = self.client_backbone([channels_in, channels_in])
        if client_backbone_name == 'MobileFaceNet':
            self.client_backbone = self._adjust_client_backbone_model(self.client_backbone, channels_in, channels_out)

    def _adjust_client_backbone_model(self, backbone, in_channels, out_channels):
        # mainly to reshape the output layer
        if backbone is None:
            backbone = self.client_backbone

        body = list(backbone.children())[1:-1]

        input_layer = list(backbone.children())[0]

        output_layer = nn.Sequential(
            nn.Conv2d(512, 512, (7, 7), (1, 1), groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        backbone = nn.Sequential()
        backbone.add_module('input_layer', input_layer)
        for i in range(len(body)):
            backbone.add_module('body_{}'.format(str(i)), body[i])
        backbone.add_module('output_layer', output_layer)

        return backbone

    def shuffle_map_byblock(self, input_tensor: torch.Tensor, block_size=8):
        from einops.layers.torch import Rearrange

        feature_map_size = input_tensor.shape[-1]
        num_channels = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]
        num_blocks = feature_map_size // block_size

        to_patch_embedding = Rearrange('b c (h h1) (w w1) -> b c (h w) h1 w1', h1=block_size, w1=block_size)
        blocks = to_patch_embedding(input_tensor)

        channel_permutations = self.channel_permutations.tolist()
        shuffled_blocks = torch.zeros_like(blocks)
        for i in range(num_blocks * num_blocks):
            for j in range(num_channels):
                shuffled_blocks[:, j, i] = blocks[:, channel_permutations[i][j], i]

        to_patch_embedding_two = Rearrange('b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)', p1=num_blocks, p2=num_blocks)
        shuffled_feature_map = to_patch_embedding_two(shuffled_blocks)
        return shuffled_feature_map

    def forward(self, x):
        aux_list = []
        key_list = []

        for idx, module in enumerate(self.client_backbone):
            x = module(x)
            if idx == 2:
                aux_feature = self.interface_1(x)
                aux_feature = self.shuffle_map_byblock(aux_feature, self.num_blocks)
                aux_list.append(aux_feature)
                key_list.append(self.weight1)
            elif idx == 4:
                aux_feature = self.interface_2(x)
                aux_list.append(aux_feature)
                key_list.append(self.weight2)
            elif idx == 6:
                aux_feature = self.interface_3(x)
                aux_list.append(aux_feature)
                key_list.append(self.weight3)
            elif idx == 8:
                aux_feature = self.interface_4(x)
                aux_list.append(aux_feature)
                key_list.append(self.weight4)

        return x, aux_list, key_list


if __name__ == '__main__':
    client_model = ClientBackbone(112, 512).cuda()
    client_model.eval()
    print(client_model)
    input = torch.randn(1, 3, 112, 112).cuda()
    x, aux_list, key_list = client_model(input)
    print(x.shape)
    for i in aux_list:
        print(i.size())
