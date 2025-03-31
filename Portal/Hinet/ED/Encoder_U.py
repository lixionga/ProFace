from . import *


class DW_Encoder(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_Encoder, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)

        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message3 = ConvBlock(1, channels, blocks=blocks)
        self.att3 = ResBlock(128 * 2 + channels, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message2 = ConvBlock(1, channels, blocks=blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message1 = ConvBlock(1, channels, blocks=blocks)
        self.att1 = ResBlock(32 * 2 + channels, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.linear0 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message0 = ConvBlock(1, channels, blocks=blocks)
        self.att0 = ResBlock(16 * 2 + channels, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

        self.message_length = message_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def forward(self, x, watermark):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        d4 = self.down4(d3)

        u3 = self.up3(d4)
        expanded_message = self.linear3(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d3.shape[2], d3.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        u3 = torch.cat((d3, u3, expanded_message), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        expanded_message = self.linear2(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)
        u2 = torch.cat((d2, u2, expanded_message), dim=1)
        u2 = self.att2(u2)

        u1 = self.up1(u2)
        expanded_message = self.linear1(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d1.shape[2], d1.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        u1 = torch.cat((d1, u1, expanded_message), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        expanded_message = self.linear0(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d0.shape[2], d0.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        u0 = torch.cat((d0, u0, expanded_message), dim=1)
        u0 = self.att0(u0)

        image = self.Conv_1x1(torch.cat((x, u0), dim=1))

        # forward_image = image.clone().detach()
        # '''read_image = torch.zeros_like(forward_image)
        #
        # for index in range(forward_image.shape[0]):
        #     single_image = ((forward_image[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
        #     im = Image.fromarray(single_image)
        #     read = np.array(im, dtype=np.uint8)
        #     read_image[index] = self.transform(read).unsqueeze(0).to(image.device)
        #
        # gap = read_image - forward_image'''
        # gap = forward_image.clamp(-1, 1) - forward_image
        # return image + gap
        return image


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down, self).__init__()
        self.layer = torch.nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=2),
            ConvBlock(in_channels, out_channels, blocks=blocks)
        )

    def forward(self, x):
        return self.layer(x)


class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
