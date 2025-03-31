from . import *


class DW_Decoder(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_Decoder, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)

        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.att3 = ResBlock(128 * 2, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.att2 = ResBlock(64 * 2, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.att1 = ResBlock(32 * 2, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.att0 = ResBlock(16 * 2, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.message_layer = nn.Linear(message_length * message_length, message_length)
        self.message_length = message_length

    def forward(self, x):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        d4 = self.down4(d3)

        u3 = self.up3(d4)
        u3 = torch.cat((d3, u3), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat((d2, u2), dim=1)
        u2 = self.att2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat((d1, u1), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        u0 = torch.cat((d0, u0), dim=1)
        u0 = self.att0(u0)

        residual = self.Conv_1x1(u0)

        message = F.interpolate(residual, size=(self.message_length, self.message_length),
                                mode='nearest')
        message = message.view(message.shape[0], -1)
        message = self.message_layer(message)

        return message


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
