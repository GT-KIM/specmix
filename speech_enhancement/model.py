import torch
from torch import nn

class Conv_Block(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None) :
        super().__init__()
        if padding is None :
            padding = (kernel_size-1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x) :
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Deconv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding=(0,0), output_padding=(0,0)):
        super().__init__()

        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Network(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.encoders = []
        self.decoders = []

        self.encoder1 = Conv_Block(2, 45, (7,1), (1,1), (3,0))
        self.encoders.append(self.encoder1)
        self.encoder2 = Conv_Block(45, 45, (1,7), (1,1), (0,3))
        self.encoders.append(self.encoder2)
        self.encoder3 = Conv_Block(45, 90, (7,5), (2,1), (3,2))
        self.encoders.append(self.encoder3)
        self.encoder4 = Conv_Block(90, 90, (7,5), (2,1), (3,2))
        self.encoders.append(self.encoder4)
        self.encoder5 = Conv_Block(90, 90, (5,3), (2,1), (2,1))
        self.encoders.append(self.encoder5)
        self.encoder6 = Conv_Block(90, 90, (5,3), (2,1), (2,1))
        self.encoders.append(self.encoder6)
        self.encoder7 = Conv_Block(90, 90, (5,3), (1,1), (2,1))
        self.encoders.append(self.encoder7)
        self.encoder8 = Conv_Block(90, 64, (5,3), (1,1), (2,1))
        self.encoders.append(self.encoder8)

        self.mid_level = Conv_Block(64,64, (3,3), (1,1), (1,1))

        self.decoder1 = Deconv_Block(64, 90, (5,3), (1,1), (2,1), (0,0))
        self.decoders.append(self.decoder1)
        self.decoder2 = Deconv_Block(90*2, 90, (5,3), (1,1), (2,1), (0,0))
        self.decoders.append(self.decoder2)
        self.decoder3 = Deconv_Block(90*2, 90, (5,3), (2,1), (2,1), (0,0))
        self.decoders.append(self.decoder3)
        self.decoder4 = Deconv_Block(90*2, 90, (5,3), (2,1), (2,1), (0,0))
        self.decoders.append(self.decoder4)
        self.decoder5 = Deconv_Block(90*2, 90, (7,5), (2,1), (3,2), (0,0))
        self.decoders.append(self.decoder5)
        self.decoder6 = Deconv_Block(90*2, 45, (7,5), (2,1), (3,2), (0,0))
        self.decoders.append(self.decoder6)
        self.decoder7 = Deconv_Block(45*2, 45, (1,7), (1,1), (0,3), (0,0))
        self.decoders.append(self.decoder7)
        self.decoder8 = Deconv_Block(45*2, 45, (7,1), (1,1), (3,0), (0,0))
        self.decoders.append(self.decoder8)

        self.last_conv = nn.Conv2d(45, 1, kernel_size=(1,1), stride=(1,1), padding=(0,0))

    def forward(self, x) :
        input = x
        encoder_outs = []
        for i, encoder in enumerate(self.encoders) :
            x = encoder(x)
            encoder_outs.append(x)
        x = self.mid_level(x)
        for i, decoder in enumerate(self.decoders) :
            if not i == 0:
                h = torch.cat((h, encoder_outs[-1 * (i + 1)]), dim=1)
            else:
                h = x
            h = decoder(h)
        mask = h
        mask = self.last_conv(mask)
        mask_expand = torch.cat((mask, mask), dim=1)

        out = mask_expand * input

        return mask, out