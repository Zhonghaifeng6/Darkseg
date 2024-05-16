import torch.nn as nn
from models.Mutual_FIM_SWtrans import *
from models.Mutual_FIM_WRfa import *
from models.model_base import *
from models.nested_unet_base import *


class DoubleConv(nn.Module):
    """
    convolution => [BN] => [ReLU]
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x):
        x1 = self.res_conv1(x)
        x2 = self.res_conv2(x1)
        out = x1 + x2
        return out


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class DarkSeg(nn.Module):
    def __init__(self,cfg):
        super(DarkSeg, self).__init__()
        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.bilinear = cfg.bilinear
        self.fusion = cfg.fusion

        self.inc = DoubleConv(self.n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256, self.bilinear)
        self.up2 = Up(512, 128, self.bilinear)
        self.up3 = Up(256, 64, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.out_put = OutConv(64, self.n_classes)

        self.lfg1 = latent_feature_guidence_1(128)
        self.lfg2 = latent_feature_guidence_2(256)
        self.lfg3 = latent_feature_guidence_3(512)
        self.lfg4 = latent_feature_guidence_4(512)

        self.re_chann_1 = nn.Conv2d(256,  128, kernel_size=1)
        self.re_chann_2 = nn.Conv2d(512,  256, kernel_size=1)
        self.re_chann_3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.re_chann_4 = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        mid_1 = self.lfg1(x2)
        mid_2 = self.lfg2(x3)
        mid_3 = self.lfg3(x4)
        mid_4 = self.lfg4(x5)


        if self.fusion:
            x = self.up1(x5+mid_4, x4+mid_3)
            x = self.up2(x, x3+mid_2)
            x = self.up3(x, x2+mid_1)
            x = self.up4(x, x1)
            logits = self.out_put(x)
        else:
            x5 = self.re_chann_4(torch.cat([mid_4, x5], dim=1))
            x4 = self.re_chann_3(torch.cat([mid_3, x4], dim=1))
            x3 = self.re_chann_2(torch.cat([mid_2, x3], dim=1))
            x2 = self.re_chann_1(torch.cat([mid_1, x2], dim=1))
            x = self.up1(x5, x4)
            x = self.up2(x,  x3)
            x = self.up3(x,  x2)
            x = self.up4(x,  x1)
            logits = self.out_put(x)

        return logits,mid_1,mid_2,mid_3,mid_4


class DarkSeg_Nest(nn.Module):
    def __init__(self, cfg):
        super(DarkSeg_Nest,self).__init__()

        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.deepsupervision = cfg.deepsupervision
        self.bilinear = cfg.bilinear

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final1 = nn.Conv2d(nb_filter[0], 128, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], 128, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], 128, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], 128, kernel_size=1)

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)

            return output, output






