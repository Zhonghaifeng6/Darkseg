import torch
from torch import nn


class feature_divide(nn.Module):
    def __init__(self,input):
        super().__init__()

        self.cha_0_1 = nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, padding=0, stride=1)
        self.cha_1_2 = nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, padding=0, stride=1)
        self.cha_2_3 = nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, padding=0, stride=1)
        self.cha_3_4 = nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, padding=0, stride=1)

    def forward(self,x):

        channels_per_part = x.size(1) // 4
        parts = []
        # cut channel
        for i in range(4):
            start_channel = i * channels_per_part
            end_channel = (i + 1) * channels_per_part
            part = x[:, start_channel:end_channel, :, :]
            parts.append(part)

        out_1 = self.cha_0_1(parts[0])
        out_2 = self.cha_1_2(parts[1])
        out_3 = self.cha_2_3(parts[2])
        out_4 = self.cha_3_4(parts[3])

        return out_1,out_2,out_3,out_4


class shrot_rang_fa(nn.Module):
    def __init__(self, aim):
        super().__init__()
        self.divide = feature_divide(aim)

        self.unit1 = nn.Sequential(
            nn.Conv2d(aim, aim, kernel_size=3, padding=1, dilation=1, stride=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(aim)
        )
        self.unit2 = nn.Sequential(
            nn.Conv2d(aim, aim, kernel_size=3, padding=2, dilation=2, stride=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(aim)
        )
        self.unit3 = nn.Sequential(
            nn.Conv2d(aim, aim, kernel_size=3, padding=3, dilation=3, stride=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(aim)
        )
        self.unit4 = nn.Sequential(
            nn.Conv2d(aim, aim, kernel_size=3, padding=5, dilation=5, stride=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(aim)
        )

    def forward(self, x):
        c1, c2, c3, c4 = self.divide(x)
        d_1 = self.unit1(c1)
        d_2 = self.unit2(c2)
        d_3 = self.unit3(c3)
        d_4 = self.unit4(c4)
        part_1, part_2, part_3, part_4 = d_1*c1, d_2*c2, d_3*c3, d_4*c4
        out = torch.cat([part_1,part_2,part_3,part_4],dim=1)
        return out



class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        # Depth_wise
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        # Point_wise
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class Multilayer_perceptron(nn.Module):
    def __init__(self,dim_input, mlp_dim, dim_output, dim, dropout=0.) :
        super().__init__()
        self.net=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv2d(dim_input,mlp_dim, kernel_size=1,padding=0,stride=1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_dim,dim_output,kernel_size=1,padding=0,stride=1),
            nn.Dropout(dropout),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.net(x)


class wide_rang_fa(nn.Module):
    def __init__(self, input, output, dim_h):
        super().__init__()
        self.dw_conv_1 = nn.Sequential(
            DepthwiseSeparableConv2d(input,output, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(output)
        )
        self.dw_conv_2 = nn.Sequential(
            DepthwiseSeparableConv2d(input, output, 3, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(output)
        )
        self.dw_conv_3 = nn.Sequential(
            DepthwiseSeparableConv2d(input, output, 5, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(output)
        )
        self.dw_conv_4 = nn.Sequential(
            DepthwiseSeparableConv2d(input, output, 7, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(output)
        )
        self.sfotmax = nn.Softmax()
        self.atten_map = nn.Conv2d(output, 1, kernel_size=1, padding=0, stride=1)
        self.mlp = Multilayer_perceptron(output,input*2,input, dim_h)

    def forward(self, x):

        init  = self.dw_conv_1(x)
        query = self.dw_conv_2(x)
        value = self.dw_conv_3(x)
        key   = self.dw_conv_4(x)
        atten = self.sfotmax(self.atten_map(query * key))
        mid_atten  = atten * value
        init_atten = mid_atten + init
        out = x * self.mlp(init_atten)
        return out


