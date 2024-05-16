import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter
from torch import nn
import torch
from einops import rearrange
from models.transformer import BasicUformerLayer, InputProj
from models.attention import SequentialAttention


class PixelNorm(torch.nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
    def forward(self,x):
        # keep_dim
        out = x / torch.sqrt(torch.mean(x**2,dim = 1,keepdim=True) + self.epsilon)
        return out


class WSLinear(torch.nn.Module):
    def __init__(self,in_features,out_features,gain = 2):
        super(WSLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features,out_features)
        self.scale = (gain / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        #initialize linear layer
        torch.nn.init.normal_(self.linear.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self,x):
        out = self.linear(x * self.scale) + self.bias
        return out


# mapping feature (MT)
class MappingNetwork(torch.nn.Module):
    def __init__(self,z_dim,w_dim):
        super(MappingNetwork, self).__init__()
        self.mapping = torch.nn.Sequential(
            WSLinear(z_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
            WSLinear(w_dim, w_dim),
            torch.nn.ReLU(),
        )
    def forward(self,x):
        return self.mapping(x)


class short_structure(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.structure1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, dilation=1, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU()
        )
        self.structure2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, dilation=3, kernel_size=3, padding=3,
                      bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU()
        )
        self.structure3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, dilation=5, kernel_size=3, padding=5,
                      bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        s1 = self.structure1(x)
        s2 = self.structure2(x)
        s3 = self.structure3(x)
        out = s1 + s2 + s3
        return out



class AdaptiveInstanceNorm_layer1(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.reshape = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=style_dim, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(style_dim)
        )
        self.mapping = MappingNetwork(style_dim,style_dim)
        self.avg = nn.AdaptiveAvgPool2d((1,1))

        self.resize = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(64)
        )

        self.resize_trans = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(16)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4,out_channels=5,kernel_size=1,padding=0,bias=False),
            nn.ReLU()
        )


        self.transformer = SequentialAttention(channel=64)
        self.shorts = short_structure(in_channel=64)
        self.inputprj = InputProj(in_channel=64, out_channel=64, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.trans1 = BasicUformerLayer(dim=64,
                                  output_dim=3,
                                  input_resolution=(200, 200),
                                  depth=3,
                                  num_heads=2,
                                  win_size=8,
                                  mlp_ratio=4,
                                  qkv_bias=True, qk_scale=None,
                                  drop=0, attn_drop=0,
                                  norm_layer=nn.LayerNorm,
                                  use_checkpoint=False,
                                  token_projection='linear', token_mlp='leff',
                                  shift_flag=True)

    def forward(self, input):

        a = input.clone()
        out = self.reshape(input)

        out = self.avg(out)
        out = out.permute(0, 3, 2, 1)
        out = self.mapping(out)
        out = out.permute(0, 3, 2, 1)
        out = self.resize(out * a)

        out_s = self.shorts(out)
        out_l = self.trans1(self.inputprj(out)).reshape(-1,64,200,200)
        out = out_l + out_s
        out_mid = self.transformer(out)
        out = self.resize_trans(out_mid)
        out = self.up(out)

        return out


class AdaptiveInstanceNorm_layer2(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.reshape = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=style_dim, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(style_dim)
        )
        self.mapping = MappingNetwork(style_dim,style_dim)
        self.avg = nn.AdaptiveAvgPool2d((1,1))

        self.resize = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(64)
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=1,padding=0,bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.resize_trans = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(32)
        )

        self.transformer = SequentialAttention(channel=64)
        self.shorts = short_structure(in_channel=64)
        self.inputprj = InputProj(in_channel=64, out_channel=64, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.trans2 = BasicUformerLayer(dim=64,
                                  output_dim=3,
                                  input_resolution=(100, 100),
                                  depth=3,
                                  num_heads=2,
                                  win_size=4,
                                  mlp_ratio=4,
                                  qkv_bias=True, qk_scale=None,
                                  drop=0, attn_drop=0,
                                  norm_layer=nn.LayerNorm,
                                  use_checkpoint=False,
                                  token_projection='linear', token_mlp='leff',
                                  shift_flag=True)

    def forward(self, input):

        a = input.clone()
        out = self.reshape(input)

        out = self.avg(out)
        out = out.permute(0, 3, 2, 1)
        out = self.mapping(out)
        out = out.permute(0, 3, 2, 1)
        out = self.resize(out * a)

        out_s = self.shorts(out)
        out_l = self.trans2(self.inputprj(out)).reshape(-1,64,100,100)
        out = out_s + out_l
        out_mid = self.transformer(out)
        out = self.up(self.resize_trans(out_mid))

        return out


class AdaptiveInstanceNorm_layer3(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.reshape = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=style_dim, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(style_dim)
        )
        self.mapping = MappingNetwork(style_dim,style_dim)
        self.avg = nn.AdaptiveAvgPool2d((1,1))

        self.resize = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(64)
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=1,padding=0,bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.resize_trans = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(64)
        )
        self.transformer = SequentialAttention(channel=64)
        self.shorts = short_structure(in_channel=64)
        self.inputprj = InputProj(in_channel=64, out_channel=64, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.trans3 = BasicUformerLayer(dim=64,
                                  output_dim=3,
                                  input_resolution=(50, 50),
                                  depth=3,
                                  num_heads=2,
                                  win_size=2,
                                  mlp_ratio=4,
                                  qkv_bias=True, qk_scale=None,
                                  drop=0, attn_drop=0,
                                  norm_layer=nn.LayerNorm,
                                  use_checkpoint=False,
                                  token_projection='linear', token_mlp='leff',
                                  shift_flag=True)

    def forward(self, input):

        a = input.clone()
        out = self.reshape(input)
        out = self.avg(out)
        out = out.permute(0, 3, 2, 1)
        out = self.mapping(out)
        out = out.permute(0, 3, 2, 1)
        out = self.resize(out * a)

        out_s = self.shorts(out)
        out_l = self.trans3(self.inputprj(out)).reshape(-1,64,50,50)
        out = out_s + out_l
        out_mid = self.transformer(out)
        out = self.up(self.resize_trans(out_mid))

        return out


if __name__ == '__main__':
    input=torch.randn(2, 64, 50, 50)
    a = AdaptiveInstanceNorm_layer3(in_channel=64,style_dim=64)
    out = a(input)
    print(out.shape)

