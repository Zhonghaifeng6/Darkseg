import torch
from torch import nn
from models.w_r_model import shrot_rang_fa, wide_rang_fa
from models.movb_Trans import *


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

# mapping feature
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


##  channel:64,128,256,512,512
class latent_feature_guidence_1_mb(torch.nn.Module):

    def __init__(self, input_dim):
        super(latent_feature_guidence_1_mb, self).__init__()

        self.srfa = shrot_rang_fa(128)
        self.mobt = Mobile_vision_Transformer_Attention(in_channel=512)
        self.mapping = MappingNetwork(512,512)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.resize_in = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.resize_out = nn.Sequential(
            nn.Conv2d(512, input_dim, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim)
        )

    def forward(self, x):
        x = self.resize_in(x)
        x_avg = self.avg(x)
        x_avg = x_avg.permute(0, 3, 2, 1)
        x_mt  = self.mapping(x_avg)
        x_mt  = x_mt.permute(0, 3, 2, 1)
        x_mid = x * x_mt
        x_wr = self.mobt(x_mid)
        x_sr = self.srfa(x_mid)
        x_fusion = x_wr + x_sr
        out = self.resize_out(x_fusion)
        return out


class latent_feature_guidence_2_mb(torch.nn.Module):

    def __init__(self, input_dim):
        super(latent_feature_guidence_2_mb, self).__init__()

        self.srfa = shrot_rang_fa(128)
        self.mobt = Mobile_vision_Transformer_Attention(in_channel=512)
        self.mapping = MappingNetwork(512,512)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.resize_in = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.resize_out = nn.Sequential(
            nn.Conv2d(512, input_dim, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim)
        )

    def forward(self, x):
        x = self.resize_in(x)
        x_avg = self.avg(x)
        x_avg = x_avg.permute(0, 3, 2, 1)
        x_mt  = self.mapping(x_avg)
        x_mt  = x_mt.permute(0, 3, 2, 1)
        x_mid = x * x_mt
        x_wr = self.mobt(x_mid)
        x_sr = self.srfa(x_mid)
        x_fusion = x_wr + x_sr
        out = self.resize_out(x_fusion)
        return out


class latent_feature_guidence_3_mb(torch.nn.Module):

    def __init__(self, input_dim):
        super(latent_feature_guidence_3_mb, self).__init__()

        self.srfa = shrot_rang_fa(128)
        self.mobt = Mobile_vision_Transformer_Attention(in_channel=512)
        self.mapping = MappingNetwork(512,512)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.resize_in = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.resize_out = nn.Sequential(
            nn.Conv2d(512, input_dim, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim)
        )

    def forward(self, x):
        x = self.resize_in(x)
        x_avg = self.avg(x)
        x_avg = x_avg.permute(0, 3, 2, 1)
        x_mt  = self.mapping(x_avg)
        x_mt  = x_mt.permute(0, 3, 2, 1)
        x_mid = x * x_mt
        x_wr = self.mobt(x_mid)
        x_sr = self.srfa(x_mid)
        x_fusion = x_wr + x_sr
        out = self.resize_out(x_fusion)
        return out


class latent_feature_guidence_4_mb(torch.nn.Module):

    def __init__(self, input_dim):
        super(latent_feature_guidence_4_mb, self).__init__()

        self.srfa = shrot_rang_fa(128)
        self.mobt = Mobile_vision_Transformer_Attention(in_channel=512)
        self.mapping = MappingNetwork(512,512)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.resize_in = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.resize_out = nn.Sequential(
            nn.Conv2d(512, input_dim, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim)
        )

    def forward(self, x):
        x = self.resize_in(x)
        x_avg = self.avg(x)
        x_avg = x_avg.permute(0, 3, 2, 1)
        x_mt  = self.mapping(x_avg)
        x_mt  = x_mt.permute(0, 3, 2, 1)
        x_mid = x * x_mt
        x_wr = self.mobt(x_mid)
        x_sr = self.srfa(x_mid)
        x_fusion = x_wr + x_sr
        out = self.resize_out(x_fusion)
        return out

