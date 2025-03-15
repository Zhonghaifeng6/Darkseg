import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from component import *



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, -1, N).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, expansion_factor=4):
        super().__init__()
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.ffn = MixFFN(dim, expansion_factor)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class SegFormerEncoder(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], depths=[2, 2, 2, 2]):
        super().__init__()
        self.stages = nn.ModuleList()

        for i in range(4):
            stage = nn.ModuleList()
            # Patch embedding
            if i == 0:
                patch_embed = OverlapPatchEmbed(in_chans, embed_dims[i], 7, 4, 3)
            else:
                patch_embed = OverlapPatchEmbed(embed_dims[i - 1], embed_dims[i], 3, 2, 1)
            stage.append(patch_embed)

            # Transformer blocks
            for _ in range(depths[i]):
                stage.append(TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    sr_ratio=sr_ratios[i]
                ))
            self.stages.append(stage)

    def forward(self, x):
        features = []
        B = x.shape[0]

        for i, stage in enumerate(self.stages):
            # Patch embedding
            x, H, W = stage[0](x)

            # Transformer blocks
            for j in range(1, len(stage)):
                x = stage[j](x, H, W)

            # Reshape to (B, C, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)

        return features  # [s1, s2, s3, s4]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, enc_channels=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if enc_channels is not None:
            self.enc_conv = nn.Conv2d(enc_channels, in_channels, kernel_size=1)
        else:
            self.enc_conv = None

        conv_in = in_channels * 2 if enc_channels is not None else in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, enc_feat=None):
        x = self.up(x)
        if enc_feat is not None:
            enc_feat = self.enc_conv(enc_feat)
            x = torch.cat([x, enc_feat], dim=1)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder_channels=[32, 64, 160, 256], decoder_channels=[256, 128, 64, 32]):
        super().__init__()
        self.dec4 = DecoderBlock(encoder_channels[3], decoder_channels[0], encoder_channels[2])
        self.dec3 = DecoderBlock(decoder_channels[0], decoder_channels[1], encoder_channels[1])
        self.dec2 = DecoderBlock(decoder_channels[1], decoder_channels[2], encoder_channels[0])
        self.dec1 = DecoderBlock(decoder_channels[2], decoder_channels[3])

    def forward(self, features):
        s1, s2, s3, s4 = features
        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2)
        return d1


class DarkSeg(nn.Module):
    def __init__(self, in_chans=3, num_classes=1):
        super().__init__()

        self.encoder = SegFormerEncoder(in_chans=in_chans)
        self.decoder = Decoder()
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

        self.srsp_1 = SRSP(in_channels=32)
        self.srsp_2 = SRSP(in_channels=64)
        self.srsp_3 = SRSP(in_channels=160)
        self.srsp_4 = SRSP(in_channels=256)

        self.wrsp_1 = WRSP(in_channels=32)
        self.wrsp_2 = WRSP(in_channels=64)
        self.wrsp_3 = WRSP(in_channels=160)
        self.wrsp_4 = WRSP(in_channels=256)

    def forward(self, x):

        mid_features = []
        features = self.encoder(x)

        mid_1 ,mid_2, mid_3, mid_4 = features[0],features[1],features[2],features[3]

        mid_1 = mid_1 + self.srsp_1(mid_1) + self.wrsp_1(mid_1)
        mid_2 = mid_2 + self.srsp_2(mid_2) + self.wrsp_2(mid_2)
        mid_3 = mid_3 + self.srsp_3(mid_3) + self.wrsp_3(mid_3)
        mid_4 = mid_4 + self.srsp_4(mid_4) + self.wrsp_4(mid_4)

        mid_features.append(mid_1)
        mid_features.append(mid_2)
        mid_features.append(mid_3)
        mid_features.append(mid_4)

        d = self.decoder(mid_features)
        out = self.seg_head(d)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out, mid_features


# if __name__ == "__main__":
#     model = DarkSeg(in_chans=3, num_classes=10)
#     x = torch.randn(2, 3, 256, 256)
#     out,mid = model(x)
#     a = mid[0]
#     b = mid[1]
#     c = mid[2]
#     d = mid[3]
#     print(a.shape)  # torch.Size([2, 10, 256, 256])
#     print(b.shape)  # torch.Size([2, 10, 256, 256])
#     print(c.shape)  # torch.Size([2, 10, 256, 256])
#     print(d.shape)  # torch.Size([2, 10, 256, 256])
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# model = DarkSeg(in_chans=3, num_classes=10)
#
# ### Model_params
# total_params = count_parameters(model)
# params_size = (total_params*4)/(1024*1024)
# print(f"Params size (MB): {params_size:.2f} MB")