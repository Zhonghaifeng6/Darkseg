import torch
import torch.nn as nn



class SequentialPolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel//2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel//2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel//2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel//2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        # channel-only self-attention
        channel_wv = self.ch_wv(x) # bs,c//2,h,w
        channel_wq = self.ch_wq(x) # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c//2, -1) # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1) # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1) # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1) # bs,c,1,1
        channel_out = channel_weight*x

        # spatial-only self-attention
        spatial_wv = self.sp_wv(channel_out) # bs,c//2,h,w
        spatial_wq = self.sp_wq(channel_out) # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq) # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c//2, -1) # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c//2) # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq,spatial_wv) # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w)) # bs,1,h,w
        spatial_out = spatial_weight*channel_out
        return spatial_out

class SRSP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 4 == 0, "Input channels must be divisible by 4"

        self.split_channels = in_channels // 4

        self.conv1 = nn.Conv2d(
            self.split_channels, self.split_channels,
            kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.conv2 = nn.Conv2d(
            self.split_channels, self.split_channels,
            kernel_size=3, stride=1, padding=2, dilation=2
        )
        self.conv3 = nn.Conv2d(
            self.split_channels, self.split_channels,
            kernel_size=3, stride=1, padding=3, dilation=3
        )
        self.conv4 = nn.Conv2d(
            self.split_channels, self.split_channels,
            kernel_size=3, stride=1, padding=5, dilation=5
        )

    def forward(self, x):
        # cut channel
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        w1 = torch.sigmoid(self.conv1(x1))
        w2 = torch.sigmoid(self.conv2(x2))
        w3 = torch.sigmoid(self.conv3(x3))
        w4 = torch.sigmoid(self.conv4(x4))

        out1 = x1 * w1
        out2 = x2 * w2
        out3 = x3 * w3
        out4 = x4 * w4

        # concatenate channel
        return torch.cat([out1, out2, out3, out4], dim=1)


class MSMT(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels

        # 初始化卷积分支定义
        self.conv_init = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=3, padding=1, dilation=1)
        self.conv_q = nn.Conv2d(in_channels, in_channels,
                                kernel_size=3, padding=3, dilation=3)
        self.conv_k = nn.Conv2d(in_channels, in_channels,
                                kernel_size=3, padding=5, dilation=5)
        self.conv_v = nn.Conv2d(in_channels, in_channels,
                                kernel_size=3, padding=7, dilation=7)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # MLP增强
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # init, Q, K, V
        init = self.conv_init(x)
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        # flatten
        B, C, H, W = x.shape
        q = q.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        k = k.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        v = v.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]

        # self-attention computing
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, N, N]
        attn = attn / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # 添加Dropout
        out = torch.matmul(attn, v)  # [B, N, C]

        # result add to the init
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = out + init

        # MLP non-Liner
        out = out + self.mlp(out)
        return out


class WRSP(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        self.split_channels = in_channels // 2
        self.msmt = MSMT(in_channels//2, dropout)

    def forward(self, x):

        # cut channel
        x1, x2 = torch.chunk(x, 2, dim=1)
        unit_1 = self.msmt(x1) + x2
        unit_2 = self.msmt(unit_1) + x1

        # concatenate channel
        return torch.cat([unit_1, unit_2], dim=1)



model = SRSP(in_channels=256)
x = torch.randn(2, 256, 32, 32)  # batch_size=2, 32x32特征图
output = model(x)
print(output.shape)  # torch.Size([2, 256, 32, 32])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 创建模型
model = WRSP(in_channels=256)

### Model_params
total_params = count_parameters(model)
params_size = (total_params*4)/(1024*1024)
print(f"Params size (MB): {params_size:.2f} MB")