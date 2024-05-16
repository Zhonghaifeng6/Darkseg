from models import DarkSeg
import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile, clever_format
from utils.config import Config

cfg = Config()

if __name__ == '__main__':
    net = DarkSeg(cfg)
    summary(model=net, input_size=(3,400,400),batch_size=2, device="cpu")

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = DarkSeg(3,3,bilinear=True,fusion=True).to(device)
#     input_shape = (3, 400, 400)
#     input_tensor = torch.randn(1, *input_shape).to(device)
#     flops, params = profile(net, inputs=(input_tensor,))
#     flops, params = clever_format([flops, params], "%.3f")
#     print("FLOPs: %s" % (flops))
#     print("params: %s" % (params))