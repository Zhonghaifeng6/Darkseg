from torch.autograd import Function

from LovaszSoftmax.pytorch import lovasz_losses as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class Gradient_Loss(nn.Module):
    def __init__(self, losstype='l2'):
        super(Gradient_Loss, self).__init__()
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        a = torch.from_numpy(a).float().unsqueeze(0)
        a = torch.stack((a, a, a))
        conv1.weight = nn.Parameter(a, requires_grad=False)
        self.conv1 = conv1.cuda()

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        b = torch.from_numpy(b).float().unsqueeze(0)
        b = torch.stack((b, b, b))
        conv2.weight = nn.Parameter(b, requires_grad=False)
        self.conv2 = conv2.cuda()

        # self.Loss_criterion = ReconstructionLoss(losstype)
        self.Loss_criterion = nn.L1Loss()

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        # x_total = torch.sqrt(torch.pow(x1, 2) + torch.pow(x2, 2))

        y1 = self.conv1(y)
        y2 = self.conv2(y)
        # y_total = torch.sqrt(torch.pow(y1, 2) + torch.pow(y2, 2))

        l_h = self.Loss_criterion(x1, y1)
        l_v = self.Loss_criterion(x2, y2)
        # l_total = self.Loss_criterion(x_total, y_total)
        return l_h + l_v #+ l_total



class LovaszLossSoftmax(nn.Module):
    def __init__(self):
        super(LovaszLossSoftmax, self).__init__()

    def forward(self, input, target):
        out = F.softmax(input, dim=1)
        loss = L.lovasz_softmax(out, target)
        return loss


class LovaszLossSoftmax_mutul(nn.Module):
    def __init__(self):
        super(LovaszLossSoftmax_mutul, self).__init__()

    def forward(self, input, target):
        out1 = F.softmax(input, dim=1)
        out2 = F.softmax(target, dim=1)
        loss = L.lovasz_softmax(out1, out2)
        return loss


class LovaszLossHinge(nn.Module):
    def __init__(self):
        super(LovaszLossHinge, self).__init__()

    def forward(self, input, target):
        loss = L.lovasz_hinge(input, target)
        return loss


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class MIOULoss(nn.Module):
    def __init__(self, num_classes):
        super(MIOULoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, predicted, target):
        # Flatten predicted and target tensors
        predicted = predicted.view(-1)
        target = target.view(-1)

        # Create one-hot encoding of target labels
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()

        # Calculate intersection and union
        intersection = torch.sum(predicted * target_one_hot)
        union = torch.sum(predicted) + torch.sum(target_one_hot) - intersection + 1e-7  # Add a small epsilon to avoid division by zero

        # Calculate IOU
        iou = intersection / union

        # Calculate MIoU loss
        miou_loss = 1 - iou

        return miou_loss