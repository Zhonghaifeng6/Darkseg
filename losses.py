import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from LovaszSoftmax.pytorch import lovasz_losses as L


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