import torch
import torch.nn as nn
import torch.nn.functional as F

# for CIFAR100
coarse_classes = 5
fine_classes = 100

class InnerComplementEntropy(nn.Module):

    def __init__(self, fine2coarse):
        super(InnerComplementEntropy, self).__init__()
        self.fine2coarse  = fine2coarse

    def forward(self, yHat, y_fine):
        self.batch_size = len(y_fine)
        self.coarse_classes = coarse_classes
        y_coarse = torch.unsqueeze(torch.from_numpy(self.fine2coarse[y_fine.cpu()]), 1) 
        y_G = torch.topk((torch.from_numpy(self.fine2coarse)==y_coarse).int(), coarse_classes)[1].cuda()
        yHat_G = F.softmax(torch.gather(yHat, 1, y_G).float(), dim=1)
        new_Yg_index = torch.topk((y_G==torch.unsqueeze(y_fine, 1)).int(), 1)[1].cuda()
        Yg = torch.gather(yHat_G, 1, new_Yg_index)
        Yg_ = (1. - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat_G / Yg_.view(len(yHat_G), 1)
        Px_log = torch.log(Px.clamp(min=1e-10))  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, coarse_classes).scatter_(
            1, new_Yg_index.data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.coarse_classes)
        return loss

class OuterComplementEntropy(nn.Module):

    def __init__(self, fine2coarse):
        super(OuterComplementEntropy, self).__init__()
        self.fine2coarse  = fine2coarse

    def forward(self, yHat, y_fine):
        self.batch_size = len(y_fine)
        self.fine_classes = fine_classes
        yHat = F.softmax(yHat, dim=1)
        y_coarse = torch.unsqueeze(torch.from_numpy(self.fine2coarse[y_fine.cpu()]), 1)
        y_G = torch.topk((torch.from_numpy(self.fine2coarse)==y_coarse).int(), 5)[1].cuda()
        Yg = torch.sum(torch.gather(yHat, 1, y_G), dim=1)
        Yg_ = (1. - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px.clamp(min=1e-10))  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.fine_classes).scatter_(
            1, y_G.data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.fine_classes)
        return loss
