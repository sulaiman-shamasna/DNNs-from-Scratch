

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, n_in_channels=3, n_out_classes=2):
        super().__init__()
        self.input_block = EntranceGate(in_channels=n_in_channels, out_channels=64)
        self.block1 = ResBlock(in_channels=64, out_channels=64, stride=1)
        self.block2 = ResBlock(in_channels=64, out_channels=128, stride=2)
        self.block3 = ResBlock(in_channels=128, out_channels=256, stride=2)
        self.block4 = ResBlock(in_channels=256, out_channels=512, stride=2)
        self.out_block = OutBlock(in_ch=512, out_ch=n_out_classes)

    def forward(self, input_tensor):
        first_output = self.input_block(input_tensor)
        res_out1 = self.block1(first_output)
        res_out2 = self.block2(res_out1)
        res_out3 = self.block3(res_out2)
        res_out4 = self.block4(res_out3)
        output_tensor = self.out_block(res_out4)
        return output_tensor

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,  padding=1, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.one_by_one = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        res_out = self.res(input_tensor)
        input_tensor = self.one_by_one(input_tensor)
        input_tensor = self.batchnorm(input_tensor)
        output_tensor = res_out + input_tensor
        return output_tensor


class EntranceGate(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, input_tensor):
        output_tensor = self.gate(input_tensor)
        return output_tensor


class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.FC = nn.Linear(in_ch, out_ch)

    def forward(self, input_tensor):
        averaged = self.avg(input_tensor)
        flattened = torch.flatten(averaged, 1)
        output_tensor = self.FC(flattened)
        output_tensor = F.sigmoid(output_tensor)
        return output_tensor
