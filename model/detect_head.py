import torch.nn as nn
from .cbam import CBAM


class SingleHead(nn.Module):
    def __init__(self, in_channel, out_channel, num_blocks, residual_flag=False, mode="loc"):
        super().__init__()
        layers = []
        
        layers.append(DilationBlock(in_channel, in_channel, residual_flag=residual_flag))
        
        if num_blocks >= 2:
            for _ in range(num_blocks - 1):
                if mode == "dir":
                    layers.append(ResBlock(in_channel, in_channel, residual_flag=residual_flag))
                    layers.append(CBAM(in_channel, ratio=16, kernel_size=7))
                    layers.append(ResBlock(in_channel, in_channel, residual_flag=residual_flag))
                if mode == "loc":
                    layers.append(ResBlock(in_channel, in_channel, residual_flag=residual_flag))
                    layers.append(CBAM(in_channel, ratio=16, kernel_size=7))
                    layers.append(ResBlock(in_channel, in_channel, residual_flag=residual_flag))

        if mode == "dir":
            layers.append(ResBlock(in_channel, out_channel, residual_flag=residual_flag))
        if mode == "loc":
            layers.append(ResBlock(in_channel, out_channel, residual_flag=residual_flag))

        self.sequence = nn.Sequential(*layers)
        self.sequence.apply(self.init_weight)

    def init_weight(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(layer.weight, a=1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, input):
        output =self.sequence(input)
        return output


class DilationBlock(nn.Module):
    def __init__(self, in_channel, out_channel, residual_flag=False):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=4, padding=4, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=8, padding=8, bias=False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=4, padding=4, bias=False),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.conv4 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channel)
        
        self.residual_flag = residual_flag
        if self.residual_flag:
            self.residual_conv = nn.Conv2d(in_channel, out_channel, 1)
            self.residual_conv.apply(self.init_weight)
        
        self.conv1.apply(self.init_weight)
        self.conv2.apply(self.init_weight)
        self.conv3.apply(self.init_weight)
        self.conv4.apply(self.init_weight)

    def init_weight(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(layer.weight, a=1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        if self.residual_flag:
            output = self.bn(self.conv1(x) + self.conv2(x) + self.conv3(x) + self.conv4(x))
        else:
            output = self.bn(self.conv1(x) + self.conv2(x) + self.conv3(x))
        return output


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, residual_flag=True):
        super().__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(in_channel))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        layers.append(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False))
        
        self.sequence = nn.Sequential(*layers)
        self.residual_flag = residual_flag
        self.bn = nn.BatchNorm2d(out_channel)
        
        if self.residual_flag:
            self.residual_conv = nn.Conv2d(in_channel, out_channel, 1, bias=False)
            self.residual_conv.apply(self.init_weight)
        self.sequence.apply(self.init_weight)

    def init_weight(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(layer.weight, a=1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, input):
        if self.residual_flag:
            output = self.bn(self.sequence(input) + self.residual_conv(input))
        else:
            output = self.bn(self.sequence(input))
        return output
