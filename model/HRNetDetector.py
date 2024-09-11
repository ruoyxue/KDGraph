import torch.nn as nn
import torch.nn.functional as F
import torch
from .detect_head import SingleHead, ResBlock
from .cbam import CBAM
from .HRNet import HighResolutionNet
import os


class HRNetDetector(nn.Module):
    def __init__(self, in_channel=3, backbone='hr-w32', pretrained_flag=True):
        super().__init__()
        features = 128  # channel of multi-scale feature maps

        assert backbone in ['hr-w32', 'hr-w48']

        if backbone == "hr-w48":
            self.conv_before_detect_loc = nn.Conv2d(48 + 96 + 192 + 384, features, 1, bias=False)
            self.conv_before_detect_dir = nn.Conv2d(48 + 96 + 192 + 384, features, 1, bias=False)
            
        elif backbone == "hr-w32":
            self.conv_before_detect_loc = nn.Conv2d(32 + 64 + 128 + 256, features, 1, bias=False)
            self.conv_before_detect_dir = nn.Conv2d(32 + 64 + 128 + 256, features, 1, bias=False)

        self.backbone = HighResolutionNet(in_channel, backbone)

        self.location_detector = SingleHead(features, features, num_blocks=4, residual_flag=True,
                                            mode="loc")
        self.direction_detector = SingleHead(features, features, num_blocks=4, residual_flag=True,
                                             mode="dir")
        
        self.up_block_loc = UpSampleBlock_new2(in_ch=features, out_ch=1)
        self.up_block_dir = UpSampleBlock_new2(in_ch=features, out_ch=18)
        self.init_weights(None)

    def forward(self, x):
        x = self.backbone(x)

        height, width = x[0].shape[2:]
        x0 = x[0]
        x1 = F.interpolate(x[1], size=(height, width), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(height, width), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(height, width), mode='bilinear', align_corners=True)

        concat_feat_loc = self.conv_before_detect_loc(torch.concat([x0, x1, x2, x3], dim=1))
        concat_feat_dir = self.conv_before_detect_dir(torch.concat([x0, x1, x2, x3], dim=1))
        loc_feat = self.location_detector(concat_feat_loc)
        dir_feat = self.direction_detector(concat_feat_dir)
        
        up_loc_feat = self.up_block_loc(loc_feat)
        up_dir_feat = self.up_block_dir(dir_feat)

        return up_loc_feat, up_dir_feat

    def init_weights(self, pretrained=''):
        if pretrained is None:
            return
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location={'cuda': 'cpu'})
            model_dict = self.state_dict()
            for k, v in list(pretrained_dict.items()):
                if str.find(k, 'last_layer') != -1:
                    pretrained_dict.pop(k)
                if str.find(k, 'conv1') != -1:
                    pretrained_dict.pop(k)
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))


class UpSampleBlock_new2(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = in_ch // 2
        last_ch = mid_ch // 2
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, mid_ch, 3, stride=2, padding=1, output_padding=1, bias=False),      
            nn.BatchNorm2d(mid_ch),
            ResBlock(mid_ch, mid_ch),
            ResBlock(mid_ch, mid_ch),
            ResBlock(mid_ch, mid_ch),
        )
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, last_ch, 3, stride=2, padding=1, output_padding=1, bias=False),      
            nn.BatchNorm2d(last_ch),
            ResBlock(last_ch, last_ch),
            ResBlock(last_ch, last_ch),
            ResBlock(last_ch, last_ch),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(last_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
        )
        self.conv1.apply(self.init_conv_kaiming)
        self.conv2.apply(self.init_conv_kaiming)
        self.conv3.apply(self.init_conv_kaiming)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3

    def init_conv_kaiming(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class UpSampleBlock_new3(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = in_ch // 2
        last_ch = mid_ch // 2
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, mid_ch, 3, stride=2, padding=1, output_padding=1, bias=False),      
            nn.BatchNorm2d(mid_ch),
            CBAM(mid_ch, ratio=4, kernel_size=3),
            ResBlock(mid_ch, mid_ch),
            ResBlock(mid_ch, mid_ch),
            ResBlock(mid_ch, mid_ch),
        )
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, last_ch, 3, stride=2, padding=1, output_padding=1, bias=False),      
            nn.BatchNorm2d(last_ch),
            CBAM(last_ch, ratio=4, kernel_size=3),
            ResBlock(last_ch, last_ch),
            ResBlock(last_ch, last_ch),
            ResBlock(last_ch, last_ch),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(last_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
        )
        self.conv1.apply(self.init_conv_kaiming)
        self.conv2.apply(self.init_conv_kaiming)
        self.conv3.apply(self.init_conv_kaiming)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3

    def init_conv_kaiming(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
