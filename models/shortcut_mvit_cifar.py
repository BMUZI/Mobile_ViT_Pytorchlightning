from models.basic_block_3dim import MobileViTBlock, MV2Block
from models.basic_block_3dim import conv_nxn_bn, conv_1x1_bn
import torch
import torch.nn as nn
import pytorch_lightning as pl

class ShortCutMobileViT_CIFAR(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=2, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[3], kernel_size, patch_size, int(dims[0]*2)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(kernel_size = (16, 16))

        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mvit[0](x)

        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)

        return x