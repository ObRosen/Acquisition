from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from reversi_config import ResNetParamSection


class CircPadding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return F.pad(x, (1, 0, 1, 0), mode="circular")


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        if in_channels != out_channels:
            self.f1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.f1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            self.short_cut = nn.Sequential()

    def forward(self, x: torch.Tensor):
        out = self.f1(x)
        out = F.relu(out + self.short_cut(x))
        return out


class ResNet(nn.Module):
    # don't call `__init__` if loaded from disk!
    def __init__(self, paramSection: "ResNetParamSection") -> None:
        super().__init__()

        self.construct(paramSection.TotalResblockNum,
                       paramSection.InitialChannelNum)

        # ensure that the model can be processed by pytorch
        self.addLayerToModule()

    def construct(self, total_res_block_num, default_channels):
        self.initConv = nn.Sequential(
            CircPadding(),
            nn.Conv2d(in_channels=2, out_channels=default_channels, kernel_size=2,
                      stride=1, bias=False),
            nn.BatchNorm2d(default_channels),
            nn.ReLU(inplace=True),
        )
        self.layers: List[ResBlock] = []

        for _ in range(total_res_block_num):
            self.make_layer(ResBlock, default_channels, 1)

        self.pConv = nn.Sequential(
            nn.Conv2d(in_channels=default_channels, out_channels=2,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.fc_p = nn.Linear(2*8*8, 64)

        self.vConv = nn.Sequential(
            nn.Conv2d(in_channels=default_channels, out_channels=1,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.fc_v1 = nn.Linear(1*8*8, 8)
        self.fc_v2 = nn.Linear(8, 1)

    def make_layer(self, block, channels: int, num_blocks: int):
        for _ in range(num_blocks):
            self.layers.append(block(channels, channels))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # print(x)
        out = self.initConv(x)
        # print('1 out: {}'.format(out))
        # print('1 out shape: {}'.format(out.shape))
        for layer in self.layers:
            out = layer(out)
            # print('2 out: {}'.format(out))
            # print('2 out shape: {}'.format(out.shape))
        p: Tensor = self.pConv(out)
        p = p.view(p.size(0), -1)
        p = self.fc_p(p)
        p = F.softmax(p, dim=1)
        v: Tensor = self.vConv(out)
        v = v.view(v.size(0), -1)
        v = self.fc_v1(v)
        v = F.relu(v)
        v = self.fc_v2(v)
        v = F.tanh(v)
        return p, v

    def addToModule(self, i, layer):
        self._modules[str(i)] = layer

    def addLayerToModule(self):
        for i, layer in enumerate(self.layers):
            self.addToModule(i, layer)
