###############################################################################
# This file is part of NNasCG, the source code for experiments in paper
# "Deriving Neural Network Design and Learning from the Probabilistic Framework
# of Chain Graphs" by Yuesong Shen and Daniel Cremers.
#
# Copyright 2020 Technical University of Munich
#
# Developed by Yuesong Shen <yuesong dot shen at tum dot de>.
#
# If you use this file for your research, please cite the aforementioned paper.
#
# NNasCG is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NNasCG is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NNasCG. If not, see <http://www.gnu.org/licenses/>.
###############################################################################
import torch
from torch import nn, Tensor
import torch.nn.functional as nnf


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: Tensor) -> Tensor:
        return nnf.avg_pool2d(t, t.size()[2:])


class ReluPCFF(nn.Module):
    def __init__(self, hardness: float, c: float = 1.0):
        super().__init__()
        self.hardness = hardness
        self.c = c

    @staticmethod
    def feedforward(t: Tensor) -> Tensor:
        return torch.relu(t)

    @staticmethod
    def samplforward(t: Tensor, c: float = 1.0) -> Tensor:
        return torch.relu(torch.randn_like(t) * torch.tanh(t) * c + t)

    def forward(self, t: Tensor) -> Tensor:
        hardness = self.hardness
        if self.training and hardness > 0:
            if hardness >= 1:
                return self.samplforward(t, self.c)
            else:
                mask = torch.bernoulli(torch.full_like(t, hardness)).bool()
                return torch.where(
                    mask, self.samplforward(t, self.c), self.feedforward(t))
        else:
            return self.feedforward(t)


class BNRelu(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, t: Tensor) -> Tensor:
        return torch.relu(self.bn(t))


class BNReluPCFF(nn.Module):
    def __init__(self, channel: int, hardness: float = 1.0, c: float = 1.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(channel)
        self.activ = ReluPCFF(hardness, c)

    def forward(self, t: Tensor) -> Tensor:
        return self.activ(self.bn(t))


class BNReluDropout(nn.Module):
    def __init__(self, channel: int, d: float):
        super().__init__()
        self.bn = nn.BatchNorm2d(channel)
        self.dropout = nn.Dropout(d)

    def forward(self, t: Tensor) -> Tensor:
        return self.dropout(torch.relu(self.bn(t)))


def get_bnrelu(channel: int):
    return BNRelu(channel)


def get_bnrelupcff(channel: int):
    return BNReluPCFF(channel)


def get_bnrelupcff09(channel: int):
    return BNReluPCFF(channel, hardness=0.9)


def get_bnrelupcff08(channel: int):
    return BNReluPCFF(channel, hardness=0.8)


def get_bnrelupcff07(channel: int):
    return BNReluPCFF(channel, hardness=0.7)


def get_bnrelupcff06(channel: int):
    return BNReluPCFF(channel, hardness=0.6)


def get_bnrelupcff05(channel: int):
    return BNReluPCFF(channel, hardness=0.5)


def get_bnrelupcff04(channel: int):
    return BNReluPCFF(channel, hardness=0.4)


def get_bnrelupcff03(channel: int):
    return BNReluPCFF(channel, hardness=0.3)


def get_bnrelupcff02(channel: int):
    return BNReluPCFF(channel, hardness=0.2)


def get_bnrelupcff01(channel: int):
    return BNReluPCFF(channel, hardness=0.1)


def get_bnreludropout09(channel: int):
    return BNReluDropout(channel, 0.9)


def get_bnreludropout08(channel: int):
    return BNReluDropout(channel, 0.8)


def get_bnreludropout07(channel: int):
    return BNReluDropout(channel, 0.7)


def get_bnreludropout06(channel: int):
    return BNReluDropout(channel, 0.6)


def get_bnreludropout05(channel: int):
    return BNReluDropout(channel, 0.5)


def get_bnreludropout04(channel: int):
    return BNReluDropout(channel, 0.4)


def get_bnreludropout03(channel: int):
    return BNReluDropout(channel, 0.3)


def get_bnreludropout02(channel: int):
    return BNReluDropout(channel, 0.2)


def get_bnreludropout01(channel: int):
    return BNReluDropout(channel, 0.1)


class BaseBlock(nn.Module):
    def __init__(self, inchannel: int, outchannel: int, stride: int = 1,
                 get_activ=get_bnrelu) -> None:
        super().__init__()
        self.activ_h = get_activ(outchannel)
        self.conv_io = nn.Conv2d(inchannel, outchannel, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        self.conv_ih = nn.Conv2d(inchannel, outchannel, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        self.conv_ho = nn.Conv2d(outchannel, outchannel, kernel_size=3,
                                 padding=1, bias=False)

    def forward(self, t: Tensor) -> Tensor:
        t_out = self.conv_io(t)
        t_out = t_out + self.conv_ho(self.activ_h(self.conv_ih(t)))
        return t_out


class RefineBlock(nn.Module):
    def __init__(self, channel: int, get_activ=get_bnrelu) -> None:
        super().__init__()
        self.activ_i = get_activ(channel)
        self.activ_h = get_activ(channel)
        self.conv_ih = nn.Conv2d(
            channel, channel, kernel_size=3, padding=1, bias=False)
        self.conv_ho = nn.Conv2d(
            channel, channel, kernel_size=3, padding=1, bias=False)

    def forward(self, t: Tensor) -> Tensor:
        t = t + self.conv_ho(self.activ_h(self.conv_ih(self.activ_i(t))))
        return t


def resnet20(get_activ=get_bnrelu):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff(get_activ=get_bnrelupcff):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff09(get_activ=get_bnrelupcff09):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff08(get_activ=get_bnrelupcff08):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff07(get_activ=get_bnrelupcff07):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff06(get_activ=get_bnrelupcff06):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff05(get_activ=get_bnrelupcff05):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff04(get_activ=get_bnrelupcff04):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff03(get_activ=get_bnrelupcff03):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff02(get_activ=get_bnrelupcff02):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20pcff01(get_activ=get_bnrelupcff01):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout09(get_activ=get_bnreludropout09):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout08(get_activ=get_bnreludropout08):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout07(get_activ=get_bnreludropout07):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout06(get_activ=get_bnreludropout06):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout05(get_activ=get_bnreludropout05):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout04(get_activ=get_bnreludropout04):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout03(get_activ=get_bnreludropout03):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout02(get_activ=get_bnreludropout02):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )


def resnet20dropout01(get_activ=get_bnreludropout01):
    c_in, c_h1, c_h2, c_h3, c_out = 3, 16, 32, 64, 10
    return nn.Sequential(
        nn.Conv2d(c_in, c_h1, kernel_size=3, padding=1, bias=False),
        * [RefineBlock(c_h1, get_activ) for _ in range(3)],
        get_activ(c_h1),
        BaseBlock(c_h1, c_h2, 2, get_activ),
        *[RefineBlock(c_h2, get_activ) for _ in range(2)],
        get_activ(c_h2),
        BaseBlock(c_h2, c_h3, 2, get_activ),
        *[RefineBlock(c_h3, get_activ) for _ in range(2)],
        get_activ(c_h3),
        GlobalAvgPool(),
        nn.Flatten(),
        nn.Linear(c_h3, c_out)
    )
