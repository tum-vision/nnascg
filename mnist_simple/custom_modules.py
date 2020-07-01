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
from torch import Tensor, nn
import torch.nn.functional as nnf


class TanhPCFF(nn.Module):
    def __init__(self, hardness: float):
        super().__init__()
        self.hardness = hardness

    @staticmethod
    def feedforward(t: Tensor) -> Tensor:
        return torch.tanh(t)

    @staticmethod
    def samplforward(t: Tensor) -> Tensor:
        return torch.bernoulli(torch.sigmoid(t * 2)) * 2 - 1

    def forward(self, t: Tensor) -> Tensor:
        hardness = self.hardness
        if self.training and hardness > 0:
            if hardness >= 1:
                return self.samplforward(t)
            else:
                mask = torch.bernoulli(torch.full_like(t, hardness)).bool()
                return torch.where(
                    mask, self.samplforward(t), self.feedforward(t))
        else:
            return self.feedforward(t)


class SigmoidPCFF(nn.Module):
    def __init__(self, hardness: float):
        super().__init__()
        self.hardness = hardness

    @staticmethod
    def feedforward(t: Tensor) -> Tensor:
        return torch.sigmoid(t)

    @staticmethod
    def samplforward(t: Tensor) -> Tensor:
        return torch.bernoulli(torch.sigmoid(t))

    def forward(self, t: Tensor) -> Tensor:
        hardness = self.hardness
        if self.training and hardness > 0:
            if hardness >= 1:
                return self.samplforward(t)
            else:
                mask = torch.bernoulli(torch.full_like(t, hardness)).bool()
                return torch.where(
                    mask, self.samplforward(t), self.feedforward(t))
        else:
            return self.feedforward(t)


class SoftplusPCFF(nn.Module):
    _FACTOR = 1.776091849725427

    def __init__(self, hardness: float):
        super().__init__()
        self.hardness = hardness

    @staticmethod
    def feedforward(t: Tensor) -> Tensor:
        return nnf.softplus(t)

    @staticmethod
    def samplforward(t: Tensor) -> Tensor:
        return torch.relu(torch.randn_like(t) * SoftplusPCFF.factor + t)

    def forward(self, t: Tensor) -> Tensor:
        hardness = self.hardness
        if self.training and hardness > 0:
            if hardness >= 1:
                return self.samplforward(t)
            else:
                mask = torch.bernoulli(torch.full_like(t, hardness)).bool()
                return torch.where(
                    mask, self.samplforward(t), self.feedforward(t))
        else:
            return self.feedforward(t)


class ReluPCFF(nn.Module):
    def __init__(self, hardness: float, c=1.0):
        super().__init__()
        self.hardness = hardness
        self.c = c

    @staticmethod
    def feedforward(t: Tensor) -> Tensor:
        return torch.relu(t)

    @staticmethod
    def samplforward(t: Tensor, c=1.0) -> Tensor:
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
