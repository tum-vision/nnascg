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
import os
import sys
import time
import pathlib
import torch
from torch import Tensor
import torch.nn as nn


def rm(filename: str) -> bool:
    try:
        os.remove(filename)
        return True
    except OSError:
        return False


def mkdir(dirpath: str, parents: bool=False, exist_ok: bool=False) -> None:
    pathlib.Path(dirpath).mkdir(parents=parents, exist_ok=exist_ok)


def mkdirp(dirpath: str) -> None:
    mkdir(dirpath, parents=True, exist_ok=True)


def get_timestamp() -> str:
    """generate time stamp of current time"""
    return time.strftime('%y%m%d%H%M%S')


class Tee(object):
    """imitation of the tee command"""
    def __init__(self, source, destination) -> None:
        self.source = source
        self.destination = destination
        self.flush()

    def write(self, msg: str) -> None:
        self.destination.write(msg)
        self.source.write(msg)

    def flush(self) -> None:
        self.destination.flush()
        self.source.flush()


class Log(object):
    """an easy implementation of logger"""
    _stdout = sys.stdout
    _stderr = sys.stderr

    def __init__(self, path: str) -> None:
        self.path = path
        self.logfile = None

    def start(self, title: str, overwrite: bool = False) -> None:
        self.logfile = open(self.path, 'w' if overwrite else 'a')
        self.logfile.write('\n{0}: starting log entry {1}\n\n'.format(
                get_timestamp(), title))
        self.logfile.flush()

    def write(self, content: str, end='\n') -> str:
        if self.logfile is None:
            raise Exception('Unable to write to closed log file.')
        timestamp = get_timestamp()
        self.logfile.write('{0}: {1}{2}'.format(
                timestamp, content, end))
        self.logfile.flush()
        return timestamp

    def start_intercept(self,
                        target_stdout: bool = True,
                        target_stderr: bool = True,
                        mute_stdout: bool = False,
                        mute_stderr: bool = False) -> None:
        """start logging content from stdout or stderr"""
        if not target_stdout and not target_stderr:
            return
        if self.logfile is None:
            raise Exception('Unable to start with closed log file.')
        if target_stdout:
            sys.stdout = self.logfile if mute_stdout else Tee(sys.stdout,
                                                              self.logfile)
        if target_stderr:
            sys.stderr = self.logfile if mute_stderr else Tee(sys.stderr,
                                                              self.logfile)

    @staticmethod
    def stop_intercept() -> None:
        sys.stdout = Log._stdout
        sys.stderr = Log._stderr

    def close(self) -> None:
        self.stop_intercept()
        if self.logfile is not None:
            self.logfile.close()
            self.logfile = None


class ProgressBar(object):
    """an easy implementation indicating the progress in console"""
    pattern = '....1....2....3....4....5....6....7....8....9....O'
    ptn_len = len(pattern)

    def __init__(self) -> None:
        self.pt = 0
        print('Pattern: [' + ProgressBar.pattern+']')
        print('Progress: ', end='')

    def progress(self, ratio: float) -> None:
        if ratio > 1.0 or ratio < 0.:
            raise ValueError('ratio should be between 0 and 1')
        newpt = int(ProgressBar.ptn_len * ratio)
        if newpt > self.pt:
            print(ProgressBar.pattern[self.pt:newpt], end='')
            sys.stdout.flush()
            self.pt = newpt

    def complete(self) -> None:
        self.progress(1.)
        print()


class EarlyStopper(object):
    def __init__(self, patience: int = 3, should_decrease: bool = True
                 ) -> None:
        self.patience = patience
        self.should_decrease = should_decrease
        self.current = float('Inf') if should_decrease else -float('Inf')
        self.strike = 0

    def update(self, value: float) -> bool:
        if (self.current > value) == self.should_decrease:
            self.strike = 0
            self.current = value
            print('EarlyStopper: Received better result.')
            return True
        else:
            self.strike += 1
            if self.strike > self.patience:
                print('EarlyStopper: Should stop now.')
            else:
                print('EarlyStopper: Strike {} / {}.'.format(self.strike,
                                                             self.patience))
            return False

    def passes(self) -> bool:
        return self.strike <= self.patience


def check_cuda() -> None:
    # check availability
    if not torch.cuda.is_available():
        raise Exception('No CUDA device available')
    # show all available GPUs
    cuda_count = torch.cuda.device_count()
    print('{0} CUDA device(s) available:'.format(cuda_count))
    for i in range(cuda_count):
        print('- {0}: {1} ({2})'.format(i, torch.cuda.get_device_name(i),
                                        torch.cuda.get_device_capability(i)))
    # showing current cuda device
    curr_idx = torch.cuda.current_device()
    print('Currently using device {0}'.format(curr_idx))


def display_param_stats(model: nn.Module) -> None:
    tot = 0
    print('\nParam stats:')
    for n, p in model.named_parameters():
        print(n, 'size:', p.numel(), 'shape:', tuple(p.size()))
        tot += p.numel()
    print('Total params:', tot)


def unzip0(w: Tensor, dim: int=-1) -> Tensor:
    sz = list(w.size())
    sz[dim] = 1
    return torch.cat((w, w.new_zeros(sz)), dim)
