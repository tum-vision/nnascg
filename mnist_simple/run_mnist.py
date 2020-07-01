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
from typing import Tuple, Optional
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as nnf
from torch import Tensor, nn
from utils import (get_timestamp, Log, check_cuda, mkdirp, display_param_stats,
                   rm, EarlyStopper, unzip0)
from mnist_data import get_mnist_dataloaders, AVAILABLE_FLAVORS
from train_bp import (train_epoch, eval_epoch, test,
                      get_param_config_no_bias_decay)
from custom_modules import ReluPCFF, TanhPCFF


ACTIVATIONS = {
    'relu': (nn.ReLU, ReluPCFF),
    'tanh': (nn.Tanh,  TanhPCFF)
}


class WarpPCFF(object):
    def __init__(self, activname, samplerate):
        self.samplerate = samplerate
        self.activname = activname

    def __call__(self):
        samplerate = self.samplerate
        activname = self.activname
        if samplerate is None or samplerate <= 0.0:
            return ACTIVATIONS[activname][0]()
        else:
            return ACTIVATIONS[activname][1](samplerate)


def range01(head):
    def _range01(v):
        err = argparse.ArgumentTypeError(
            f"{head} should be a float between 0 and 1, received {v}")
        try:
            fv = float(v)
        except ValueError:
            raise err
        if not 0 <= fv <= 1:
            raise err
        return fv

    return _range01


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="MNIST",
                        choices=AVAILABLE_FLAVORS, help="dataset to test on")
    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=ACTIVATIONS.keys(), help="activation function")
    parser.add_argument("-p", "--droprate", type=range01("droprate"),
                        default=0, help="dropout drop rate")
    parser.add_argument("-s", "--samplerate", type=range01("samplerate"),
                        default=0, help="PCFF sample rate")
    parser.add_argument(
        "-r", "--resume", type=str,
        help="resume pretrained model for evaluation, skips training")
    parser.add_argument("-c", "--cuda", action="store_true",
                        help="if specified, use gpu")
    return parser.parse_args()


class Dense(nn.Sequential):
    def __init__(self, nhidden: int, hsize: int, amod, p: float = 0.0):
        assert nhidden >= 1
        p = 0.0 if p is None else p
        super().__init__()
        self.nhidden = nhidden
        self.hsize = hsize
        self.amod = amod
        self.p = p
        self.h0 = nn.Sequential(nn.Flatten(), nn.Linear(28*28, hsize), amod())
        for i in range(1, nhidden):
            if p > 0.0:
                self.add_module(f'h{i}', nn.Sequential(
                    nn.Dropout(p), nn.Linear(hsize, hsize), amod()))
            else:
                self.add_module(f'h{i}', nn.Sequential(
                    nn.Linear(hsize, hsize), amod()))
        if p > 0.0:
            self.o = nn.Sequential(nn.Dropout(p), nn.Linear(hsize, 9))
        else:
            self.o = nn.Linear(hsize, 9)


def training_backup(net: Dense, optimizer: optim.Optimizer, path: str,
                    optim_kwargs=None) -> None:
    if optim_kwargs is None:
        optim_kwargs = {}
    dic = {'state_dict': net.state_dict(),
           'params': (
               net.nhidden, net.hsize, net.amod, net.p),
           'optim_type': optimizer.__class__.__name__,
           'optim_state_dict': optimizer.state_dict(),
           'optim_kwargs': optim_kwargs}
    torch.save(dic, path)


def training_resume(path: str, use_cuda: bool, get_params_fn=None
                    ) -> Tuple[Dense, optim.Optimizer]:
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    dic = torch.load(path, map_location=device)
    net = Dense(*dic['params'][:4])
    net.load_state_dict(dic['state_dict'])
    net.cuda() if use_cuda else net.cpu()
    if get_params_fn is None:
        params = net.parameters()
    else:
        params = get_params_fn(net)
    optimizer = optim.__dict__[dic['optim_type']](
        params, **dic['optim_kwargs'])
    optimizer.load_state_dict(dic['optim_state_dict'])
    return net, optimizer


if __name__ == '__main__':
    timestamp_run = get_timestamp()
    parser = parse_args()

    # training params
    use_cuda = parser.cuda
    train_val_split = 0.8
    train_batch = 32
    val_batch = 32
    test_batch = 32

    # earlystopper params
    epochs = None  # int or None: set to None to enjoy early stopping
    patience = 20

    # optimizer and scheduler params
    optim_type = 'SGD'
    optim_kwargs = {'lr': 1e-2, 'momentum': 0.9}
    lrreducefactor = 0.1
    scpatience = 10

    def get_model():
        samplerate = parser.samplerate
        activname = parser.activation
        droprate = parser.droprate
        return Dense(2, 1024, WarpPCFF(activname, samplerate), droprate)

    # dataset and save directories
    dataset_flavor = parser.dataset
    data_dir = {'MNIST': 'data/MNIST/',
                'FashionMNIST': 'data/FashionMNIST/'}[dataset_flavor]
    base_dir = __file__[:-3] + '/'
    save_dir = base_dir + 'model/'
    log_dir = base_dir + 'log/'

    # checks for resuming pretrained models
    resume_from = parser.resume  # save_dir + 'xxx.pickle'
    if resume_from:  # skips training
        epochs = 0

    # create dirs if not there already
    mkdirp(data_dir)
    mkdirp(save_dir)
    mkdirp(log_dir)

    # prepare output ops and loss function
    lossfunc = nnf.cross_entropy

    def output_ops(out: Tuple[Tensor, ...]) -> Tensor:
        return unzip0(out, 1)

    # start logger
    def get_model_name():
        samplerate = parser.samplerate
        activname = parser.activation
        droprate = parser.droprate
        pcff_str = '' if samplerate <= 0.0 else f'pcff_s{samplerate}'
        drop_str = '' if droprate <= 0.0 else f'_p{droprate}'
        return f'dense_2_1024_{activname}{pcff_str}{drop_str}'.replace('.', '')

    log_file = 'drop_{}_{}_{}_{}.log'.format(
        get_model_name(), '-'.join([
            str(i) for i in (train_batch, val_batch, test_batch)]),
        dataset_flavor, timestamp_run)
    log_title = log_file[:-4]
    logger = Log(log_dir + log_file)
    logger.start(log_title)
    logger.start_intercept()

    # check cuda availablility when needed
    if use_cuda:
        check_cuda()

    # set up mnist dataset image size (c, h, w) = (1, 28, 28)
    if dataset_flavor in AVAILABLE_FLAVORS:
        ((train_loader, val_loader, test_loader),
         (nb_train, nb_val, nb_test)) = get_mnist_dataloaders(
            data_dir, train_batch, val_batch, test_batch,
            train_val_split, use_cuda, dataset_flavor,
            keep_shape=False)
    else:
        raise Exception('Unknown dataset: {}'.format(dataset_flavor))
    print('dataset: {}, location: {}'.format(dataset_flavor, data_dir))
    print('sample / batch number for training:  ',
          nb_train, len(train_loader))
    print('sample / batch number for validation:',
          nb_val, len(val_loader))
    print('sample / batch number for testing:   ',
          nb_test, len(test_loader))
    print(f'train / val / test batchsizes: '
          f'{train_batch} / {val_batch} / {test_batch}')
    print(f'{optim_type} {optim_kwargs} {lrreducefactor} {scpatience}')
    print(f'')

    # load the model and optimizer
    if resume_from is None or not os.path.exists(resume_from):
        print('initializing model ...')
        # set up model
        net = get_model()
        if use_cuda:
            net.cuda()
        # set up optimizer
        optimizer = optim.__dict__[optim_type](
            get_param_config_no_bias_decay(net), **optim_kwargs)
    else:
        print('Resume training from {0} ...'.format(resume_from))
        net, optimizer = training_resume(
            resume_from, use_cuda, get_param_config_no_bias_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=lrreducefactor, patience=scpatience)
    display_param_stats(net)

    # training part
    def update_backup(backup: Optional[str], i: int, time_stamp: str) -> str:
        tmp = save_dir + '{0}_{1}_{2}.pickle'.format(log_title, i, time_stamp)
        training_backup(net, optimizer, tmp, optim_kwargs)
        if backup is not None:
            if not rm(backup):
                print('Failed to delete {0}'.format(backup))
        return tmp

    def do_train_epoch(i: int) -> Tuple[float, float, str]:
        train_epoch(
            net, optimizer, train_loader, i, use_cuda, loss_func=lossfunc,
            log_interval=100, output_ops=output_ops)
        time_stamp = get_timestamp()
        avg_loss, acc = eval_epoch(net, val_loader, i, use_cuda,
                                   loss_func=lossfunc, output_ops=output_ops)
        scheduler.step(avg_loss)
        return avg_loss, acc, time_stamp

    backup = None
    if epochs is None:  # use early stopping and backup only the best one
        i = 0
        earlystop = EarlyStopper(patience=patience, should_decrease=True)
        while earlystop.passes():
            avg_loss, acc, time_stamp = do_train_epoch(i)
            isbest = earlystop.update(avg_loss)
            if isbest:
                backup = update_backup(backup, i, time_stamp)
            i += 1
        # revert to the best one for testing
        net, _ = training_resume(
            backup, use_cuda, get_param_config_no_bias_decay)
    else:  # learning with fixed epochs and backup weights for each epoch
        for i in range(epochs):
            _, time_stamp = do_train_epoch(i)
            backup = update_backup(None, i, time_stamp)

    # testing part
    test(net, test_loader, use_cuda, loss_func=lossfunc, output_ops=output_ops)

    # stop logger
    logger.close()
