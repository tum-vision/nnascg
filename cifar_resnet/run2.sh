#!/bin/bash

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

# get current time
ts=$(date +"%Y%m%dT%H%M%S")

# run training for ResNet20 model without stochastic regularization
for model in resnet20
do
    echo "python -u trainer2.py  --arch=$model  --save-dir=save_$model$ts |& tee -a log_$model$ts"
    python -u trainer2.py  --arch=$model  --save-dir=save_$model$ts |& tee -a log_$model$ts
done

# run training for ResNet20 dropout models
for model in resnet20dropout01 resnet20dropout02 resnet20dropout03 resnet20dropout04 resnet20dropout05 resnet20dropout06 resnet20dropout07 resnet20dropout08 resnet20dropout09
do
    echo "python -u trainer2.py  --arch=$model  --save-dir=save_$model$ts |& tee -a log_$model$ts"
    python -u trainer2.py  --arch=$model  --save-dir=save_$model$ts |& tee -a log_$model$ts
done

# run training for ResNet20 PCFF models
for model in resnet20pcff01 resnet20pcff02 resnet20pcff03 resnet20pcff04 resnet20pcff05 resnet20pcff06 resnet20pcff07 resnet20pcff08 resnet20pcff09 resnet20pcff
do
    echo "python -u trainer2.py  --arch=$model  --save-dir=save_$model$ts |& tee -a log_$model$ts"
    python -u trainer2.py  --arch=$model  --save-dir=save_$model$ts |& tee -a log_$model$ts
done
