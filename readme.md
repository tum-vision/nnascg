# NNasCG: neural networks as chain graphs

by *Yuesong Shen*

This is the source code for experiments in paper ["Deriving Neural Network Design and Learning from the Probabilistic Framework of Chain Graphs"](https://arxiv.org/pdf/2006.16856.pdf) by Yuesong Shen and Daniel Cremers.

This source code is released under the GPL v3 license. A part of the code is derived from the source code from <https://github.com/akamaster/pytorch_resnet_cifar10> which is released under the BSD license.

If you find our implementation useful for your research, please consider citing our paper ([Arxiv page](https://arxiv.org/abs/2006.16856)):

```
@misc{shen2020deriving,
    title={Deriving Neural Network Design and Learning from the Probabilistic Framework of Chain Graphs},
    author={Yuesong Shen and Daniel Cremers},
    year={2020},
    eprint={2006.16856},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

Enjoy ;)

---

Package structure:

This code package has two independent folders:

- `mnist_simple/`:
    This corresponds to the source code for the "Simple dense network" experiments.

- `cifar_resnet/`:
    This corresponds to the source code for the "Convolutional residual network" experiments. It is based on the source code from <https://github.com/akamaster/pytorch_resnet_cifar10>.

---

Dependencies:

- python (tested on Python 3.7)
- pytorch (tested on PyTorch 1.4.0)
- torchvision (tested on torchvision 0.5.0)

---

Usage of code in `mnist_simple/`:

Run `python run_mnist.py --help` to get the list of command line arguments and their descriptions.

Example usages:

- Train a baseline with Tanh activation on MNIST, no dropout, no PCFF:
  `python run_mnist.py -d MNIST -a tanh`

- Train a dropout baseline with ReLU activation on FashionMNIST with drop rate 0.5, no PCFF, use GPU (cuda):
  `python run_mnist.py -d FashionMNIST -a relu -p 0.5 -c`

- Train a PCFF model with ReLU activation on MNIST with sample rate 1.0, no dropout, use GPU (cuda):
  `python run_mnist.py -d MNIST -a relu -s 1.0 -c`

- Evaluate a pretrained model stored at "models/mymodel.pickle" with MNIST dataset, Tanh activation and PCFF sample rate 0.4:
  `python run_mnist.py -d MNIST -a tanh -s 0.4 -r "models/mymodel.pickle"`

--------------------------------------------------------------------------------

Usage of code in `cifar_resnet/`:

To run the full set of experiments, simply run `bash run2.sh` (might take a while to run all trainings one by one).

In case your system doesn't support bash script or you want to customize the training, you can have a look at the `run2.sh` file for example usages, as well as running `python trainer2.py --help` to get the list of command line arguments and their descriptions.
