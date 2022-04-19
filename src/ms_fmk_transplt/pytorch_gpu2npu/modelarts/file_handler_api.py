#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

FILE_HANDLER_API = {
    # built-in
    'builtins.open': {
        'arg_no': 0,
    },
    # os
    'os.mkdir': {
        'arg_no': 0,
    },
    'os.makedirs': {
        'arg_no': 0,
    },
    'os.listdir': {
        'arg_no': 0,
    },
    'os.path.isfile': {
        'arg_no': 0,
    },
    'os.path.isdir': {
        'arg_no': 0,
    },
    'os.path.exists': {
        'arg_no': 0,
    },
    # shutil
    'shutil.copy': {
        'arg_no': [0, 1],
    },
    'shutil.copyfile': {
        'arg_no': [0, 1],
    },
    'shutil.rmtree': {
        'arg_no': 0,
    },
    # glob
    'glob.glob': {
        'arg_no': 0,
    },
    # logging
    'logging.FileHandler': {
        'arg_no': 0,
    },
    # pathlib
    'pathlib.Path': {
        'arg_no': 0,
    },
    # numpy
    'numpy.load': {
        'arg_no': 0,
    },
    'numpy.loadtxt': {
        'arg_no': 0,
    },
    'numpy.save': {
        'arg_no': 0,
    },
    'numpy.savetxt': {
        'arg_no': 0,
    },
    # PIL
    'PIL.Image.open': {
        'arg_no': 0,
    },
    # matplotlib
    'matplotlib.pyplot.imsave': {
        'arg_no': 0,
    },
    # pandas
    'pandas.read_pickle': {
        'arg_no': 0,
    },
    'pandas.read_csv': {
        'arg_no': 0,
    },
    'pandas.read_json': {
        'arg_no': 0,
    },
    # cv2
    'cv2.imread': {
        'arg_no': 0,
    },
    'cv2.imwrite': {
        'arg_no': 0,
    },
    # matplotlib
    'matplotlib.pyplot.savefig': {
        'arg_no': 0,
    },
    # tensorboardX
    'tensorboardX.SummaryWriter': {
        'arg_no': 0,
    },
    # json
    'json.load': {
        'arg_no': 0,
    },
    # scipy
    'scipy.io.loadmat': {
        'arg_no': 0,
    },
    # mmcv
    'mmcv.imread': {
        'arg_no': 0,
    },
    # tarfile
    'tarfile.open': {
        'arg_no': 0,
    },
    # codecs
    'codecs.open': {
        'arg_no': 0,
    },
    # torch
    'torch.load': {
        'arg_no': 0,
    },
    'torch.save': {
        'arg_no': 1,
    },
    'torch.utils.data.Dataset': {
        'arg_no': 0,
    },
    'torchvision.datasets.MNIST': {
        'arg_no': 0,
    },
    'torchvision.datasets.CIFAR10': {
        'arg_no': 0,
    },
    'torchvision.datasets.CIFAR100': {
        'arg_no': 0,
    },
    'torchvision.datasets.ImageFolder': {
        'arg_no': 0,
    },
    'torchtext.data.TabularDataset': {
        'arg_no': 0,
    }
}
