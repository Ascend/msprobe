#! /usr/bin/env python
# coding=utf-8
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import sys
import argparse
import torch_npu
import torch
import model
import data
from config import Config
parser = argparse.ArgumentParser(description='Evaluate Text CNN classificer')
parser.add_argument(
    '-model',
    type=str,
    default="model/textcnn.model",
    help='filename of pre-trained model [model/textcnn.model]')

if __name__ == '__main__':
    conf = Config()
    args = parser.parse_args()

    print("Loading data...")
    test_iter, text_field, label_field = data.fasttext_dataloader(
        "data/test.txt", conf.batch_size, shuffle=False)

    # model
    if os.path.exists(args.model):
        print('Loading model from {}...'.format(args.model))
        cnn = torch.load(args.model)
    else:
        print("Model doesn't exist.")
        sys.exit(-1)

    text_field.vocab = data.load_vocab("model/text.vocab")
    label_field.vocab = data.load_vocab("model/label.vocab")

    print(cnn)

    try:
        model.eval(test_iter, cnn, conf)
    except Exception as e:
        print("Sorry. The test dataset doesn't exist.")
