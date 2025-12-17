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
import argparse
import torch_npu

import torch
import model
import data
from config import Config

parser = argparse.ArgumentParser(description='Train Text CNN classificer')
parser.add_argument(
    '-model',
    type=str,
    default="model/textcnn.model",
    help='file name of pre-trained model [model/textcnn.model]')


if __name__ == '__main__':
    conf = Config()
    conf.dump()
    args = parser.parse_args()

    if not os.path.isdir("logs"):
        os.mkdir("logs")
    if not os.path.isdir("model"):
        os.mkdir("model")

    print("Loading data...")
    train_iter, text_field, label_field = data.fasttext_dataloader(
        "data/train.txt", conf.batch_size)
    data.save_vocab(text_field.vocab, "model/text.vocab")
    data.save_vocab(label_field.vocab, "model/label.vocab")

    # Update configurations
    conf.embed_num = len(text_field.vocab)
    conf.class_num = len(label_field.vocab) - 1
    conf.kernel_sizes = [int(k) for k in conf.kernel_sizes.split(',')]

    # model
    if os.path.exists(args.model):
        print('Loading model from {}...'.format(args.model))
        cnn = torch.load(args.model)
    else:
        cnn = model.TextCNN(conf)

    print(cnn)
    try:
        model.train(train_iter, cnn, conf)
    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early')
