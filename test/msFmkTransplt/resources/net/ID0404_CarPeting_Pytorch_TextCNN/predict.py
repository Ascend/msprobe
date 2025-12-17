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
import torch
import model
import data
from config import Config

parser = argparse.ArgumentParser(description='Text CNN classificer Predictor')
# model
parser.add_argument(
    '-model',
    type=str,
    default="model/textcnn.model",
    help='file name of the pre-trained model[model/textcnn.model]')
parser.add_argument(
    'predict', type=str, default=None, help='predict the sentence')

if __name__ == '__main__':
    conf = Config()
    args = parser.parse_args()

    text_field = data.FastTextTEXT
    label_field = data.FastTextLABEL

    text_field.vocab = data.load_vocab("model/text.vocab")
    label_field.vocab = data.load_vocab("model/label.vocab")

    # model
    if os.path.exists(args.model):
        print('Loading model from {}...'.format(args.model))
        cnn = torch.load(args.model)
    else:
        print("Model doesn't exist.")
        sys.exit(-1)

    if args.predict is not None:
        label = model.predict(args.predict, cnn, text_field, label_field, conf.cuda)
        print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
