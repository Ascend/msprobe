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

import pickle
import torch_npu

import torchtext


def text_token(x):
    return [w for w in x.split(" ") if len(w) > 0]


FastTextTEXT = torchtext.data.Field(sequential=True, tokenize=text_token, lower=True)


def label_token(x):
    return [x.replace("__label__", "")]


FastTextLABEL = torchtext.data.Field(sequential=False, tokenize=label_token, lower=True)


class FastTextDataset(torchtext.data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, sep='\t', **kwargs):
        """Create an dataset instance given a path and fields.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """

        fields = [('text', text_field), ('label', label_field)]
        examples = []
        with open(path, errors='ignore') as f:
            for line in f:
                s = line.strip().split(sep)
                if len(s) != 2:
                    continue

                text, label = s[0], s[1]
                label = label.replace("__label__", "")
                e = torchtext.data.Example()
                setattr(e, "text", text_field.preprocess(text))
                setattr(e, "label", label_field.preprocess(label))
                examples.append(e)

        super(FastTextDataset, self).__init__(examples, fields, **kwargs)


def fasttext_dataloader(datafile, batchsize, shuffle=False):
    text_field = FastTextTEXT
    label_field = FastTextLABEL

    dataset = FastTextDataset(datafile, text_field, label_field)
    text_field.build_vocab(dataset)
    label_field.build_vocab(dataset)

    dataiter = torchtext.data.Iterator(dataset, batchsize, shuffle, repeat=False)
    # dataiter.init_epoch()

    return dataiter, text_field, label_field


def save_vocab(vocab, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
