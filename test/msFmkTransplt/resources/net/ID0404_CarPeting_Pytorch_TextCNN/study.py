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

import data
import torchtext


text_field = data.FastTextTEXT
label_field = data.FastTextLABEL
train_data = data.FastTextDataset("data/train.txt", text_field, label_field)


train_iter = torchtext.data.Iterator(
    train_data, batch_size=1, shuffle=False, repeat=False)

print(train_data[0].text)
print(train_data[0].label)

text_field.build_vocab(train_data)
label_field.build_vocab(train_data)
# data.save_vocab(label_field.vocab, "label.vocab")

# label_field.vocab = data.load_vocab("label.vocab")
# print(label_field.vocab.freqs.most_common(10))


print(label_field.vocab.freqs.most_common(100))
for w in train_data[0].text:
    print(w, text_field.vocab.stoi[w])

i = 0
for batch in train_iter:
    feature, target = batch.text, batch.label

    # feature.data.t_(), target.data.sub_(1)  # batch first, index align
    # print("feature: ", feature, feature.size())
    # print("target: ", target, target.size())
    i += 1
    if i > 2000:
        break
print(i)
