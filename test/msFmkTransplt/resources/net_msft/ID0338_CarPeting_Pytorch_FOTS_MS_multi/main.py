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

import argparse

import datasets
import torch_npu
import torch
from train import fit
import torch.npu
import os
DEVICE_ID= 0
if os.getenv('DEVICE_ID') and str.isdigit(os.getenv('DEVICE_ID')):
    DEVICE_ID= int(os.getenv('DEVICE_ID'))
if torch.npu.current_device() != DEVICE_ID:
    torch.npu.set_device(f'npu:{DEVICE_ID}')
RANK_SIZE = int(os.getenv('RANK_SIZE'))
RANK_ID = int(os.getenv('RANK_ID'))
torch.distributed.init_process_group('hccl', rank=RANK_ID, world_size=RANK_SIZE)

parser = argparse.ArgumentParser()
parser.add_argument('--train-folder', type=str, required=True, help='Path to folder with train images and labels')
parser.add_argument('--batch-size', type=int, default=21, help='Number of batches to process before train step')
parser.add_argument('--batches-before-train', type=int, default=2, help='Number of batches to process before train step')
parser.add_argument('--num-workers', type=int, default=8, help='Path to folder with train images and labels')
parser.add_argument('--continue-training', action='store_true', help='continue training')
args = parser.parse_args()

#data_set = datasets.SynthText(args.train_folder, datasets.transform)
data_set = datasets.ICDAR2015(args.train_folder, datasets.transform)

# SynthText and ICDAR2015 have different layouts. One will probably need to provide two different paths to train
# on concatination of these two data sets. But the paper doesn't concat them so me neither
# datai_set = torch.utils.data.ConcatDataset((synth, icdar))

dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                                 batch_sampler=None, num_workers=args.num_workers, pin_memory = True, drop_last = True, sampler = torch.utils.data.distributed.DistributedSampler(data_set))
checkoint_dir = 'runs'
epoch, model, optimizer, lr_scheduler, best_score = restore_checkpoint(checkoint_dir, args.continue_training)
model = model.npu()
if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ID], broadcast_buffers=False)
#model = torch.nn.DataParallel(model)
fit(epoch, model, detection_loss, optimizer, lr_scheduler, best_score, args.batches_before_train, checkoint_dir, dl, None)