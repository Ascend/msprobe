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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from config import LR, WEIGHT_DECAY, NON_BLOCKING
from utils import EarlyStopping


def init_optimizer(model):
    return optim.RMSprop(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        momentum=0.9
    )


def init_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for seq, labels in loader:
        seq = seq.to(device, non_blocking=NON_BLOCKING)
        labels = labels.to(device, non_blocking=NON_BLOCKING)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(seq)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * seq.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += seq.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for seq, labels in loader:
            seq = seq.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            with autocast():
                outputs = model(seq)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * seq.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += seq.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc
