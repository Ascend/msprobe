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
from torch.utils.data import DataLoader
from config import BATCH_SIZE, VAL_BATCH_SIZE, NUM_EPOCHS, PATIENCE, SAVE_PATH, PIN_MEMORY
from dataset import SyntheticTemporalDataset
from model import TemporalClassifier
from utils import set_random_seed, init_cuda_device, EarlyStopping, clear_cuda_cache
from train.trainer import init_optimizer, init_scheduler, train_one_epoch, validate
from torch.cuda.amp import GradScaler


def main():
    set_random_seed()
    device = init_cuda_device()

    train_dataset = SyntheticTemporalDataset(is_train=True)
    val_dataset = SyntheticTemporalDataset(is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=PIN_MEMORY,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        num_workers=0
    )

    model = TemporalClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = init_optimizer(model)
    scheduler = init_scheduler(optimizer)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    print("\n=== Starting Training ===")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        early_stopping(val_acc, model, optimizer, SAVE_PATH)
        if early_stopping.early_stop:
            print("Early stopping triggered - stopping training")
            break

    print("\n=== Final Evaluation ===")
    checkpoint = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_val_loss, final_val_acc = validate(model, val_loader, criterion, device)
    print(f"Final Val Loss: {final_val_loss:.4f} | Final Val Acc: {final_val_acc:.4f}")

    clear_cuda_cache()


if __name__ == "__main__":
    main()
