import argparse

import datasets
import torch
from train import fit

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

dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                 sampler=None, batch_sampler=None, num_workers=args.num_workers)
checkoint_dir = 'runs'
epoch, model, optimizer, lr_scheduler, best_score = restore_checkpoint(checkoint_dir, args.continue_training)
#model = torch.nn.DataParallel(model)
fit(epoch, model, detection_loss, optimizer, lr_scheduler, best_score, args.batches_before_train, checkoint_dir, dl, None)