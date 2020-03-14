import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from core import models
from core.dataloader import get_dataset
from core.evaler import eval_model

def check_args(args):
    if not os.path.exists(args.val_img_dir):
        raise ValueError(f"'{args.val_img_dir}' does not exist")

    if not os.path.isdir(args.val_img_dir):
        raise ValueError(f"'{args.val_img_dir}' is not a directory")

    if not os.path.isfile(args.pretrained_weights):
        raise ValueError(f"'{args.pretrained_weights}' is not a file")


def main(args):
    val_img_dir = args.val_img_dir
    val_landmarks_dir = args.val_landmarks_dir
    ckpt_save_path = args.ckpt_save_path
    batch_size = args.batch_size
    pretrained_weights = args.pretrained_weights
    gray_scale = args.gray_scale
    hg_blocks = args.hg_blocks
    end_relu = args.end_relu 
    num_landmarks = args.num_landmarks

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Available device: {device}")

    writer = SummaryWriter(ckpt_save_path)
    dataloaders, dataset_sizes = get_dataset(val_img_dir, val_landmarks_dir,
                                            batch_size, num_landmarks)
    use_gpu = torch.cuda.is_available()
    model_ft = models.FAN(hg_blocks, end_relu, gray_scale, num_landmarks)

    if os.path.exists(pretrained_weights):
        checkpoint = torch.load(pretrained_weights)
        if 'state_dict' not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)

    model_ft = model_ft.to(device)
    model_ft = eval_model(model_ft, dataloaders, dataset_sizes, writer, use_gpu, 1, 'val', ckpt_save_path, num_landmarks)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Dataset paths
    parser.add_argument('--val_img_dir', type=str,
                        help='A path to validation image directory', required=True)
    parser.add_argument('--val_landmarks_dir', type=str,
                        help='A path to validation landmarks directory')
    parser.add_argument('--num_landmarks', type=int, default=98,
                        help='Number of landmarks')

    # Checkpoint and pretrained weights
    parser.add_argument('--ckpt_save_path', type=str,
                        help='A directory to save checkpoint file', required=True)
    parser.add_argument('--pretrained_weights', type=str,
                        help='A path to load pretrained_weights', required=True)

    # Eval options
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=1, help="A number of workers to load image from directory")

    # Network parameters
    parser.add_argument('--hg_blocks', type=int, default=4,
                        help='Number of HG blocks to stack')
    parser.add_argument('--gray_scale', action="store_true",
                        help='Whether to convert RGB image into gray scale during training')
    parser.add_argument('--end_relu', action="store_true",
                        help='Whether to add relu at the end of each HG module')

    args = parser.parse_args()

    check_args(args)
    main(args)
