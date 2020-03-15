import argparse
import logging
import os
import time

import torch

from core import logconfig, models
from core.dataloader import get_test_dataset
from core.evaler import eval_model, find_landmarks

logging.config.dictConfig(logconfig.LOGGER_CONFIG)

_LOGGER = logging.getLogger()


def check_args(args):
    if not os.path.exists(args.val_img_dir):
        raise ValueError(f"'{args.val_img_dir}' does not exist")

    if not os.path.isdir(args.val_img_dir):
        raise ValueError(f"'{args.val_img_dir}' is not a directory")

    if not os.path.isfile(args.pretrained_weights):
        raise ValueError(f"'{args.pretrained_weights}' is not a file")

    pred_dir_name = os.path.dirname(args.pred)

    if not os.path.exists(pred_dir_name):
        _LOGGER.warning(
            "Parent directory of file with predcitions does not exist. Create it.")
        os.makedirs(pred_dir_name, exist_ok=True)

    name, ext = os.path.splitext(args.pred)

    if ext != ".json":
        raise ValueError(f"'{args.pred}' must have json extension")

    if args.debug_dir is not None:
        os.makedirs(args.debug_dir, exist_ok=True)


def load_weights(path_to_pretrain_weights: str, model):
    _LOGGER.info(f"Load pretrained model from '{path_to_pretrain_weights}'")
    if os.path.exists(path_to_pretrain_weights):
        checkpoint = torch.load(path_to_pretrain_weights)
        if 'state_dict' not in checkpoint:
            model.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items()
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            model.load_state_dict(model_weights)


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
    num_workers = args.num_workers

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _LOGGER.info(f"Available device: {device}")

    dataloader = get_test_dataset(val_img_dir,
                                  batch_size, num_landmarks, num_workers)
    use_gpu = torch.cuda.is_available()
    model_ft = models.FAN(hg_blocks, end_relu, gray_scale, num_landmarks)

    load_weights(pretrained_weights, model_ft)

    model_ft = model_ft.to(device)
    find_landmarks(model=model_ft, dataloader=dataloader, root_img_dir=val_img_dir, path_to_pred=args.pred,
                   num_landmarks=num_landmarks, use_gpu=use_gpu, debug_dir=args.debug_dir)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Dataset paths
    parser.add_argument('--val_img_dir', type=str,
                        help='A path to validation image directory', required=True)
    parser.add_argument('--val_landmarks_dir', type=str, default=None,
                        help='A path to validation landmarks directory.')
    parser.add_argument('--num_landmarks', type=int, default=98,
                        help='Number of landmarks')
    parser.add_argument('--pred', type=str,
                        help='A path to json which will be store predictions', required=True)

    # Checkpoint and pretrained weights
    parser.add_argument('--ckpt_save_path', type=str,
                        help='A directory to save checkpoint file', default=None)
    parser.add_argument('--pretrained_weights', type=str,
                        help='A path to load pretrained_weights', required=True)

    # Eval options
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=1,
                        help="A number of workers to load image from directory")
    parser.add_argument("--debug-dir", type=str, default=None,
                        help="A path to directory where store photo with predicted landmarks")

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
