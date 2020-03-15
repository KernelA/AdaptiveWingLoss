import copy
import math
import os
import sys
import time
import logging
import json
import pathlib


import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import skimage
from skimage import io, draw

from utils.utils import fan_NME, get_preds_fromhm, show_landmarks, resize_landmarks, show_pred_landmarks, convert_to_labels

matplotlib.use('Agg')

try:
    ski_disk_draw = draw.disk
except AttributeError:
    def ski_disk_draw(point, r, *, shape=None):
        return draw.circle(point[0], point[1], r, shape)


def find_landmarks(model, dataloader: torch.utils.data.DataLoader, root_img_dir: str, path_to_pred: str, num_landmarks: int = 98, use_gpu: bool = True, debug_dir: str = None):
    logger = logging.getLogger()
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    landmarks = []
    total = len(dataloader.dataset)
    progress = 0

    if debug_dir is not None:
        debug_dir = pathlib.Path(debug_dir)

    with torch.no_grad():
        for batch_num, img_info in enumerate(dataloader, 1):
            logger.info(f"Loaded batch {batch_num} from {len(dataloader)}")
            inputs = img_info["image"]

            if use_gpu:
                inputs = inputs.to(device)

            outputs, boundary_channels = model(inputs)
            # Swap axes to convert tensor B x C x H x W to  B x H x W x C
            images = inputs.cpu().numpy().transpose((0, 2, 3, 1))

            for i, path in zip(range(inputs.shape[0]), map(pathlib.Path, img_info["path"])):
                pred_heatmap = outputs[-1][i, :-1, ...].cpu()
                heatmap_height, heatmap_width = pred_heatmap.shape[1:]
                img_height, img_width = images[i].shape[0:2]
                pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
                pred_landmarks = pred_landmarks.squeeze().numpy()

                pred_landmarks = np.rint(pred_landmarks)
                pred_landmarks = np.clip(
                    pred_landmarks, (0, 0), (img_width - 1, img_height - 1)).astype(np.int32)
                labels = convert_to_labels(pred_landmarks)
                rel_path = path.relative_to(root_img_dir)
                landmarks.append({"path": str(rel_path), "heat_map_width": heatmap_width,
                                  "heat_map_height": heatmap_height, "labels": labels})

                if debug_dir is not None:
                    resized_landmarks = resize_landmarks(
                        pred_landmarks, img_width / heatmap_width, img_height / heatmap_height)
                    dir_name = debug_dir / rel_path.parent
                    dir_name.mkdir(parents=True, exist_ok=True)
                    image_with_landmarks = skimage.img_as_ubyte(images[i])

                    green_color = (0, 255, 0)
                    for point in resized_landmarks:
                        rr, cc = ski_disk_draw(
                            tuple(point[::-1]), 2, shape=image_with_landmarks.shape)
                        image_with_landmarks[rr, cc] = green_color

                    img_with_landmarks_name = "landmarks_" + rel_path.name
                    path_to_image = dir_name / img_with_landmarks_name
                    io.imsave(str(path_to_image), image_with_landmarks)
                    logger.debug(
                        f"Save image with predicted landmarks to '{img_with_landmarks_name}'")

            progress += inputs.shape[0]

            logger.info(f"Progress: {progress / total:.2%}")

        with open(path_to_pred, "w", encoding="utf-8") as pred_file:
            json.dump(landmarks, pred_file)
            logger.info(f"Save predcitions to '{path_to_pred}'")


def eval_model(model, dataloaders, dataset_sizes,
               writer, use_gpu=True, epoches=5, dataset='val',
               save_path='./', num_landmarks=68):
    """This function as is on the original code. It may no work.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_nme = 0
    model.eval()
    for epoch in range(epoches):
        running_loss = 0
        step = 0
        total_nme = 0
        total_count = 0
        fail_count = 0
        nmes = []
        # running_corrects = 0

        # Iterate over data.
        with torch.no_grad():
            for data in dataloaders[dataset]:
                total_runtime = 0
                run_count = 0
                step_start = time.time()
                step += 1
                # get the inputs
                inputs = data['image'].type(torch.FloatTensor)
                labels_heatmap = data['heatmap'].type(torch.FloatTensor)
                labels_boundary = data['boundary'].type(torch.FloatTensor)
                landmarks = data['landmarks'].type(torch.FloatTensor)
                loss_weight_map = data['weight_map'].type(torch.FloatTensor)
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.to(device)
                    labels_heatmap = labels_heatmap.to(device)
                    labels_boundary = labels_boundary.to(device)
                    loss_weight_map = loss_weight_map.to(device)

                labels = torch.cat((labels_heatmap, labels_boundary), 1)
                single_start = time.time()
                outputs, boundary_channels = model(inputs)
                single_end = time.time()
                total_runtime += time.time() - single_start
                run_count += 1
                step_end = time.time()
                for i in range(inputs.shape[0]):
                    img = inputs[i]
                    img = img.cpu().numpy()
                    img = img.transpose((1, 2, 0))*255.0
                    img = img.astype(np.uint8)
                    img = Image.fromarray(img)
                    # pred_heatmap = outputs[-1][i].detach().cpu()[:-1, :, :]
                    pred_heatmap = outputs[-1][:, :-1, :, :][i].detach().cpu()
                    pred_landmarks, _ = get_preds_fromhm(
                        pred_heatmap.unsqueeze(0))
                    pred_landmarks = pred_landmarks.squeeze().numpy()

                    gt_landmarks = data['landmarks'][i].numpy()
                    if num_landmarks == 68:
                        left_eye = np.average(gt_landmarks[36:42], axis=0)
                        right_eye = np.average(gt_landmarks[42:48], axis=0)
                        norm_factor = np.linalg.norm(left_eye - right_eye)
                        # norm_factor = np.linalg.norm(gt_landmarks[36]- gt_landmarks[45])

                    elif num_landmarks == 98:
                        norm_factor = np.linalg.norm(
                            gt_landmarks[60] - gt_landmarks[72])
                    elif num_landmarks == 19:
                        left, top = gt_landmarks[-2, :]
                        right, bottom = gt_landmarks[-1, :]
                        norm_factor = math.sqrt(
                            abs(right - left)*abs(top-bottom))
                        gt_landmarks = gt_landmarks[:-2, :]
                    elif num_landmarks == 29:
                        # norm_factor = np.linalg.norm(gt_landmarks[8]- gt_landmarks[9])
                        norm_factor = np.linalg.norm(
                            gt_landmarks[16] - gt_landmarks[17])
                    single_nme = (np.sum(np.linalg.norm(
                        pred_landmarks*4 - gt_landmarks, axis=1)) / pred_landmarks.shape[0]) / norm_factor

                    nmes.append(single_nme)
                    total_count += 1
                    if single_nme > 0.1:
                        fail_count += 1
                if step % 10 == 0:
                    print('Step {} Time: {:.6f} Input Mean: {:.6f} Output Mean: {:.6f}'.format(
                        step, step_end - step_start,
                        torch.mean(labels),
                        torch.mean(outputs[0])))
                # gt_landmarks = landmarks.numpy()
                # pred_heatmap = outputs[-1].to('cpu').numpy()
                gt_landmarks = landmarks
                batch_nme = fan_NME(
                    outputs[-1][:, :-1, :, :].detach().cpu(), gt_landmarks, num_landmarks)
                # batch_nme = 0
                total_nme += batch_nme
        epoch_nme = total_nme / dataset_sizes['val']
        global_nme += epoch_nme
        nme_save_path = os.path.join(save_path, 'nme_log.npy')
        np.save(nme_save_path, np.array(nmes))
        print('NME: {:.6f} Failure Rate: {:.6f} Total Count: {:.6f} Fail Count: {:.6f}'.format(
            epoch_nme, fail_count/total_count, total_count, fail_count))
    print('Evaluation done! Average NME: {:.6f}'.format(global_nme/epoches))
    print('Everage runtime for a single batch: {:.6f}'.format(
        total_runtime/run_count))
    return model
