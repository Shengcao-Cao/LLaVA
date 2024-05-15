import argparse
import json
import numpy as np
import os

from pycocotools import mask as mask_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    record = json.load(open(args.path, 'r'))
    intersections = []
    unions = []
    ious = []

    for i in range(len(record)):
        pred_mask = mask_utils.decode(record[i]['pred_mask'])
        gt_mask = mask_utils.decode(record[i]['gt_mask'])

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / (union + 1e-6)

        intersections.append(intersection)
        unions.append(union)
        ious.append(iou)

    miou = round(float(np.mean(ious)) * 100, 2)
    oiou = round(float(np.sum(intersections) / np.sum(unions) * 100), 2)

    print('Mean Intersection over Union:', miou)
    print('Overall Intersection over Union:', oiou)
