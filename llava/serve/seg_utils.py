import numpy as np
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1 - x))

def convert_trivial_box_SAM(mask):
    h, w = mask.shape
    return np.array([0, 0, w, h])

def convert_box_SAM(mask, threshold=0.75):
    mask = mask > threshold
    if mask.sum() == 0:
        return None
    x, y = np.where(mask)
    box = [y.min(), x.min(), y.max() + 1, x.max() + 1]
    return np.array(box)

def convert_mask_SAM(mask, eps=1e-3, edge=256):
    mask = mask.copy()
    mask = np.clip(mask.astype(np.float32), eps, 1 - eps)
    mask = inv_sigmoid(mask)
    H, W = mask.shape
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    if H == W:
        pad_h = 0
        pad_w = 0
        mask = torch.nn.functional.interpolate(mask, (edge, edge), mode='bicubic', align_corners=False)
    elif H > W:
        w = int(W / H * edge)
        pad_w = edge - w
        mask = torch.nn.functional.interpolate(mask, (edge, w), mode='bicubic', align_corners=False)
        mask = torch.nn.functional.pad(mask, (0, pad_w, 0, 0), mode='constant', value=0)
    else:
        h = int(H / W * edge)
        pad_h = edge - h
        mask = torch.nn.functional.interpolate(mask, (h, edge), mode='bicubic', align_corners=False)
        mask = torch.nn.functional.pad(mask, (0, 0, 0, pad_h), mode='constant', value=0)
    mask = mask[0].numpy()
    return mask

def convert_point_SAM(mask):
    positive_points = mask > np.random.rand(*mask.shape)
    if positive_points.sum() == 0:
        return None, None
    x, y = np.where(positive_points)
    positive_points = np.stack([y, x], axis=1)
    negative_points = mask < -np.random.rand(*mask.shape)
    if negative_points.sum() == 0:
        return None, None
    x, y = np.where(negative_points)
    negative_points = np.stack([y, x], axis=1)
    all_points = np.concatenate([positive_points, negative_points], axis=0)
    labels = np.concatenate([np.ones(positive_points.shape[0]), np.zeros(negative_points.shape[0])], axis=0)

    return all_points, labels

def mask_to_box(mask):
    x, y = np.where(mask)
    box = [y.min(), x.min(), y.max() + 1, x.max() + 1]
    return np.array(box)

def compute_iou_boxes(box1, boxes):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_iou_masks(mask1, masks):
    intersection = np.logical_and(mask1, masks).sum(axis=(1, 2))
    union = np.logical_or(mask1, masks).sum(axis=(1, 2))
    return intersection / (union + 1e-6)

def nms(masks, boxes, scores, iou=0.5):
    N = boxes.shape[0]

    # sort by scores
    indices = np.argsort(scores)[::-1]
    inv_indices = np.argsort(indices)
    masks = masks[indices]
    boxes = boxes[indices]
    scores = scores[indices]

    keep = np.ones(N, dtype=bool)
    nms_boxes = None
    nms_masks = None

    for i in range(N):
        box = boxes[i]
        mask = masks[i]
        should_keep = True
        if nms_boxes is not None:
            box_ious = compute_iou_boxes(box, nms_boxes)
            indices = box_ious >= iou
            if np.any(indices):
                mask_ious = compute_iou_masks(mask, nms_masks[indices])
                if np.any(mask_ious >= iou):
                    should_keep = False

        keep[i] = should_keep
        if should_keep:
            if nms_boxes is None:
                nms_boxes = np.array([box])
                nms_masks = np.array([mask])
            else:
                nms_boxes = np.concatenate([nms_boxes, np.array([box])], axis=0)
                nms_masks = np.concatenate([nms_masks, np.array([mask])], axis=0)

    # return nms_masks, nms_boxes, scores[keep]
    return inv_indices[keep]
