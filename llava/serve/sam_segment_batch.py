import argparse
import json
import numpy as np
import os
import torch

from PIL import Image
from pycocotools import mask as mask_utils
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoTokenizer

def convert_box_SAM(mask, threshold=0.75):
    mask = mask > threshold
    if mask.sum() == 0:
        return None
    x, y = np.where(mask)
    box = [y.min(), x.min(), y.max() + 1, x.max() + 1]
    return np.array(box)

def convert_mask_SAM(mask, eps=1e-3, edge=256):
    def inv_sigmois(x):
        return np.log(x / (1 - x))
    mask = mask.copy()
    mask = np.clip(mask.astype(np.float32), eps, 1 - eps)
    mask = inv_sigmois(mask)
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

    return nms_masks, nms_boxes, scores[keep]

def cos_sim(a, b):
    a = a.reshape(-1)
    a = a / np.linalg.norm(a)
    b = b.reshape(-1)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def get_clustering(seq, attns, threshold=0.5):
    seq = seq[1:]
    clusters = []
    for i in range(len(seq)):
        cluster = {
            'phrase': [seq[i]],
            'attns': [attns[i]],
            'mean_attn': attns[i],
        }
        clusters.append(cluster)

    while True:
        max_sim = -1.0
        pair = None
        for i in range(len(clusters) - 1):
            j = i + 1
            sim = cos_sim(clusters[i]['mean_attn'], clusters[j]['mean_attn'])
            if sim > max_sim:
                max_sim = sim
                pair = (i, j)

        if max_sim < threshold:
            break

        i, j = pair
        clusters[i]['phrase'] += clusters[j]['phrase']
        clusters[i]['attns'] += clusters[j]['attns']
        clusters[i]['mean_attn'] = sum(clusters[i]['attns']) / len(clusters[i]['attns'])
        del clusters[j]

    return clusters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--image-folder', type=str)
    parser.add_argument('--input-folder', type=str)
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='save/sam_checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--attn-thr', type=float, default=0.0002)
    parser.add_argument('--cluster-thr', type=float, default=0.5)
    parser.add_argument('--sample', type=str)
    args = parser.parse_args()

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to(device=args.device)
    predictor = SamPredictor(sam)

    image_files = os.listdir(args.image_folder)
    image_files = sorted(image_files)
    os.makedirs(args.output_folder, exist_ok=True)

    if args.sample is not None:
        image_ids = json.load(open(args.sample))
        image_ids = set(image_ids)
        image_files = [x for x in image_files if x[:-4] in image_ids]

    for count, image_file in enumerate(image_files):
        if (count + 1) % 100 == 0:
            print(f'Processing {count + 1}/{len(image_files)}', flush=True)
        image_path = os.path.join(args.image_folder, image_file)
        json_path = os.path.join(args.input_folder, image_file[:-4] + '.json')
        attn_path = os.path.join(args.input_folder, image_file[:-4] + '_attn.pth')
        output_path = os.path.join(args.output_folder, image_file[:-4] + '.json')
        image = Image.open(image_path).convert('RGB')
        W, H = image.size
        S = max(W, H)
        if W == H:
            crop_coor = (0, 0, W, H)
        elif W > H:
            crop_coor = (0, (W - H) // 2, W, (W + H) // 2)
        else:
            crop_coor = ((H - W) // 2, 0, (H + W) // 2, H)

        predictor.set_image(np.array(image))

        outputs = torch.load(attn_path, map_location='cpu')
        seq = outputs['sequences'].cpu().numpy()
        attentions = outputs['attentions']
        attns = []
        for i in range(len(attentions)):
            attn = attentions[i].numpy()
            attns.append(attn)

        attns = np.array(attns)
        mean_attn = attns.mean(axis=0)
        attns = attns - mean_attn

        clusters = get_clustering(seq, attns, threshold=args.cluster_thr)
        save_phrases = []
        save_masks = []
        save = json.load(open(json_path))
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)

        for cluster in clusters:
            phrase = tokenizer.decode(cluster['phrase'])
            attn = cluster['mean_attn']
            attn = np.clip(attn / args.attn_thr, 0.0, 1.0).astype(np.float32)
            attn = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
            attn = torch.nn.functional.interpolate(attn, (S, S), mode='bicubic', align_corners=False)
            attn = attn[0, 0, crop_coor[1]:crop_coor[3], crop_coor[0]:crop_coor[2]]
            attn = attn.numpy()
            attn_mask = convert_mask_SAM(attn)
            attn_box = convert_box_SAM(attn)
            if attn_box is not None:
                mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
                mask = mask.reshape(H, W).astype(np.uint8)
                score = score.item()
                if mask.sum() > 0 and score > 0.5:
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    save_phrases.append(phrase)
                    save_masks.append(rle)

        save['phrases'] = save_phrases
        save['pred_masks'] = save_masks

        with open(output_path, 'w') as f:
            json.dump(save, f, indent=2)
