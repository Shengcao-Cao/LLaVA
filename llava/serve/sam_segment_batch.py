import argparse
import cv2
import json
import numpy as np
import os
import spacy
import torch

from matplotlib import pyplot as plt
from pycocotools import mask as mask_utils
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoTokenizer

from llava.serve.seg_utils import *


def cos_sim(a, b):
    a = a.reshape(-1)
    a = a / np.linalg.norm(a)
    b = b.reshape(-1)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def get_clustering_attn(seq, attns, tokenizer, threshold=0.5):
    seq = seq[1:]
    token_start_char = []
    token_end_char = []
    prev_length = 0
    curr_string = ''
    for i in range(len(seq)):
        curr_string = tokenizer.decode(seq[:i + 1], skip_special_tokens=True)
        curr_length = len(curr_string)
        token_start_char.append(prev_length)
        token_end_char.append(curr_length)
        prev_length = curr_length

    clusters = []
    for i in range(len(seq)):
        cluster = {
            'phrase': [seq[i]],
            'attns': [attns[i]],
            'mean_attn': attns[i],
            'start': i,
            'end': i + 1,
            'start_char': token_start_char[i],
            'end_char': token_end_char[i],
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
        clusters[i]['end'] = clusters[j]['end']
        clusters[i]['end_char'] = clusters[j]['end_char']
        del clusters[j]

    for cluster in clusters:
        cluster['phrase'] = tokenizer.decode(cluster['phrase'], skip_special_tokens=True)

    return clusters

def get_clustering_spacy(seq, attns, tokenizer, spacy_model):
    seq = seq[1:]
    token_start_char = []
    token_end_char = []
    prev_length = 0
    curr_string = ''
    for i in range(len(seq)):
        curr_string = tokenizer.decode(seq[:i + 1], skip_special_tokens=True)
        curr_length = len(curr_string)
        token_start_char.append(prev_length)
        token_end_char.append(curr_length)
        prev_length = curr_length

    noun_phrases = list(spacy_model(curr_string).noun_chunks)
    phrase_start_char = [p.start_char for p in noun_phrases]
    phrase_end_char = [p.end_char for p in noun_phrases]
    phrases = [p.text for p in noun_phrases]

    clusters = []
    for i in range(len(phrases)):
        token_indices = []
        for j in range(len(token_start_char)):
            if token_start_char[j] < phrase_end_char[i] and token_end_char[j] > phrase_start_char[i]:
                token_indices.append(j)
        if token_indices:
            cluster = {
                'phrase': phrases[i],
                'attns': [attns[j] for j in token_indices],
                'mean_attn': sum([attns[j] for j in token_indices]) / len(token_indices),
                'start': token_indices[0],
                'end': token_indices[-1] + 1,
                'start_char': phrase_start_char[i],
                'end_char': phrase_end_char[i],
            }
            clusters.append(cluster)

    return clusters

def get_clustering_spacy_attn(seq, attns, tokenizer, spacy_model, threshold=0.5):
    seq = seq[1:]
    token_start_char = []
    token_end_char = []
    prev_length = 0
    curr_string = ''
    for i in range(len(seq)):
        curr_string = tokenizer.decode(seq[:i + 1], skip_special_tokens=True)
        curr_length = len(curr_string)
        token_start_char.append(prev_length)
        token_end_char.append(curr_length)
        prev_length = curr_length

    noun_phrases = list(spacy_model(curr_string).noun_chunks)
    phrase_start_char = [p.start_char for p in noun_phrases]
    phrase_end_char = [p.end_char for p in noun_phrases]
    phrases = [p.text for p in noun_phrases]

    clusters = []
    for i in range(len(phrases)):
        token_indices = []
        for j in range(len(token_start_char)):
            if token_start_char[j] < phrase_end_char[i] and token_end_char[j] > phrase_start_char[i]:
                token_indices.append(j)
        if token_indices:
            cluster = {
                'phrase': phrases[i],
                'attns': [attns[j] for j in token_indices],
                'mean_attn': sum([attns[j] for j in token_indices]) / len(token_indices),
                'start': token_indices[0],
                'end': token_indices[-1] + 1,
                'start_char': phrase_start_char[i],
                'end_char': phrase_end_char[i],
            }
            # clusters.append(cluster)

            # get sub-clusters
            sub_clusters = []
            for j in token_indices:
                cluster = {
                    'phrase': phrases[i],
                    'attns': [attns[j]],
                    'mean_attn': attns[j],
                    'start': j,
                    'end': j + 1,
                    'start_char': phrase_start_char[i],
                    'end_char': phrase_end_char[i],
                }
                sub_clusters.append(cluster)

            while True:
                max_sim = -1.0
                pair = None
                for j in range(len(sub_clusters) - 1):
                    k = j + 1
                    sim = cos_sim(sub_clusters[j]['mean_attn'], sub_clusters[k]['mean_attn'])
                    if sim > max_sim:
                        max_sim = sim
                        pair = (j, k)

                if max_sim < threshold:
                    break

                j, k = pair
                sub_clusters[j]['attns'] += sub_clusters[k]['attns']
                sub_clusters[j]['mean_attn'] = sum(sub_clusters[j]['attns']) / len(sub_clusters[j]['attns'])
                sub_clusters[j]['end'] = sub_clusters[k]['end']
                del sub_clusters[k]

            # add all sub-clusters
            clusters += sub_clusters
            # # add the largest (longest) sub-cluster only
            # max_length = 0
            # max_cluster = []
            # for sub_cluster in sub_clusters:
            #     length = sub_cluster['end'] - sub_cluster['start']
            #     if length > max_length:
            #         max_length = length
            #         max_cluster = [sub_cluster]
            #     elif length == max_length:
            #         max_cluster.append(sub_cluster)
            # if max_length > 0:
            #     clusters += max_cluster

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
    parser.add_argument('--attn-box-thr', type=float, default=0.75)
    parser.add_argument('--cluster-thr', type=float, default=0.5)
    parser.add_argument('--attn-temp', type=float, default=0.0002)
    parser.add_argument('--sample', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualize-folder', type=str)
    parser.add_argument('--nms-thr', type=float, default=-1.0)
    parser.add_argument('--cluster-method', type=str, default='spacy_attn')
    parser.add_argument('--filter-score', action='store_true')
    parser.add_argument('--filter-sim', action='store_true')
    args = parser.parse_args()

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to(device=args.device)
    predictor = SamPredictor(sam)

    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    spacy_model = spacy.load('en_core_web_lg')

    image_files = os.listdir(args.image_folder)
    image_files = sorted(image_files)
    os.makedirs(args.output_folder, exist_ok=True)
    if args.visualize:
        os.makedirs(args.visualize_folder, exist_ok=True)

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
        if args.visualize:
            vis_path = os.path.join(args.visualize_folder, image_file[:-4] + '.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        S = max(W, H)
        if W == H:
            crop_coor = (0, 0, W, H)
        elif W > H:
            crop_coor = (0, (W - H) // 2, W, (W + H) // 2)
        else:
            crop_coor = ((H - W) // 2, 0, (H + W) // 2, H)

        predictor.set_image(image)

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

        if args.cluster_method == 'attn':
            clusters = get_clustering_attn(seq, attns, tokenizer=tokenizer, threshold=args.cluster_thr)
        elif args.cluster_method == 'spacy':
            clusters = get_clustering_spacy(seq, attns, tokenizer=tokenizer, spacy_model=spacy_model)
        elif args.cluster_method == 'spacy_attn':
            clusters = get_clustering_spacy_attn(seq, attns, tokenizer=tokenizer, spacy_model=spacy_model, threshold=args.cluster_thr)
        else:
            raise ValueError(f'Invalid cluster method: {args.cluster_method}')
        save_phrases = []
        save_masks = []
        save_token_locs = []
        save_scores = []
        save_ious = []
        save = json.load(open(json_path))
        if args.visualize:
            vis_all = []

        for i, cluster in enumerate(clusters):
            phrase = cluster['phrase']
            attn = cluster['mean_attn']
            attn = (attn / args.attn_temp).astype(np.float32)
            # attn = (attn / args.attn_temp).astype(np.float32)
            # attn_logits = ((attn - attn.mean()) / args.attn_temp).astype(np.float32)
            attn = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
            attn = torch.nn.functional.interpolate(attn, (S, S), mode='bicubic', align_corners=False)
            attn = attn[0, 0, crop_coor[1]:crop_coor[3], crop_coor[0]:crop_coor[2]]
            attn = attn.numpy()
            attn_mask = convert_mask_SAM(attn)
            attn_box = convert_box_SAM(attn, threshold=args.attn_box_thr)
            # attn_box = convert_trivial_box_SAM(attn)
            # attn_points, attn_point_labels = convert_point_SAM(attn)
            if attn_box is not None:
                mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
                # mask, score, logits = predictor.predict(point_coords=attn_points, point_labels=attn_point_labels, box=attn_box, multimask_output=False)
                mask = mask.reshape(H, W).astype(np.uint8)
                score = score.item()
                if mask.sum() > 0 and score > 0.5:
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    iou = compute_iou_masks(attn, mask[np.newaxis])[0]
                    save_phrases.append(phrase)
                    save_masks.append(rle)
                    save_token_locs.append((cluster['start_char'], cluster['end_char']))
                    save_scores.append(score)
                    save_ious.append(iou)
                if args.visualize:
                    vis_box = image.copy()
                    vis_attn = image.copy()
                    vis_mask = image.copy()
                    vis_box = cv2.rectangle(vis_box, (attn_box[0], attn_box[1]), (attn_box[2], attn_box[3]), (255, 0, 0), 2)
                    attn_clip = np.clip(attn, 0.0, 1.0)[:, :, np.newaxis]
                    vis_attn = (1 - attn_clip) * vis_attn + attn_clip * np.array([255, 0, 0]).reshape(1, 1, 3).astype(np.uint8)
                    vis_mask[mask.astype(bool)] = (255, 0, 0)
                    vis = np.concatenate([vis_box, vis_attn, vis_mask], axis=1)
                    vis_phrase = np.ones((int(vis.shape[0]) // 10, vis.shape[1], 3), dtype=np.uint8) * 255
                    text_width, text_height = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, 1, 4)[0]
                    cv2.putText(vis_phrase, phrase, (vis_phrase.shape[1] // 2 - text_width // 2, vis_phrase.shape[0] // 2 + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                    vis = np.concatenate([vis_phrase, vis], axis=0)
                    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    vis_all.append(vis)

        if args.nms_thr > 0:
            masks = np.array([mask_utils.decode(x) for x in save_masks])
            boxes = np.array([mask_to_box(x) for x in masks])
            scores = np.array(save_ious)
            indices = nms(masks, boxes, scores, iou=args.nms_thr)
            save_phrases = [save_phrases[i] for i in indices]
            save_masks = [save_masks[i] for i in indices]
            save_token_locs = [save_token_locs[i] for i in indices]

        if args.filter_score:
            filtered_phrases = []
            filtered_masks = []
            filtered_token_locs = []
            filtered_scores = []
            filtered_ious = []
            visited = [0] * len(save_phrases)
            for i in range(len(save_phrases)):
                if not visited[i]:
                    group = [i]
                    visited[i] = 1
                    for j in range(i + 1, len(save_phrases)):
                        if not visited[j] and save_token_locs[i][0] == save_token_locs[j][0] and save_token_locs[i][1] == save_token_locs[j][1]:
                            group.append(j)
                            visited[j] = 1
                    if len(group) == 1:
                        filtered_phrases.append(save_phrases[i])
                        filtered_masks.append(save_masks[i])
                        filtered_token_locs.append(save_token_locs[i])
                        filtered_scores.append(save_scores[i])
                        filtered_ious.append(save_ious[i])
                    else:
                        max_score = -1.0
                        max_index = -1
                        for j in group:
                            if save_scores[j] > max_score:
                                max_score = save_scores[j]
                                max_index = j
                        filtered_phrases.append(save_phrases[max_index])
                        filtered_masks.append(save_masks[max_index])
                        filtered_token_locs.append(save_token_locs[max_index])
                        filtered_scores.append(save_scores[max_index])
                        filtered_ious.append(save_ious[max_index])
            save_phrases = filtered_phrases
            save_masks = filtered_masks
            save_token_locs = filtered_token_locs
            save_scores = filtered_scores
            save_ious = filtered_ious

        if args.filter_sim:
            before_N = len(save_phrases)
            masks = np.array([mask_utils.decode(x) for x in save_masks])
            boxes = np.array([mask_to_box(x) for x in masks])
            scores = np.array(save_scores)
            phrases = save_phrases
            indices = nms_phrase(masks, boxes, scores, phrases, spacy_model)
            save_phrases = [save_phrases[i] for i in indices]
            save_masks = [save_masks[i] for i in indices]
            save_token_locs = [save_token_locs[i] for i in indices]
            after_N = len(save_phrases)
            print(f'Filtering: {before_N} -> {after_N}', flush=True)

        if args.visualize:
            vis_all = [vis_all[i] for i in indices]
            vis_all = np.concatenate(vis_all, axis=0)
            cv2.imwrite(vis_path, vis_all)

        save['phrases'] = save_phrases
        save['pred_masks'] = save_masks
        save['token_locs'] = save_token_locs

        with open(output_path, 'w') as f:
            json.dump(save, f, indent=2)
