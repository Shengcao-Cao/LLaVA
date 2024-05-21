import argparse
import json
import math
import os

import numpy as np
import spacy
import torch

from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from llava.serve.refcoco_dataset import ReferSegmDataset
from llava.serve.seg_utils import *


class RegionCapDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file):
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def __getitem__(self, idx):
        image_id = self.image_dict_keys[idx]
        filename = self.image_dict[image_id]['file_name']
        x1, y1, w, h = self.ann_dict[image_id]['bbox']
        bbox = [x1, y1, x1 + w, y1 + h]
        gt = self.ann_dict[image_id]['caption']
        return image_id, filename, bbox, gt


def box_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter / (areaA + areaB - inter + 1e-6)
    return iou


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to("cuda")
    predictor = SamPredictor(sam)
    spacy_model = spacy.load("en_core_web_lg")

    regioncap_dataset = RegionCapDataset(args.annotation_file)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    record = []

    for regioncap_data in tqdm(regioncap_dataset):
        image_id, filename, bbox, gt = regioncap_data
        image_path = os.path.join(args.dataset_dir, filename)

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        image_np, image_height, image_width, image_s, crop_coor = load_image_sam(image_path)
        # predictor.set_image(image_np)

        qs = args.template1
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True)

        seq = output_ids["sequences"][0][1:]    # skip the first special token <s>
        attns = []
        for attn in output_ids["attentions"]:
            attn = torch.cat(attn)
            attn = attn[:, :, -1, args.seq_start:args.seq_end].detach().cpu().numpy()
            attn = attn.mean(axis=(0, 1)).reshape(args.attn_h, args.attn_w)
            attns.append(attn)
        attns = np.array(attns)

        response, token_start_char, token_end_char = decode_token_seq(seq, tokenizer)
        noun_phrases, phrase_start_char, phrase_end_char = parse_response(response, spacy_model)
        phrase_token = associate_phrase_token(phrase_start_char, phrase_end_char, token_start_char, token_end_char)

        attns = attns - attns.mean(axis=0)
        attns = (attns / args.attn_temp).astype(np.float32)
        masks = []
        boxes = []
        for token_idx in range(len(seq)):
            attn = attns[token_idx]
            attn_upsampled = upsample_attn(attn, image_s, crop_coor)
            attn_mask = convert_mask_SAM(attn_upsampled)
            attn_box = convert_box_SAM(attn_upsampled, threshold=args.attn_box_thr)
            masks.append(attn_mask)
            boxes.append(attn_box)
            # if attn_box is not None:
            #     mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
            #     mask = mask.reshape(image_height, image_width).astype(np.uint8)
            #     score = score.item()
            #     if mask.sum() > 0 and score > 0.5:
            #         masks.append(mask)
            #         boxes.append(mask_to_box(mask))
            #     else:
            #         masks.append(None)
            #         boxes.append(None)
            # else:
            #     masks.append(None)
            #     boxes.append(None)

        best_phrase = -1
        best_iou = -1.0
        best_box = None

        for i in range(len(phrase_token)):
            for j in phrase_token[i]:
                iou = box_iou(boxes[j], bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_phrase = noun_phrases[i]
                    best_box = boxes[j]

        conv.messages[-1][-1] = response
        conv.append_message(conv.roles[0], args.template2.format(best_phrase))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print('#'*80)
        print(prompt)
        print(best_iou, best_box, bbox)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True)

        response = tokenizer.decode(output_ids["sequences"][0], skip_special_tokens=True).strip()

        print(gt, '||', best_phrase, '||', response)
        print('#'*80)
        save_dict = {
            "image_id": image_id,
            "phrase": best_phrase,
            "caption": response,
        }
        record.append(save_dict)

    with open(args.output_path, "w") as f:
        json.dump(record, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--template1", type=str, default="Describe the image in detail.")
    parser.add_argument("--template2", type=str, default='Describe "{}" in detail.')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seq-start", type=int, default=35)
    parser.add_argument("--seq-end", type=int, default=35+576)
    parser.add_argument("--attn-h", type=int, default=24)
    parser.add_argument("--attn-w", type=int, default=24)
    parser.add_argument("--sam-model", type=str, default="vit_h")
    parser.add_argument("--sam-ckpt", type=str, default="save/sam_checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--attn-temp", type=float, default=0.0002)
    parser.add_argument("--attn-box-thr", type=float, default=0.75)
    args = parser.parse_args()

    eval_model(args)
