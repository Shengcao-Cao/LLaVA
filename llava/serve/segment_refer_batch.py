import argparse
import json
import math
import os

import numpy as np
import spacy
import torch

from PIL import Image
from pycocotools import mask as mask_utils
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from llava.serve.refcoco_dataset import ReferSegmDataset
from llava.serve.seg_utils import *


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to("cuda")
    predictor = SamPredictor(sam)
    spacy_model = spacy.load("en_core_web_md")

    refer_dataset = ReferSegmDataset(args.dataset_dir, refer_segm_data=args.refer_segm_data, split=args.split, sample=args.sample)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    record = []

    for refer_data in tqdm(refer_dataset):
        image_path, gt_masks, refer_phrases = refer_data

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        image_np, image_height, image_width, image_s, crop_coor = load_image_sam(image_path)
        predictor.set_image(image_np)

        for refer_phrase, gt_mask in zip(refer_phrases, gt_masks):
            qs = args.template.format(refer_phrase)
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

            # response, token_start_char, token_end_char = decode_token_seq(seq, tokenizer)
            # noun_phrases, phrase_start_char, phrase_end_char = parse_response(response, spacy_model)
            # phrase_token = associate_phrase_token(phrase_start_char, phrase_end_char, token_start_char, token_end_char)

            # max_sim = -1.0
            # best_phrase_idx = -1
            # refer_phrase_spacy = spacy_model(refer_phrase)
            # for i, noun_phrase in enumerate(noun_phrases):
            #     sim = refer_phrase_spacy.similarity(noun_phrase)
            #     if sim > max_sim:
            #         max_sim = sim
            #         best_phrase_idx = i
            # best_tokens = phrase_token[best_phrase_idx]

            response, token_start_char, token_end_char = decode_token_seq(seq, tokenizer)
            common_substring, start_char_response, start_char_refer, longest = longest_common_substring(response.lower(), refer_phrase.lower())
            # print(response, refer_phrase)
            # print(common_substring, response[start_char_response:start_char_response+longest], refer_phrase[start_char_refer:start_char_refer+longest])
            phrase_token = associate_phrase_token([start_char_response], [start_char_response + longest], token_start_char, token_end_char)
            best_tokens = phrase_token[0]
            # print(best_tokens)
            # print(tokenizer.decode(seq[best_tokens]))

            if args.oracle:
                best_tokens = list(range(len(seq)))

            best_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            best_mask_score = 0.0
            best_token_idx = -1

            attns = attns - attns.mean(axis=0)
            attns = (attns / args.attn_temp).astype(np.float32)
            for token_idx in best_tokens:
                attn = attns[token_idx]
                attn_upsampled = upsample_attn(attn, image_s, crop_coor)
                attn_mask = convert_mask_SAM(attn_upsampled)
                attn_box = convert_box_SAM(attn_upsampled, threshold=args.attn_box_thr)
                if attn_box is not None:
                    mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
                    mask = mask.reshape(image_height, image_width).astype(np.uint8)
                    score = score.item()
                    if args.oracle:
                        iou = np.logical_and(mask, gt_mask).sum() / (np.logical_or(mask, gt_mask).sum() + 1e-6)
                        score = iou
                    if mask.sum() > 0 and score > best_mask_score:
                        best_mask = mask
                        best_mask_score = score
                        best_token_idx = token_idx

            # attns = attns - attns.mean(axis=0)
            # attns = (attns / args.attn_temp).astype(np.float32)
            # attn = attns[best_tokens]
            # attn = attn.min(axis=0)
            # attn_upsampled = upsample_attn(attn, image_s, crop_coor)
            # attn_mask = convert_mask_SAM(attn_upsampled)
            # attn_box = convert_box_SAM(attn_upsampled, threshold=args.attn_box_thr)
            # if attn_box is not None:
            #     mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
            #     mask = mask.reshape(image_height, image_width).astype(np.uint8)
            #     score = score.item()
            #     if args.oracle:
            #         iou = np.logical_and(mask, gt_mask).sum() / (np.logical_or(mask, gt_mask).sum() + 1e-6)
            #         score = iou
            #     if mask.sum() > 0:
            #         best_mask = mask

            # print(best_token_idx, len(seq))

            # # visualize prediction and ground truth
            # pred_vis = image_np.copy()
            # pred_vis[best_mask.astype(bool)] = [255, 0, 0]
            # pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("pred_vis.jpg", pred_vis)

            # gt_vis = image_np.copy()
            # gt_vis[gt_mask.astype(bool)] = [255, 0, 0]
            # gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("gt_vis.jpg", gt_vis)

            save_dict = {
                "image_path": image_path,
                "refer_phrase": refer_phrase,
                "response": response,
                "pred_mask": binary_mask_to_rle(best_mask),
                "gt_mask": binary_mask_to_rle(gt_mask),
            }
            record.append(save_dict)

    with open(args.output_path, "w") as f:
        json.dump(record, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--refer-segm-data", type=str, default="refcoco")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--template", type=str, default="Locate the {} in the image.")
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
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()

    eval_model(args)
