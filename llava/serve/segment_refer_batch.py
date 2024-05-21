import argparse
import cv2
import json
import math
import os

import numpy as np
import spacy
import torch

from PIL import Image
from pycocotools import mask as mask_utils
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from llava.serve.refcoco_dataset import ReferSegmDataset
from llava.serve.seg_utils import *


# def get_nlp_embedding(text, nlp_tokenizer, nlp_model):
#     with torch.no_grad():
#         inputs = nlp_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
#         inputs = {k: v.cuda() for k, v in inputs.items()}
#         outputs = nlp_model(**inputs)
#         embedding = outputs.last_hidden_state[0].mean(dim=0).detach().cpu().numpy()
#     return embedding


# def get_nlp_similarity(text1, text2, nlp_tokenizer, nlp_model):
#     emb1 = get_nlp_embedding(text1, nlp_tokenizer, nlp_model)
#     emb2 = get_nlp_embedding(text2, nlp_tokenizer, nlp_model)
#     sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6)
#     return sim


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to("cuda")
    predictor = SamPredictor(sam)
    spacy_model = spacy.load(args.spacy_model)
    # clip_model = CLIPModel.from_pretrained(args.clip_model).to("cuda")
    # clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    # nlp_tokenizer = AutoTokenizer.from_pretrained(args.nlp_model, device_map='cuda')
    # nlp_model = AutoModel.from_pretrained(args.nlp_model, device_map='cuda')

    refer_dataset = ReferSegmDataset(args.dataset_dir, refer_segm_data=args.refer_segm_data, split=args.split, sample=args.sample)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.visualize:
        os.makedirs(args.visualize, exist_ok=True)
    record = []

    # counter = 0
    # ious = []
    for refer_data in tqdm(refer_dataset):
        # counter += 1
        # if counter > 10:
        #     break
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

            # print(input_ids.shape)
            # print(input_ids[0, args.ref_start:args.ref_end])
            refer_phrase_input = input_ids[0, args.ref_start:args.ref_end].cpu().numpy()
            refer_phrase_spacy = spacy_model(tokenizer.decode(refer_phrase_input))
            for word in refer_phrase_spacy:
                if word.dep_ == 'ROOT':
                    head_word_start = word.idx
                    head_word_end = word.idx + len(word.text)
                    break
            for chunk in refer_phrase_spacy.noun_chunks:
                if chunk.start_char <= head_word_start and chunk.end_char >= head_word_end:
                    head_word_start = chunk.start_char
                    head_word_end = chunk.end_char
                    break
            # clip_text = refer_phrase_spacy.text[head_word_start:head_word_end]
            # clip_text_inputs = clip_processor(text=[clip_text], return_tensors="pt")
            # clip_text_inputs = {k: v.to('cuda') for k, v in clip_text_inputs.items()}
            # clip_text_outputs = clip_model.get_text_features(**clip_text_inputs)
            # clip_text_embedding = (clip_text_outputs / clip_text_outputs.norm(dim=-1, keepdim=True))[0]
            # print('#' * 80)
            # print(clip_text)
            # print(os.path.basename(image_path))
            # print(input_ids)
            # print(refer_phrase, '||', refer_phrase_spacy.text[head_word_start:head_word_end])
            phrase_decode, token_start_char, token_end_char = decode_token_seq(refer_phrase_input, tokenizer)
            head_word_token = associate_phrase_token([head_word_start], [head_word_end], token_start_char, token_end_char)[0]
            # print(phrase_decode, token_start_char, token_end_char, head_word_start, head_word_end)
            # print(head_word_token)
            head_word_token_start = min(head_word_token) + args.ref_start + args.seq_end - args.seq_start - 1
            head_word_token_end = max(head_word_token) + args.ref_start + args.seq_end - args.seq_start
            # print(head_word_token_start, head_word_token_end)

            seq = output_ids["sequences"][0][1:]    # skip the first special token <s>
            attns = []
            attns_text = []
            for attn in output_ids["attentions"]:
                attn = torch.cat(attn)
                attn_text = attn[:, :, -1, head_word_token_start:head_word_token_end].detach().cpu().numpy()
                attn_text = attn_text.mean()
                attns_text.append(attn_text)
                attn = attn[:, :, -1, args.seq_start:args.seq_end].detach().cpu().numpy()
                attn = attn.mean(axis=(0, 1)).reshape(args.attn_h, args.attn_w)
                attns.append(attn)
            attns = np.array(attns)
            attns_text = np.array(attns_text)

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

            # response, token_start_char, token_end_char = decode_token_seq(seq, tokenizer)
            # noun_phrases, phrase_start_char, phrase_end_char = parse_response(response, spacy_model)
            # phrase_token = associate_phrase_token(phrase_start_char, phrase_end_char, token_start_char, token_end_char)

            # best_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            # best_mask_score = 0.0
            # best_token_idx = -1
            # best_phrase_idx = -1
            # qualified_scores = []
            # qualified_tokens = []
            # qualified_phrases = []

            # attns = attns - attns.mean(axis=0)
            # attns = (attns / args.attn_temp).astype(np.float32)
            # for phrase_idx in range(len(phrase_token)):
            #     # phrase_sim = get_nlp_similarity(refer_phrase, noun_phrases[phrase_idx].text, nlp_tokenizer, nlp_model)
            #     for token_idx in phrase_token[phrase_idx]:
            #         attn = attns[token_idx]
            #         attn_upsampled = upsample_attn(attn, image_s, crop_coor)
            #         attn_mask = convert_mask_SAM(attn_upsampled)
            #         attn_box = convert_box_SAM(attn_upsampled, threshold=args.attn_box_thr)
            #         if attn_box is not None:
            #             mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
            #             mask = mask.reshape(image_height, image_width).astype(np.uint8)
            #             score = score.item()
            #             # score = phrase_sim * score
            #             if args.oracle:
            #                 iou = np.logical_and(mask, gt_mask).sum() / (np.logical_or(mask, gt_mask).sum() + 1e-6)
            #             if mask.sum() > 0 and score > best_mask_score:
            #                 best_mask = mask
            #                 best_mask_score = score
            #                 best_token_idx = token_idx
            #                 best_phrase_idx = phrase_idx

            response, token_start_char, token_end_char = decode_token_seq(seq, tokenizer)

            best_attn = np.zeros((image_height, image_width), dtype=np.uint8)
            best_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            best_token_idx = -1
            best_sam_score = 0.0
            best_attn_score = 0.0
            # best_clip_score = 0.0
            best_total_score = 0.0

            attns = attns - attns.mean(axis=0)
            attns = (attns / args.attn_temp).astype(np.float32)
            attns_text = attns_text - attns_text.mean()
            attns_text = (attns_text / args.attn_text_temp).astype(np.float32)
            # print(attns_text.mean(), attns_text.max())
            attns_text = np.clip(attns_text, 0.0, 1.0)
            # print(refer_phrase)
            for token_idx in range(len(seq)):
                attn_score = attns_text[token_idx]
                if attn_score < 0.5:
                    continue
                attn = attns[token_idx]
                attn_upsampled = upsample_attn(attn, image_s, crop_coor)
                attn_mask = convert_mask_SAM(attn_upsampled)
                attn_box = convert_box_SAM(attn_upsampled, threshold=args.attn_box_thr)
                if attn_box is not None:
                    mask, sam_score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
                    mask = mask.reshape(image_height, image_width).astype(np.uint8)
                    sam_score = sam_score.item()
                    if sam_score < 0.5:
                        continue
                    box = mask_to_box(mask)
                    if box is None or box[2] - box[0] < 5 or box[3] - box[1] < 5:
                        continue
                    # clip_image = image.crop(box)
                    # clip_image_inputs = clip_processor(images=clip_image, return_tensors="pt")
                    # clip_image_inputs = {k: v.to('cuda') for k, v in clip_image_inputs.items()}
                    # clip_image_outputs = clip_model.get_image_features(**clip_image_inputs)
                    # clip_image_embedding = (clip_image_outputs / clip_image_outputs.norm(dim=-1, keepdim=True))[0]
                    # clip_score = (clip_text_embedding * clip_image_embedding).sum().item()
                    # clip_score = 0.0

                    # total_score = sam_score + attn_score + clip_score
                    total_score = sam_score + attn_score
                    if total_score > best_total_score:
                        best_attn = attn_upsampled
                        best_mask = mask
                        best_token_idx = token_idx
                        best_sam_score = sam_score
                        best_attn_score = attn_score
                        # best_clip_score = clip_score
                        best_total_score = total_score
                    # iou = np.logical_and(mask, gt_mask).sum() / (np.logical_or(mask, gt_mask).sum() + 1e-6)
                    # if iou > 0.75:
                    #     print('Qualified:', response[token_start_char[token_idx]:token_end_char[token_idx]], round(iou, 4), round(sam_score, 4), round(attn_score, 4), round(clip_score, 4), round(total_score, 4))
            # iou = np.logical_and(best_mask, gt_mask).sum() / (np.logical_or(best_mask, gt_mask).sum() + 1e-6)
            # ious.append(iou)
            # print(attns_text.max())
            # print('Selected:', response[token_start_char[best_token_idx]:token_end_char[best_token_idx]], round(iou, 4), round(best_sam_score, 4), round(best_attn_score, 4), round(best_clip_score, 4), round(best_total_score, 4))
            # print('#' * 80)

            # best_attn = np.zeros_like(attns[0], dtype=np.float32)
            # best_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            # best_mask_score = 0.0

            # attns = attns - attns.mean(axis=0)
            # attns = (attns / args.attn_temp).astype(np.float32)
            # attns_text = attns_text.astype(np.float32)
            # # print(attns_text.max())
            # attns_text_sum = []
            # for token_idx in range(len(seq)):
            #     if attns_text[token_idx] > attns_text.max() * 0.8:
            #         best_attn += attns_text[token_idx] * attns[token_idx]
            #         attns_text_sum.append(attns_text[token_idx])
            #         print(response[token_start_char[token_idx]:token_end_char[token_idx]], end=' ')
            # print()

            # print(len(attns_text_sum))
            # best_attn = best_attn / sum(attns_text_sum)
            # # best_attn = best_attn / attns_text.sum()
            # # best_attn = best_attn / best_attn.max() / 0.1
            # attn_upsampled = upsample_attn(best_attn, image_s, crop_coor)
            # attn_mask = convert_mask_SAM(attn_upsampled)
            # attn_box = convert_box_SAM(attn_upsampled, threshold=args.attn_box_thr)
            # if attn_box is not None:
            #     mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
            #     mask = mask.reshape(image_height, image_width).astype(np.uint8)
            #     score = score.item()
            #     best_mask = mask
            #     best_mask_score = score
            # iou = np.logical_and(best_mask, gt_mask).sum() / (np.logical_or(best_mask, gt_mask).sum() + 1e-6)
            # ious.append(iou)
            # print(refer_phrase, iou)
            # print('#' * 80)

            # print('#'*80)
            # # print('Best:', best_mask_score, '||', refer_phrase, '||', noun_phrases[best_phrase_idx], '||', response[token_start_char[best_token_idx]:token_end_char[best_token_idx]])
            # print(refer_phrase)
            # print([noun_phrases[i] for i in qualified_phrases])
            # print('#'*80)

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

            # visualize prediction and ground truth
            if args.visualize:
                image_file = os.path.basename(image_path)
                # attn_upsampled = attn_upsampled / attn_upsampled.max() / 0.1
                attn_vis = (np.clip(best_attn, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(args.visualize, image_file).replace(".jpg", "_attn.jpg"), attn_vis)

                pred_vis = image_np.copy()
                pred_vis[best_mask.astype(bool)] = [255, 0, 0]
                pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.visualize, image_file).replace(".jpg", "_pred.jpg"), pred_vis)

                gt_vis = image_np.copy()
                gt_vis[gt_mask.astype(bool)] = [255, 0, 0]
                gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.visualize, image_file).replace(".jpg", "_gt.jpg"), gt_vis)

            save_dict = {
                "image_path": image_path,
                "refer_phrase": refer_phrase,
                "response": response,
                "pred_mask": binary_mask_to_rle(best_mask),
                "gt_mask": binary_mask_to_rle(gt_mask),
            }
            record.append(save_dict)

    # print("Mean IoU:", np.mean(ious))

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
    parser.add_argument("--template", type=str, default='Describe the "{}."')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seq-start", type=int, default=35)
    parser.add_argument("--seq-end", type=int, default=35+576)
    parser.add_argument("--ref-start", type=int, default=42)
    parser.add_argument("--ref-end", type=int, default=-6)
    parser.add_argument("--attn-h", type=int, default=24)
    parser.add_argument("--attn-w", type=int, default=24)
    parser.add_argument("--sam-model", type=str, default="vit_h")
    parser.add_argument("--sam-ckpt", type=str, default="save/sam_checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_lg")
    # parser.add_argument("--nlp-model", type=str, default="bert-base-uncased")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--attn-temp", type=float, default=0.0002)
    parser.add_argument("--attn-text-temp", type=float, default=0.005)
    parser.add_argument("--attn-box-thr", type=float, default=0.75)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--visualize", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
