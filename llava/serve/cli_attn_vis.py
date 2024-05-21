import argparse
import cv2
import numpy as np
import os
import spacy
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from segment_anything import SamPredictor, sam_model_registry

from llava.serve.seg_utils import *


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to(device=args.device)
    predictor = SamPredictor(sam)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        if args.vis_path is not None:
            os.makedirs(args.vis_path, exist_ok=True)
            file_name = os.path.split(args.image_file)[-1].split('.')[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True)

            print(input_ids)

            save_sequences = output_ids["sequences"][0]
            save_scores = torch.cat(output_ids["scores"]).detach().cpu().numpy()
            save_attn = []
            save_attn_text = []
            for i in range(len(output_ids["attentions"])):
                save_attn_i = torch.cat(output_ids["attentions"][i])
                save_attn_i = save_attn_i[:, :, -1, args.seq_start:args.seq_end].detach().cpu().numpy()
                save_attn_i = save_attn_i.mean(axis=(0, 1)).reshape(args.attn_h, args.attn_w)
                save_attn.append(save_attn_i)

                save_attn_text_i = torch.cat(output_ids["attentions"][i])
                save_attn_text_i = save_attn_text_i[:, :, -1, 618:619].detach().cpu().numpy()
                save_attn_text_i = save_attn_text_i.mean()
                save_attn_text.append(save_attn_text_i)

            save_attn = np.array(save_attn)
            save_attn_text = np.array(save_attn_text)
            save_dict = {"sequences": save_sequences, "scores": save_scores, "attentions": save_attn, "attentions_text": save_attn_text}
            if args.vis_path is not None:
                torch.save(save_dict, os.path.join(args.vis_path, file_name + "_save.pth"))

        outputs = tokenizer.decode(output_ids["sequences"][0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        if args.vis_path is not None:
            image_np = cv2.imread(args.image_file)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            H, W = image_np.shape[:2]
            S = max(W, H)
            if W == H:
                crop_coor = (0, 0, W, H)
            elif W > H:
                crop_coor = (0, (W - H) // 2, W, (W + H) // 2)
            else:
                crop_coor = ((H - W) // 2, 0, (H + W) // 2, H)

            predictor.set_image(image_np)

            attns = save_dict['attentions']
            attns = ((attns - attns.mean(axis=0)) / args.attn_temp).astype(np.float32)
            attns_text = save_dict['attentions_text']
            # attns_text = ((attns_text - attns_text.mean()) / args.attn_temp).astype(np.float32)
            attns_text = (attns_text / args.attn_temp).astype(np.float32)
            tokens = tokenizer.convert_ids_to_tokens(save_dict['sequences'][1:])
            masks = []
            scores = []
            for i in range(len(tokens)):
                attn = attns[i]
                attn = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
                attn = torch.nn.functional.interpolate(attn, (S, S), mode='bicubic', align_corners=False)
                attn = attn[0, 0, crop_coor[1]:crop_coor[3], crop_coor[0]:crop_coor[2]]
                attn = attn.numpy()
                attn_mask = convert_mask_SAM(attn)
                attn_box = convert_box_SAM(attn, threshold=args.attn_box_thr)
                if attn_box is not None:
                    mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
                    mask = mask.reshape(H, W).astype(np.uint8)
                    score = score.item()
                else:
                    mask = np.zeros((H, W), dtype=np.uint8)
                    score = 0.0
                masks.append(mask)
                scores.append(score)

            if args.select_token:
                spacy_model = spacy.load("en_core_web_lg")
                response, token_start_char, token_end_char = decode_token_seq(save_dict['sequences'][1:], tokenizer)
                noun_phrases, phrase_start_char, phrase_end_char = parse_response(response, spacy_model)
                phrase_token = associate_phrase_token(phrase_start_char, phrase_end_char, token_start_char, token_end_char)
                ref_phrase = spacy_model(inp[17:-1])
                print(ref_phrase)
                sims = []
                for phrase in noun_phrases:
                    sim = ref_phrase.similarity(phrase)
                    sims.append(sim)
                    print(phrase, sim)

                # best_phrase = -1
                # best_sim = -1.0
                # for i in range(len(noun_phrases)):
                #     if sims[i] > best_sim:
                #         best_phrase = i
                #         best_sim = sims[i]

                # best_token = -1
                # best_score = -1.0
                # for i in phrase_token[best_phrase]:
                #     if scores[i] > best_score:
                #         best_token = i
                #         best_score = scores[i]

                best_phrase = -1
                best_token = -1
                best_score = -1.0
                for i in range(len(noun_phrases)):
                    for j in phrase_token[i]:
                        score = sims[i] * scores[j]
                        if score > best_score:
                            best_phrase = i
                            best_token = j
                            best_score = score

                print(f'Best token: "{response[token_start_char[best_token]:token_end_char[best_token]]}" from "{response[phrase_start_char[best_phrase]:phrase_end_char[best_phrase]]}"')

            if args.plot_token:
                # visualize attention maps and segmentation
                for i in range(len(tokens)):
                    token = tokens[i]
                    attn = attns[i]
                    plt.figure(figsize=(4, 4))
                    plt.imshow(attn, cmap='Reds', interpolation='nearest', vmin=0.0, vmax=1.0)
                    plt.axis('off')
                    # plt.title(f'"{token}"')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.vis_path, file_name + f"_attn_{i}.png"), bbox_inches='tight', transparent=True, pad_inches=0)
                    plt.close()

                    mask = masks[i]
                    score = scores[i]
                    vis_mask = image_np.copy()
                    vis_mask[mask.astype(bool)] = (255, 0, 0)
                    plt.figure(figsize=(8, 8))
                    plt.imshow(vis_mask)
                    plt.axis('off')
                    # plt.title(f'"{token}" {score:.2f}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.vis_path, file_name + f"_seg_{i}.png"), bbox_inches='tight', transparent=True, pad_inches=0)
                    plt.close()

            if args.plot_token_compact:
                # visualize attention maps
                plots_W = 5
                plots_H = (len(tokens) + plots_W - 1) // plots_W
                plt.figure(figsize=(plots_W * 2, plots_H * 2))
                for i in range(len(tokens)):
                    token = tokens[i]
                    attn = attns[i]
                    plt.subplot(plots_H, plots_W, i + 1)
                    plt.imshow(attn, cmap='Reds', interpolation='nearest', vmin=0.0, vmax=1.0)
                    plt.axis('off')
                    plt.title(f'"{token}"')
                plt.tight_layout()
                plt.savefig(os.path.join(args.vis_path, file_name + "_attn.png"))
                plt.close()

                # visualize segmentation
                plt.figure(figsize=(plots_W * 2, plots_H * 2))
                for i in range(len(tokens)):
                    token = tokens[i]
                    attn = attns[i]
                    attn_text = attns_text[i]
                    mask = masks[i]
                    score = scores[i]
                    vis_mask = image_np.copy()
                    vis_mask[mask.astype(bool)] = (255, 0, 0)
                    plt.subplot(plots_H, plots_W, i + 1)
                    plt.imshow(vis_mask)
                    plt.axis('off')
                    # plt.title(f'"{token}" {score:.2f} {attn_text:.2f} {score * attn_text:.2f}')
                    plt.title(f'"{token}" {score * attn_text:.2f}')
                plt.tight_layout()
                plt.savefig(os.path.join(args.vis_path, file_name + "_seg.png"))
                plt.close()

            if args.plot_final:
                spacy_model = spacy.load("en_core_web_lg")
                response, token_start_char, token_end_char = decode_token_seq(save_dict['sequences'][1:], tokenizer)
                noun_phrases, phrase_start_char, phrase_end_char = parse_response(response, spacy_model)
                phrase_token = associate_phrase_token(phrase_start_char, phrase_end_char, token_start_char, token_end_char)
                phrase_token = [phrase_token[4], phrase_token[7], phrase_token[12]]
                vis_mask = image_np.copy()
                colors = []
                for i in range(len(phrase_token)):
                    best_tokens = phrase_token[i]
                    best_mask = np.zeros((H, W), dtype=np.uint8)
                    best_mask_score = 0.0
                    best_token_idx = -1
                    for token_idx in best_tokens:
                        mask = masks[token_idx]
                        score = scores[token_idx]
                        if score > best_mask_score:
                            best_mask = mask
                            best_mask_score = score
                            best_token_idx = token_idx

                    hue = int(180 / len(phrase_token) * i)
                    hsv_color = np.uint8([[[hue, 255, 255]]])
                    color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
                    colors.append(color)

                    kernel = np.ones((3, 3), np.uint8)
                    boundary = cv2.dilate(best_mask.astype(np.uint8), kernel, iterations=1) - cv2.erode(best_mask.astype(np.uint8), kernel, iterations=1)
                    best_mask = best_mask.astype(bool)
                    boundary = boundary.astype(bool)
                    vis_mask[best_mask] = (1 - args.alpha) * vis_mask[best_mask] + args.alpha * color
                    vis_mask[boundary] = np.array([255, 255, 255])

                vis_mask = cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.vis_path, file_name + "_final.jpg"), vis_mask)

                for i in range(len(phrase_token)):
                    print(f"Phrase {i}:", noun_phrases[i].text, colors[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vis-path", type=str, default=None)
    parser.add_argument('--seq-start', type=int, default=35)
    parser.add_argument('--seq-end', type=int, default=35+576)
    parser.add_argument('--attn-h', type=int, default=24)
    parser.add_argument('--attn-w', type=int, default=24)
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='save/sam_checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--attn-temp', type=float, default=0.0002)
    parser.add_argument('--attn-box-thr', type=float, default=0.75)
    parser.add_argument('--plot-token', action='store_true')
    parser.add_argument('--plot-token-compact', action='store_true')
    parser.add_argument('--plot-final', action='store_true')
    parser.add_argument('--select-token', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
