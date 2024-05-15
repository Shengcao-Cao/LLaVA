import argparse
import cv2
import numpy as np
import os
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

from transformers import TextStreamer
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from segment_anything import SamPredictor, sam_model_registry

from llava.serve.seg_utils import *

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


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

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        qs = inp
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if images is not None:
            # first message
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv.append_message(conv.roles[0], qs)
            images = None
        else:
            # later messages
            conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        print(input_ids)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        if args.vis_path is not None:
            os.makedirs(args.vis_path, exist_ok=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True)

            save_sequences = output_ids["sequences"][0]
            save_scores = torch.cat(output_ids["scores"]).detach().cpu().numpy()
            save_attn = []
            for j in range(len(image_files)):
                save_attn_j = []
                for i in range(len(output_ids["attentions"])):
                    save_attn_i = torch.cat(output_ids["attentions"][i])
                    save_attn_i = save_attn_i[:, :, -1, args.seq_start[j]:args.seq_end[j]].detach().cpu().numpy()
                    save_attn_i = save_attn_i.mean(axis=(0, 1)).reshape(args.attn_h, args.attn_w)
                    save_attn_j.append(save_attn_i)
                save_attn.append(save_attn_j)

            save_attn = np.array(save_attn)
            save_dict = {"sequences": save_sequences, "scores": save_scores, "attentions": save_attn}
            if args.vis_path is not None:
                torch.save(save_dict, os.path.join(args.vis_path, "save.pth"))

        outputs = tokenizer.decode(output_ids["sequences"][0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        if args.vis_path is not None:
            for image_idx in range(len(image_files)):
                file_name = os.path.split(image_files[image_idx])[-1].split('.')[0]
                image_np = cv2.imread(image_files[image_idx])
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

                attns = save_dict['attentions'][image_idx]
                attns = ((attns - attns.mean(axis=0)) / args.attn_temp).astype(np.float32)
                tokens = tokenizer.convert_ids_to_tokens(save_dict['sequences'])

                # visualize attention maps
                plots_W = 5
                plots_H = (len(tokens) + plots_W - 2) // plots_W
                plt.figure(figsize=(plots_W * 2, plots_H * 2))
                for i in range(len(tokens) - 1):
                    token = tokens[i + 1]
                    attn = attns[i]
                    plt.subplot(plots_H, plots_W, i + 1)
                    plt.imshow(attn, cmap='Reds', interpolation='nearest', vmin=0.0, vmax=1.0)
                    plt.axis('off')
                    plt.title(token)
                plt.tight_layout()
                plt.savefig(os.path.join(args.vis_path, file_name + "_attn.png"))
                plt.close()

                # visualize segmentation
                plt.figure(figsize=(plots_W * 2, plots_H * 2))
                for i in range(len(tokens) - 1):
                    token = tokens[i + 1]
                    attn = attns[i]
                    attn = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
                    attn = torch.nn.functional.interpolate(attn, (S, S), mode='bicubic', align_corners=False)
                    attn = attn[0, 0, crop_coor[1]:crop_coor[3], crop_coor[0]:crop_coor[2]]
                    attn = attn.numpy()
                    attn_mask = convert_mask_SAM(attn)
                    attn_box = convert_box_SAM(attn, threshold=args.attn_box_thr)
                    flag = False
                    if attn_box is not None:
                        mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
                        mask = mask.reshape(H, W).astype(np.uint8)
                        score = score.item()
                        if mask.sum() > 0 and score > 0.5:
                            flag = True
                            vis_mask = image_np.copy()
                            vis_mask[mask.astype(bool)] = (255, 0, 0)
                            plt.subplot(plots_H, plots_W, i + 1)
                            plt.imshow(vis_mask)
                            plt.axis('off')
                            plt.title(f'"{token}"{score:.2f}')
                    if not flag:
                        plt.subplot(plots_H, plots_W, i + 1)
                        plt.imshow(image_np)
                        plt.axis('off')
                        plt.title(f'"{token}"')
                plt.tight_layout()
                plt.savefig(os.path.join(args.vis_path, file_name + "_seg.png"))
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vis-path", type=str, default=None)
    parser.add_argument('--seq-start', type=int, nargs='+', default=[35, 52+576-1, 69+576*2-2])
    parser.add_argument('--seq-end', type=int, nargs='+', default=[35+576, 52+576*2-1, 69+576*3-2])
    parser.add_argument('--attn-h', type=int, default=24)
    parser.add_argument('--attn-w', type=int, default=24)
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='save/sam_checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--attn-temp', type=float, default=0.0002)
    parser.add_argument('--attn-box-thr', type=float, default=0.75)
    args = parser.parse_args()
    main(args)
