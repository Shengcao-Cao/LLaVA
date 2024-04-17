import argparse
import os
import numpy as np
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
from transformers import TextStreamer, AutoTokenizer
from matplotlib import pyplot as plt


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
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)

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

    outputs = torch.load(args.attn_path, map_location='cpu')
    sequences = outputs['sequences']
    scores = outputs['scores']
    attentions = outputs['attentions']
    tokens = []
    attns = []
    for i in range(sequences.shape[0] - 1):
        token_index = sequences[i + 1].item()
        token = tokenizer._convert_id_to_token(token_index)
        tokens.append(token)
        attn = attentions[i][:, :, args.seq_start:args.seq_end].numpy()
        # attn = attn.reshape(32, 32, args.attn_h, args.attn_w)
        attn = attn.mean(axis=(0, 1)).reshape(args.attn_h, args.attn_w)
        attns.append(attn)

        # # visualize each head and layer
        # n_layers, n_heads, _, _ = attn.shape
        # fig, axs = plt.subplots(n_layers, n_heads, figsize=(n_heads*2, n_layers*2))
        # for p in range(n_layers):
        #     for q in range(n_heads):
        #         ax = axs[p, q]
        #         ax.imshow(attn[p, q], cmap='Reds', interpolation='nearest')
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         if p == 0:
        #             ax.xaxis.set_ticks_position('top')
        #             ax.xaxis.set_label_position('top')
        #             ax.set_xlabel(f'Head {q}')
        #         if q == 0:
        #             ax.set_ylabel(f'Layer {p}', rotation=90)
        # plt.tight_layout()
        # plt.savefig(args.vis_output.replace('.png', f'_{i}.png'))
        # plt.close()
        # print(i, token)

    attns = np.array(attns)
    mean_attn = attns.mean(axis=0)
    # std_attn = attns.std(axis=0)
    # print(attns.shape, mean_attn.shape, std_attn.shape)
    attns = attns - mean_attn
    range_min = attns.min() * 0.0
    range_max = attns.max() * 0.2
    # attns = attns / range_max
    W = 10
    H = (len(tokens) + W - 1) // W
    plt.figure(figsize=(W * 2, H * 2))
    for i in range(len(tokens)):
        token = tokens[i]
        attn = attns[i]
        print(attn.min(), attn.max())
        plt.subplot(H, W, i+1)
        # plt.imshow(attn, cmap='Reds', interpolation='nearest')
        plt.imshow(attn, cmap='Reds', interpolation='nearest', vmin=range_min, vmax=range_max)
        plt.axis('off')
        plt.title(token)
    plt.tight_layout()
    plt.savefig(args.vis_output)
    plt.close()


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
    parser.add_argument("--attn-path", type=str)
    parser.add_argument("--vis-output", type=str)
    parser.add_argument("--seq-start", type=int, default=35)
    parser.add_argument("--seq-end", type=int, default=35+576)
    parser.add_argument("--attn-h", type=int, default=24)
    parser.add_argument("--attn-w", type=int, default=24)
    args = parser.parse_args()
    main(args)
