import argparse
import numpy as np
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
from pycocotools import mask as mask_utils
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_dummy_rle(h, w):
    mask = np.ones((h, w), dtype=np.uint8)
    mask = np.asfortranarray(mask)
    mask = mask_utils.encode(mask)
    mask['counts'] = mask['counts'].decode('utf-8')
    return mask


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    image_folder = os.path.expanduser(args.image_folder)
    image_files = sorted(os.listdir(image_folder))
    image_files = get_chunk(image_files, args.num_chunks, args.chunk_idx)
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    os.makedirs(args.output_folder, exist_ok=True)
    output_paths = [os.path.join(args.output_folder, f[:-4] + ".json") for f in image_files]
    attn_paths = [os.path.join(args.output_folder, f[:-4] + "_attn.pth") for f in image_files]

    for i in tqdm(range(len(image_files))):
        image_path = image_paths[i]
        output_path = output_paths[i]
        attn_path = attn_paths[i]
        qs = args.question
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

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

        save_sequences = output_ids["sequences"][0]
        # print(save_sequences.shape, save_sequences.dtype)
        # save_scores = torch.cat(output_ids["scores"])
        # print(save_scores.shape, save_scores.dtype)
        save_attn = []
        for i in range(len(output_ids["attentions"])):
            save_attn_i = torch.cat(output_ids["attentions"][i])
            save_attn_i = save_attn_i[:, :, -1, :]
            save_attn_i = save_attn_i.mean(dim=(0, 1))[35:35+576].reshape(24, 24)
            # print(save_attn_i.shape, save_attn_i.dtype)
            save_attn.append(save_attn_i)
        save_dict = {"sequences": save_sequences, "attentions": save_attn}
        torch.save(save_dict, attn_path)

        outputs = tokenizer.decode(output_ids["sequences"][0], skip_special_tokens=True).strip()

        # dummy phrase and mask for now
        w, h = image.size
        phrases = ['']
        pred_masks = [get_dummy_rle(h, w)]

        with open(output_path, "w") as f:
            result_dict = {
                "image_id": image_path[:-4],
                "caption": outputs,
                "phrases": phrases,
                "pred_masks": pred_masks,
            }
            json.dump(result_dict, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--output-folder", type=str, default="")
    parser.add_argument("--question", type=str, default="Describe the image in detail.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
