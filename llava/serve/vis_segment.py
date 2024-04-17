import argparse
import json
import numpy as np
import os
import cv2

from pycocotools import mask as mask_utils

def mask_to_box(mask):
    x, y = np.where(mask)
    box = [y.min(), x.min(), y.max() + 1, x.max() + 1]
    return np.array(box)

def find_substr(s, sub):
    start = []
    prefix = 0
    while True:
        pos = s.find(sub)
        if pos == -1:
            break
        start.append(prefix + pos)
        prefix += pos + 1
        s = s[pos + 1:]
    return start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str)
    parser.add_argument('--json-folder', type=str)
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    image_files = sorted(os.listdir(args.image_folder))
    # sample N images
    image_files = np.random.choice(image_files, args.N, replace=False)
    # # get first N images
    # image_files = image_files[:args.N]
    json_files = [x[:-4] + '.json' for x in image_files]

    os.makedirs(args.output_folder, exist_ok=True)

    for image_file, json_file in zip(image_files, json_files):
        print(image_file)
        image_path = os.path.join(args.image_folder, image_file)
        json_path = os.path.join(args.json_folder, json_file)
        output_path = os.path.join(args.output_folder, image_file)

        # image = Image.open(image_path).convert('RGB')
        # image = np.array(image)
        image = cv2.imread(image_path)
        # print(image.shape)
        with open(json_path, 'r') as f:
            pred = json.load(f)

        caption = pred['caption']
        phrases = pred['phrases']
        masks = pred['pred_masks']
        masks = [mask_utils.decode(m) for m in masks]
        areas = [m.sum() for m in masks]
        masks, phrases = zip(*sorted(zip(masks, phrases), key=lambda x: x[0].sum(), reverse=True))
        colors = []

        for i in range(len(masks)):
            mask = masks[i]
            # # generate a random RGB color
            # color = np.random.randint(0, 256, 3)
            # generate a bright and saturated color
            hue = int(180 / len(masks) * i)
            hsv_color = np.uint8([[[hue, 255, 255]]])
            color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
            colors.append(color)

            # get boundary of mask
            kernel = np.ones((3, 3), np.uint8)
            boundary = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) - cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

            mask = mask.astype(bool)
            boundary = boundary.astype(bool)
            image[mask] = (1 - args.alpha) * image[mask] + args.alpha * color
            image[boundary] = np.array([255, 255, 255])

        # print a colored caption
        occurence = [find_substr(caption, phrase) for phrase in phrases]
        for i in range(len(phrases)):
            phrase = phrases[i]
            color = colors[i][0, 0]
            starts = occurence[i]
            # print(phrase, color, starts)
            if len(starts) == 0:
                # print('Phrase not found in caption:', phrase)
                continue
            elif len(starts) == 1:
                caption = caption.replace(phrase, f'<span style="color:rgb({color[2]},{color[1]},{color[0]})">{phrase}</span>', 1)
            else:
                # print('Multiple occurences found in caption:', phrase)
                continue
        print(caption)
        print('')

        cv2.imwrite(output_path, image)
