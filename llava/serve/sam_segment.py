import argparse
import numpy as np
import torch

from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

def convert_box_SAM(mask, threshold=0.5):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='save/sam_checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--attn-path', type=str)
    parser.add_argument('--vis-output', type=str)
    parser.add_argument('--seq-start', type=int, default=35)
    parser.add_argument('--seq-end', type=int, default=35+576)
    parser.add_argument('--attn-h', type=int, default=24)
    parser.add_argument('--attn-w', type=int, default=24)
    parser.add_argument('--threshold', type=float, default=0.001)
    # parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).to(device=args.device)
    predictor = SamPredictor(sam)

    image = Image.open(args.image_file).convert('RGB')
    W, H = image.size
    S = max(W, H)
    if W == H:
        crop_coor = (0, 0, W, H)
    elif W > H:
        crop_coor = (0, (W - H) // 2, W, (W + H) // 2)
    else:
        crop_coor = ((H - W) // 2, 0, (H + W) // 2, H)

    predictor.set_image(np.array(image))

    outputs = torch.load(args.attn_path, map_location='cpu')
    attentions = outputs['attentions']
    attns = []
    for i in range(len(attentions)):
        attn = attentions[i][:, :, args.seq_start:args.seq_end].numpy()
        attn = attn.mean(axis=(0, 1)).reshape(args.attn_h, args.attn_w)
        attns.append(attn)

    attns = np.array(attns)
    mean_attn = attns.mean(axis=0)
    attns = attns - mean_attn
    # print(attns.min(), attns.max())
    for i in range(len(attentions)):
        # normalize attention map
        attn = attns[i]
        attn = np.clip(attn / args.threshold, 0.0, 1.0).astype(np.float32)
        # resize and crop attention map to image size
        # # using Image
        # attn = Image.fromarray(attn)
        # attn = attn.resize((S, S), Image.BICUBIC)
        # attn = attn.crop(crop_coor)
        # attn = np.array(attn)
        # using torch
        attn = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
        attn = torch.nn.functional.interpolate(attn, (S, S), mode='bicubic', align_corners=False)
        attn = attn[0, 0, crop_coor[1]:crop_coor[3], crop_coor[0]:crop_coor[2]]
        attn = attn.numpy()
        # get score and color
        score = attn[:, :, np.newaxis]
        color = np.stack((attn, np.zeros_like(attn), np.zeros_like(attn)), axis=-1)
        color = np.clip(color * 255, 0, 255).astype(np.uint8)
        image_np = np.array(image)
        # print(score.shape, color.shape, image_np.shape)
        # print(score.max(), color.max(), image_np.max())
        blend = image_np.astype(np.float32) * (1 - score) + color.astype(np.float32) * score
        blend = np.clip(blend, 0, 255).astype(np.uint8)
        blend = Image.fromarray(blend)
        blend.save(args.vis_output.replace('.png', f'_{i}.png'))

        # get attn_mask for SAM
        # attn_mask = attns[i]
        # attn_mask = np.clip(attn_mask / args.threshold, 0.0, 1.0).astype(np.float32)
        # attn_mask = torch.tensor(attn_mask).unsqueeze(0).unsqueeze(0)
        # attn_mask = torch.nn.functional.interpolate(attn_mask, (256, 256), mode='bicubic', align_corners=False)
        # attn_mask = attn_mask[0].numpy() > 0.5
        # attn_mask = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
        # attn_mask = torch.nn.functional.interpolate(attn_mask, (256, 256), mode='bicubic', align_corners=False)
        # attn_mask = attn_mask[0].numpy()
        # print(i, attn_mask.shape, attn_mask.sum(), attn_mask.max())
        attn_mask = convert_mask_SAM(attn)
        attn_box = convert_box_SAM(attn)

        if attn_box is not None:
            # get segmentation mask
            mask, score, logits = predictor.predict(box=attn_box, mask_input=attn_mask, multimask_output=False)
            # mask, score, logits = predictor.predict(point_coords=points, multimask_output=True)
            # print(masks.shape, scores.shape, logits.shape)
            # print(masks.dtype, masks.max(), scores.dtype, scores.max(), logits.dtype, logits.max())
            mask = mask.reshape(H, W, 1).astype(np.float32)
            print(i, mask.sum(), score.item())
            if mask.sum() > 0 and score.item() > 0.5:
                blend = image_np.astype(np.float32) * (1 - mask) + np.array([255, 0, 0], dtype=np.float32) * mask
                blend = np.clip(blend, 0, 255).astype(np.uint8)
                blend = Image.fromarray(blend)
                blend.save(args.vis_output.replace('attn.png', f'seg_{i}.png'))

            # print(i, mask.shape, mask.sum(), score)
            # for j in range(mask.shape[0]):
            #     mask_j = mask[j].reshape(H, W, 1)
            #     if mask_j.sum() > 0:
            #         blend = image_np.astype(np.float32) * (1 - mask_j) + np.array([255, 0, 0], dtype=np.float32) * mask_j
            #         blend = np.clip(blend, 0, 255).astype(np.uint8)
            #         blend = Image.fromarray(blend)
            #         blend.save(args.vis_output.replace('attn.png', f'seg_{i}_{j}.png'))
