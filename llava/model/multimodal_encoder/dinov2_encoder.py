import torch
import torch.nn as nn
from torchvision import transforms

from transformers import CLIPImageProcessor


class DINOv2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.args = args
            self.load_model()
        else:
            self.args = args

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor(size=self.args.mm_vision_resize, crop_size=self.args.mm_vision_crop,
                                                  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.vision_tower_name == 'facebookresearch/dinov2_vits14':
            self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.arch = 's'
        elif self.vision_tower_name == 'facebookresearch/dinov2_vitb14':
            self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.arch = 'b'
        elif self.vision_tower_name == 'facebookresearch/dinov2_vitl14':
            self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.arch = 'l'
        elif self.vision_tower_name == 'facebookresearch/dinov2_vitg14':
            self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            self.arch = 'g'
        else:
            raise ValueError(f'Unexpected vision tower name: {self.vision_tower_name}')

        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.vision_tower.cuda()
        self.vision_tower.eval()

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        image_features = self.vision_tower.forward_features(images)['x_norm_patchtokens']
        # print(image_features.shape)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.norm.weight.dtype

    @property
    def device(self):
        return self.vision_tower.norm.weight.device

    @property
    def config(self):
        return self.args

    @property
    def hidden_size(self):
        # hard-coded for DINOv2
        hidden_size = {
            's': 384,
            'b': 768,
            'l': 1024,
            'g': 1536,
        }[self.arch]
        return hidden_size

    @property
    def num_patches_per_side(self):
        # hard-coded for DINOv2
        patch_size = 14
        return self.args.mm_vision_crop // patch_size

    @property
    def num_patches(self):
        # hard-coded for DINOv2
        patch_size = 14
        return (self.args.mm_vision_crop // patch_size) ** 2
