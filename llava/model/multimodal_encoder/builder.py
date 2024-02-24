import os
from .clip_encoder import CLIPVisionTower
from .sd_encoder import SDVisionTower, SDMSVisionTower
from .dinov2_encoder import DINOv2VisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("stabilityai") or vision_tower.startswith("runwayml"):
        ms = getattr(vision_tower_cfg, 'mm_vision_ms', False)
        # print(f"Using SDMSVisionTower: {ms}")
        if ms:
            return SDMSVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SDVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("facebookresearch/dinov2"):
        return DINOv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
