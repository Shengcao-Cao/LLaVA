import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class SDMSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=[32, 16, 8, 8], num_layers=4):
        super().__init__()
        proj_layers = []
        for i in range(num_layers):
            # hard-coded for now, expecting output scale to be H/16 x W/16
            if scale[i] == 32:
                proj_layer_i = nn.Sequential(
                    nn.ConvTranspose2d(in_channels[i], out_channels // 4, kernel_size=2, stride=2),
                    nn.GELU(),
                )
            elif scale[i] == 16:
                proj_layer_i = nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels // 4, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            elif scale[i] == 8:
                proj_layer_i = nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels // 4, kernel_size=2, stride=2),
                    nn.GELU(),
                )
            else:
                raise ValueError(f'Unknown scale: {scale[i]}')
            proj_layers.append(proj_layer_i)
        self.proj_layers = nn.ModuleList(proj_layers)
        self.linear = nn.Linear(out_channels // 4 * num_layers, out_channels)

    def forward(self, x):
        for i in range(len(self.proj_layers)):
            x[i] = self.proj_layers[i](x[i])
            # print(x[i].shape)
        x_cat = torch.cat(x, dim=1)
        b, c, h, w = x_cat.shape
        x_cat = x_cat.view(b, c, h * w).permute(0, 2, 1)
        y = self.linear(x_cat)
        return y


class SDMSLNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=[32, 16, 8, 8], num_layers=4):
        super().__init__()
        norm_layers = []
        for i in range(num_layers):
            norm_layer_i = nn.LayerNorm(in_channels[i])
            norm_layers.append(norm_layer_i)
        self.norm_layers = nn.ModuleList(norm_layers)
        proj_layers = []
        for i in range(num_layers):
            # hard-coded for now, expecting output scale to be H/16 x W/16
            if scale[i] == 32:
                proj_layer_i = nn.Sequential(
                    nn.ConvTranspose2d(in_channels[i], out_channels // 4, kernel_size=2, stride=2),
                    nn.GELU(),
                )
            elif scale[i] == 16:
                proj_layer_i = nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels // 4, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            elif scale[i] == 8:
                proj_layer_i = nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels // 4, kernel_size=2, stride=2),
                    nn.GELU(),
                )
            else:
                raise ValueError(f'Unknown scale: {scale[i]}')
            proj_layers.append(proj_layer_i)
        self.proj_layers = nn.ModuleList(proj_layers)
        self.linear = nn.Linear(out_channels // 4 * num_layers, out_channels)

    def forward(self, x):
        for i in range(len(self.proj_layers)):
            x[i] = x[i].permute(0, 2, 3, 1)
            x[i] = self.norm_layers[i](x[i])
            x[i] = x[i].permute(0, 3, 1, 2)
            x[i] = self.proj_layers[i](x[i])
            # print(x[i].shape)
        x_cat = torch.cat(x, dim=1)
        b, c, h, w = x_cat.shape
        x_cat = x_cat.view(b, c, h * w).permute(0, 2, 1)
        y = self.linear(x_cat)
        return y


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'SDMSBlock':
        return SDMSBlock(config.mm_hidden_size, config.hidden_size, **kwargs)

    if projector_type == 'SDMSLNBlock':
        return SDMSLNBlock(config.mm_hidden_size, config.hidden_size, **kwargs)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
