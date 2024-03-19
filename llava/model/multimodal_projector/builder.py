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


class SDMSCLIPLNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=[32, 16, 8, 8], num_layers=4,
                 clip_proj_in=1024, clip_proj_out=768, append_clip=False, pe=-1):
        # TODO: add option for appending clip features
        # TODO: add option for positional encoding
        super().__init__()
        self.clip_projector = nn.Sequential(
            nn.Linear(clip_proj_in, clip_proj_out),
            nn.GELU(),
            nn.Linear(clip_proj_out, clip_proj_out),
        )
        self.append_clip = append_clip
        if append_clip:
            in_channels = in_channels + [clip_proj_in]
            scale = scale + [16]
            num_layers += 1
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
                    nn.ConvTranspose2d(in_channels[i], out_channels // num_layers, kernel_size=2, stride=2),
                    nn.GELU(),
                )
            elif scale[i] == 16:
                proj_layer_i = nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels // num_layers, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            elif scale[i] == 8:
                proj_layer_i = nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels // num_layers, kernel_size=2, stride=2),
                    nn.GELU(),
                )
            else:
                raise ValueError(f'Unknown scale: {scale[i]}')
            proj_layers.append(proj_layer_i)
        self.proj_layers = nn.ModuleList(proj_layers)
        self.linear = nn.Linear(out_channels // num_layers * num_layers, out_channels)
        if pe > 0:
            self.add_pe = True
            # self.clip_pe = nn.Embedding(pe, clip_proj_out)
            # self.vt_pe = nn.Embedding(pe, out_channels)
            self.clip_pe = nn.Parameter(torch.randn(pe, clip_proj_out))
            self.vt_pe = nn.Parameter(torch.randn(pe, out_channels))
        else:
            self.add_pe = False

    def forward(self, x):
        for i in range(len(self.proj_layers)):
            x[i] = x[i].permute(0, 2, 3, 1)
            x[i] = self.norm_layers[i](x[i])
            x[i] = x[i].permute(0, 3, 1, 2)
            x[i] = self.proj_layers[i](x[i])
            # print(i, x[i].shape, x[i].min(), x[i].max())
        x_cat = torch.cat(x, dim=1)
        b, c, h, w = x_cat.shape
        x_cat = x_cat.view(b, c, h * w).permute(0, 2, 1)
        y = self.linear(x_cat)
        if self.add_pe:
            y = y + self.vt_pe
        return y


class SDMSLNSSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, select_layer=0, scale=[32, 16, 8, 8], num_layers=4):
        super().__init__()
        self.norm_layer = nn.LayerNorm(in_channels[select_layer])
        i = select_layer
        # hard-coded for now, expecting output scale to be H/16 x W/16
        if scale[i] == 32:
            self.proj_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels[i], out_channels // 4, kernel_size=2, stride=2),
                nn.GELU(),
            )
        elif scale[i] == 16:
            self.proj_layer = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels // 4, kernel_size=1, stride=1),
                nn.GELU(),
            )
        elif scale[i] == 8:
            self.proj_layer = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels // 4, kernel_size=2, stride=2),
                nn.GELU(),
            )
        else:
            raise ValueError(f'Unknown scale: {scale[i]}')
        self.linear = nn.Linear(out_channels // 4, out_channels)
        self.select_layer = select_layer

    def forward(self, x):
        x = x[self.select_layer]
        x = x.permute(0, 2, 3, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 3, 1, 2)
        x = self.proj_layer(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        y = self.linear(x)
        return y


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'SDMSBlock':
        return SDMSBlock(config.mm_hidden_size, config.hidden_size, **kwargs)

    if projector_type == 'SDMSLNBlock':
        return SDMSLNBlock(config.mm_hidden_size, config.hidden_size, **kwargs)

    if projector_type == 'SDMSCLIPLNBlock':
        return SDMSCLIPLNBlock(config.mm_hidden_size, config.hidden_size,
                               clip_proj_in=config.mm_vision_clip_proj_in,
                               clip_proj_out=config.mm_vision_clip_proj_out,
                               append_clip=config.mm_vision_append_clip,
                               pe=config.mm_vision_pe,
                               **kwargs)

    if projector_type == 'SDMSLNSSBlock':
        return SDMSLNSSBlock(config.mm_hidden_size, config.hidden_size, config.mm_vision_select_layer, **kwargs)

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
