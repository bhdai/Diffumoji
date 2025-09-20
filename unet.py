import math

import torch
import torch.nn.functional as F
from torch import nn


def Downsample(dim, dim_out=None):
    dim_out = dim_out or dim
    return nn.Conv2d(dim, dim_out, kernel_size=4, stride=2, padding=1)


def Upsample(dim, dim_out=None):
    dim_out = dim_out or dim
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(dim, dim_out, kernel_size=3, padding=1),
    )


class RMSNorm(nn.Module):
    """
    Instead of normalizing by mean or variance, it only use the root mean square
    of the activation across the channel dimension.

    Why it matter? The diffusion model mixes in a lot of condition signals like
    time step embedding text embeddings, etc. by not subtrating the mean RMSNorm
    preserves more more of the original structure while still controlling the scale
    of the activation
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones((1, dim, 1, 1)))  # learnable gain per channel

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class Block(nn.Module):
    """
    A Block is just Conv -> Norm -> Activation, it also handles the feture modulation
    by applying the scale and shift
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.GELU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # ensure the scalel starts centered at 1
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    A ResnetBlock consists of two Blocks with a residual connection.
    it is responsible for creating the scale and shift from the context
    Block1 -> Dropout -> Block2 + Residual
    """

    def __init__(self, dim_in, dim_out, *, context_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.GELU(),
                nn.Linear(context_dim, dim_out * 2),  # each half for scale and shift
            )
            if context_dim is not None
            else None
        )
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.dropout = nn.Dropout(0.1)
        if dim_in == dim_out:
            self.res_conv = nn.Identity()
        else:
            self.res_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x, context=None):
        scale_shift = None

        if self.mlp is not None and context is not None:
            mlp_out = self.mlp(context)
            # reshape mlp's output to be broadcastable with image features
            mlp_out = mlp_out.view(mlp_out.shape[0], mlp_out.shape[1], 1, 1)
            scale, shift = mlp_out.chunk(2, dim=1)
            scale_shift = (scale, shift)

        h = self.block1(x, scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class SinusoidalPosEmb(nn.Module):
    """
    As context is a combination of time embedding and text embedding(already precomputed)
    this class is for time embedding, which is a classic technique borrowed from Transformers.
    It takes a single number the (time step) and turn it into a rich, high dimensional vector
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Unet(nn.Module):
    """
    A U-Net is a model architecture that consists of a downsampling path and an upsampling path
    """

    def __init__(self, dim, condition_dim, dim_mults=(1, 2, 4, 8), channels=3):
        super().__init__()
        self.init_conv = nn.Conv2d(channels, dim, kernel_size=3, padding=1)
        context_dim = dim * 4  # context space
        # define time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # downsampling path
        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList()  # container for downsampling layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, context_dim=context_dim),
                        ResnetBlock(dim_out, dim_out, context_dim=context_dim),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        # upsampling path
        self.ups = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)

            # first block takes double channels when there's a skip connection
            block1_in_dim = dim_in * 2 if not is_last else dim_out
            self.ups.append(
                nn.ModuleList(
                    [
                        Upsample(dim_out, dim_in) if not is_last else nn.Identity(),
                        ResnetBlock(block1_in_dim, dim_in, context_dim=context_dim),
                        ResnetBlock(dim_in, dim_in, context_dim=context_dim),
                    ]
                )
            )
        self.final_conv = nn.Conv2d(dim, 3, kernel_size=1)

    def forward(self, x, time, context):
        time_emb = self.time_mlp(time)
        text_emb = self.condition_mlp(context)
        combined_context = time_emb + text_emb

        x = self.init_conv(x)

        # send x through the downsampling loop, saving skip connections
        skips = []
        for block1, block2, downsample in self.downs:
            x = block1(x, combined_context)
            x = block2(x, combined_context)

            # only save skip connection if this level actually downsample
            if not isinstance(downsample, nn.Identity):
                skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, combined_context)
        x = self.mid_block2(x, combined_context)

        for upsample, block1, block2 in self.ups:
            x = upsample(x)

            # only use skip connections if this level actually upsamples
            if not isinstance(upsample, nn.Identity):
                skip = skips.pop()
                x = torch.cat((x, skip), dim=1)  # concatenate along channel dim

            x = block1(x, combined_context)
            x = block2(x, combined_context)

        return self.final_conv(x)
