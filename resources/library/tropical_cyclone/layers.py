from timm.layers import to_3tuple, DropPath, trunc_normal_
from typing import Optional, Any, Union, List, Tuple
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import numpy as np
import torch


from timm.layers import trunc_normal_
import torch.nn.functional as F
from functools import lru_cache
from einops import rearrange
from torch import nn
import numpy as np
import itertools
import torch
import math



def get_padded_shape(shape, patch_size):
    sh, ps = np.array(shape), np.array(patch_size)
    mod = sh % ps
    return tuple(sh + ps - np.where(mod == 0, ps, mod))


def window_partition_3d(x, window_size):
    """
    Args:to_2tuple
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], C)
    return windows


def window_reverse_3d(windows, window_size, T, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (T * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, T // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    return x


@lru_cache
def get_attn_mask_3d(input_resolution, win_size, shift_size, suppl_attn_mask=None):
    if any(i > 0 for i in shift_size):
        T_, H_, W_ = get_padded_shape(shape=input_resolution, patch_size=win_size)
        img_mask = torch.zeros((1, T_, H_, W_, 1))  # 1 H W 1
        cnt = 0
        for d in slice(-win_size[0]), slice(-win_size[0], -shift_size[0]), slice(-shift_size[0],None):
            for h in slice(-win_size[1]), slice(-win_size[1], -shift_size[1]), slice(-shift_size[1],None):
                for w in slice(-win_size[2]), slice(-win_size[2], -shift_size[2]), slice(-shift_size[2],None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition_3d(img_mask, win_size)  # nW, ws[0]*ws[1]*ws[2], 1
        mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn_mask = attn_mask + suppl_attn_mask if suppl_attn_mask is not None else attn_mask
    else:
        # attn_mask = None
        attn_mask = suppl_attn_mask if suppl_attn_mask is not None else None
    return attn_mask



def padding(x: torch.Tensor, 
            rs: tuple[int, int, int], # reference shape to pad from x
            ps: tuple[int, int, int], # patch size wrt pad the reference shape
            crop: bool=False, 
            ch_last: bool=True):
    if len(x.shape) == 5:
        dims1 = (0,4,1,2,3)
        dims2 = (0,2,3,4,1)
    elif len(x.shape) == 4:
        dims1 = (0,3,1,2)
        dims2 = (0,2,3,1)
    s, p = np.array(rs), np.array(ps)
    mod = s % p
    pad_amount = p - np.where(mod == 0, p, mod)
    left_pad, right_pad = np.array([0]*pad_amount.shape[0]), pad_amount
    if crop:
        left_pad, right_pad = -left_pad, -right_pad
    padding = np.stack([left_pad, right_pad], axis=1).astype(np.int16)
    padding = tuple(padding[::-1].flatten())
    if ch_last:
        x = x.permute(dims=dims1)
        x = F.pad(x, padding)
        x = x.permute(dims=dims2)
        return x
    else:
        return F.pad(x, padding)



class SpaceTimeConv(nn.Module):
    def __init__(self, in_features, space_kernel_size, time_kernel_size, temporal_conv=nn.Conv1d, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.spatial_conv = nn.Conv2d(in_features, hidden_features, kernel_size=space_kernel_size, padding='same', bias=bias)
        self.temporal_conv = temporal_conv(hidden_features, out_features, kernel_size=time_kernel_size, bias=bias)

        nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x: torch.Tensor):
        b, _, _, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.spatial_conv(x)
        h, w = x.shape[-2:]
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        x = rearrange(x, "b c t h w -> (b h w) c t")
        x = self.temporal_conv(x)
        x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 time_steps: int, 
                 lead_times: int, 
                 patch_size: Optional[Union[int,Tuple[int],List[int]]]=(1,4,4), 
                 embed_dim: Optional[int]=96, 
                 hidden_dim: Optional[int]=96, 
                 norm_layer: Optional[Any]=None, 
                 use_bias: Optional[bool]=True):
        """ Patch Embedding class similar to Swin-Transformer one.
        Parameters
        ----------
            input_size (Union[int,Tuple[int],List[int]]): size of the input.
            in_channels (int): number of input channels
            patch_size (Optional[Union[int,Tuple[int],List[int]]], optional): Size of the patch for the patch embedding. Defaults to 4.
            embed_dim (Optional[int], optional): Size of the output patch embedding. Defaults to 96.
            norm_layer (Optional[Any], optional): Layer for normalization after the embedding. Defaults to None.

        """
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        time_kernel_size = lead_times-time_steps+1 if time_steps < lead_times else 1
        temporal_conv = nn.Conv1d if time_steps > lead_times else nn.ConvTranspose1d

        self.space_time_conv = SpaceTimeConv(in_features=in_channels, hidden_features=hidden_dim, out_features=hidden_dim, space_kernel_size=4, time_kernel_size=time_kernel_size, temporal_conv=temporal_conv, bias=use_bias)
        self.proj = nn.Conv3d(hidden_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=use_bias)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = padding(x, x.shape[-3:], self.patch_size, False, False)
        x = self.space_time_conv(x)
        x = self.proj(x)
        T, H, W = x.shape[-3:]
        res = (T, H, W)
        x = x.flatten(3).permute(0,2,3,1) # B Ph*Pw C
        _, T, L, _ = x.shape
        x = rearrange(x, "b t l c -> b (t l) c")
        x = self.norm(x)
        x = rearrange(x, "b (t l) c -> b t l c", t=T, l=L)
        return x, res


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Downsample(nn.Module):
    r""" Patch Merging Layer.

    Parameters
    ----------
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """
    def __init__(self, 
                 dim: int, 
                 dim_factor: Optional[int] = 2, 
                 norm_layer: Optional[Union[Any,nn.Module]]=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim) if norm_layer is not None else nn.Identity()
        self.reduction = nn.Linear(in_features=4 * dim, out_features=dim_factor * dim, bias=False)

    def _merge(self, x: torch.Tensor, res: tuple[int, int, int]):
        _, H, W = res
        _, T, L, _ = x.shape
        assert L == H * W, f"Input feature has wrong size. L ({L}) != H ({H}) x W ({W})"
        # reshape the data
        x = rearrange(x, "b t (h w) c -> b t h w c", h=H, w=W)
        # pad if needed
        x = padding(x, res, (1,2,2))
        _, T, H, W, _ = x.shape
        x = rearrange(x, "B T (H h) (W w) C -> B T (H W) (h w C)", 
                      H=H//2, h=2, W=W//2, w=2)
        return x, (T, H//2, W//2)

    def forward(self, inputs:tuple[torch.Tensor, tuple[int, int, int]]):
        x, res = inputs
        # merging phase
        x, res = self._merge(x, res)
        # normalize
        x = self.norm(x)
        # halve the number of channels
        x = self.reduction(x)
        return x, res

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class WindowAttention3D_v1(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = win_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1) * (2 * win_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None, out_attn: bool=False):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).to(x.device)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        at = attn if out_attn else None

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, at


class WindowAttention3D_v2(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, win_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0, 0]):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(3, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_t = torch.arange(-(self.win_size[0] - 1), self.win_size[0], dtype=torch.float32)
        relative_coords_h = torch.arange(-(self.win_size[1] - 1), self.win_size[1], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.win_size[2] - 1), self.win_size[2], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_t,
                            relative_coords_h, 
                            relative_coords_w])).permute(1, 2, 3, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, :, 1] /= (pretrained_window_size[1] - 1)
            relative_coords_table[:, :, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
            relative_coords_table[:, :, :, :, 0] /= (self.win_size[0] - 1)
            relative_coords_table[:, :, :, :, 1] /= (self.win_size[1] - 1)
            relative_coords_table[:, :, :, :, 2] /= (self.win_size[2] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_t = torch.arange(self.win_size[0])
        coords_h = torch.arange(self.win_size[1])
        coords_w = torch.arange(self.win_size[2])
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w]))  # 3, Wt, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wt*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wt*Wh*Ww, Wt*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wt*Wh*Ww, Wt*Wh*Ww, 3
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 2] += self.win_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.win_size[1] - 1) * (2 * self.win_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.win_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, 
                x: torch.Tensor,         # input data
                mask: torch.Tensor=None, # masked attention
                out_attn: bool=False):   # output the attention
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(x.device)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1] * self.win_size[2], self.win_size[0] * self.win_size[1] * self.win_size[2], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn1 = attn.view(B_ // nW, nW, self.num_heads, N, N)
            mask1 = mask.unsqueeze(1).unsqueeze(0)
            attn = attn1 + mask1
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attns = attn if out_attn else None

        attn = self.attn_drop(attn)

        x = (attn @ v)
        attns = (attns, x) if out_attn else None
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attns

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.win_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock3D_v1(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, 
                 dim, 
                 num_heads, win_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 fused_window_process=False, suppl_attn_mask=None, pretrained_window_size=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.suppl_attn_mask = suppl_attn_mask
        assert 0 <= self.shift_size[0] < self.win_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.win_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.win_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D_v1(
            dim, win_size=self.win_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.fused_window_process = fused_window_process

    def forward(self, x: torch.Tensor, res: tuple[int, int, int], out_attn: bool=False):
        _, H, W = res
        B, T, L, C = x.shape
        # if window size is larger than input resolution, we don't partition windows
        if res[0] <= self.win_size[0] and res[1] <= self.win_size[1] and res[2] <= self.win_size[2]:
            shift_size = (0,0,0)
            win_size = res
        else:
            shift_size = self.shift_size
            win_size = self.win_size
        assert L == H * W, f"Input feature has wrong size. L ({L}) != H ({H}) x W ({W})"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)

        # pad
        x = padding(x, res, win_size)
        # get new data shape
        B, T, H, W, C = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_3d(shifted_x, win_size)  # B*nW, Wd*Wh*Ww, C

        # W-MSA/SW-MSA
        attn_mask = get_attn_mask_3d(res, tuple(win_size), shift_size, None)
        attn_windows, attn = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, win_size[0], win_size[1], win_size[2], C)
        shifted_x = window_reverse_3d(attn_windows, win_size, T, H, W)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # cropping
        x = padding(x, res, win_size, True)
        # get new data shape
        B, T, H, W, C = x.shape

        x = x.view(B, T, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, (T, H, W), attn


class SwinTransformerBlock3D_v2(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """
    def __init__(self, 
                 dim, 
                 num_heads, win_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, 
                 suppl_attn_mask=None):
        super().__init__()
        self.channels = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.suppl_attn_mask = suppl_attn_mask
        assert 0 <= self.shift_size[0] < self.win_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.win_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.win_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D_v2(
            dim, win_size=self.win_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_3tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, res: tuple[int, int, int], out_attn: bool=False):
        _, H, W = res
        B, T, L, C = x.shape
        # if window size is larger than input resolution, we don't partition windows
        if res[0] <= self.win_size[0] and res[1] <= self.win_size[1] and res[2] <= self.win_size[2]:
            shift_size = (0,0,0)
            win_size = res
        else:
            shift_size = self.shift_size
            win_size = self.win_size
        assert L == H * W, f"input feature has wrong size. L ({L}) != H ({H}) * W ({W})"

        shortcut = x
        x = x.view(B, T, H, W, C)

        # pad
        x = padding(x, res, self.win_size)
        # get new data shape
        B, T, H, W, C = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_3d(shifted_x, win_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, win_size[0] * win_size[1] * win_size[2], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_mask = get_attn_mask_3d(res, tuple(win_size), shift_size, None)
        attn_windows, attn = self.attn(x_windows, mask=attn_mask, out_attn=out_attn)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, win_size[0], win_size[1], win_size[2], C)
        shifted_x = window_reverse_3d(attn_windows, win_size, T, H, W)  # B H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # cropping
        x = padding(x, res, win_size, True)
        # get new data shape
        B, T, H, W, C = x.shape

        x = x.view(B, T, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x, (T, H, W), attn

    def extra_repr(self) -> str:
        return f"dim={self.channels}, num_heads={self.num_heads}, " \
               f"window_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"



class BasicLayer(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 num_heads, 
                 win_size, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0., 
                 norm_layer=nn.LayerNorm, 
                 pretrained_window_size=0, 
                 suppl_attn_mask=None, 
                 version=1):
        super().__init__()
        self.dim = dim
        self.depth = depth

        shift_sizes = [(0,0,0) if (i % 2 == 0) else (win_size[0] // 2, win_size[1] // 2, win_size[2] // 2) for i in range(depth)]
        suppl_attn_masks = [suppl_attn_mask for i in range(depth)]

        if version == 1:
            swin_transformer_block = SwinTransformerBlock3D_v1
        elif version == 2:
            swin_transformer_block = SwinTransformerBlock3D_v2

        # Construct basic blocks
        self.blocks = nn.Sequential(*[
            swin_transformer_block(dim=dim, 
                                   num_heads=num_heads, 
                                   win_size=win_size,
                                   shift_size=shift_sizes[i], 
                                   mlp_ratio=mlp_ratio, 
                                   qkv_bias=qkv_bias,
                                   drop=drop, 
                                   attn_drop=attn_drop, 
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer, 
                                   pretrained_window_size=pretrained_window_size, 
                                   suppl_attn_mask=suppl_attn_masks[i])
            for i in range(depth)])

    def forward(self, x: torch.Tensor, res: tuple[int, int, int], out_attn: bool=False):
        attns = []
        for blk in self.blocks:
            x, res, attn = blk(x, res, out_attn)
            attns.append(attn)
        return x, res, attns

    def _init_respostnorm(self):
        for blk in self.blocks:
            try:
                nn.init.constant_(blk.norm1.bias, 0)
                nn.init.constant_(blk.norm1.weight, 0)
                nn.init.constant_(blk.norm2.bias, 0)
                nn.init.constant_(blk.norm2.weight, 0)
            except:
                pass
