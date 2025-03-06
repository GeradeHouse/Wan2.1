# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import numpy as np
from typing import Union, Optional

from .attention import attention as pay_attention


__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)
    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def identify_k(b: float, d: int, N: int):
    """
    This function identifies the index of the intrinsic frequency component in a RoPE-based pre-trained diffusion transformer.

    Args:
        b (`float`): The base frequency for RoPE.
        d (`int`): Dimension of the frequency tensor
        N (`int`): the first observed repetition frame in latent space
    Returns:
        k (`int`): the index of intrinsic frequency component
        N_k (`int`): the period of intrinsic frequency component in latent space
    Example:
        In HunyuanVideo, b=256 and d=16, the repetition occurs approximately 8s (N=48 in latent space).
        k, N_k = identify_k(b=256, d=16, N=48)
        In this case, the intrinsic frequency index k is 4, and the period N_k is 50.
    """
    # Compute the period of each frequency in RoPE according to Eq.(4)
    periods = []
    for j in range(1, d // 2 + 1):
        theta_j = 1.0 / (b ** (2 * (j - 1) / d))
        N_j = round(2 * torch.pi / theta_j)
        periods.append(N_j)
    # Identify the intrinsic frequency whose period is closed to N (see Eq.(7))
    diffs = [abs(N_j - N) for N_j in periods]
    k = diffs.index(min(diffs)) + 1
    N_k = periods[k - 1]
    return k, N_k


def rope_params_riflex(max_seq_len, dim, theta=10000, L_test=30, k=6):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    inv_theta_pow[k - 1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply_(x, grid_sizes, freqs):
    assert x.shape[0] == 1

    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    f, h, w = grid_sizes[0]
    seq_len = f * h * w
    x_i = x[0, :seq_len, :, :]

    x_i = x_i.to(torch.float32)
    x_i = x_i.reshape(seq_len, n, -1, 2)        
    x_i = torch.view_as_complex(x_i)
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1)
    freqs_i = freqs_i.reshape(seq_len, 1, -1)

    # apply rotary embedding
    x_i *= freqs_i
    x_i = torch.view_as_real(x_i).flatten(2)
    x[0, :seq_len, :, :] = x_i.to(torch.bfloat16)
    return x


def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes):
        seq_len = f * h * w

        x_i = x[i]
        x_i = x_i[:seq_len, :, :]

        x_i = x_i.to(torch.float32)
        x_i = x_i.reshape(seq_len, n, -1, 2)        
        x_i = torch.view_as_complex(x_i)
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i *= freqs_i
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = x_i.to(torch.bfloat16)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x (Tensor): Shape [B, L, C]
        """
        y = x.float()
        y.pow_(2)
        y = y.mean(dim=-1, keepdim=True)
        y += self.eps
        y.rsqrt_()
        x *= y
        x *= self.weight
        return x

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def my_LayerNorm(norm, x):
    y = x.float()
    y_m = y.mean(dim=-1, keepdim=True)
    y -= y_m 
    del y_m
    y.pow_(2)
    y = y.mean(dim=-1, keepdim=True)
    y += norm.eps
    y.rsqrt_()
    x = x * y
    return x


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x (Tensor): Shape [B, L, C]
        """
        y = super().forward(x)
        x = y.type_as(x)
        return x


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, xlist, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x (Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens (Tensor): Shape [B]
            grid_sizes (List): List of [F, H, W] for each sample.
            freqs (Tensor): Rotary frequencies, shape [1024, C / num_heads / 2]
        """
        x = xlist[0]
        xlist.clear()
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Compute query, key, value
        q = self.q(x)
        self.norm_q(q)
        q = q.view(b, s, n, d)
        k = self.k(x)
        self.norm_k(k)
        k = k.view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        del x
        rope_apply_(q, grid_sizes, freqs)
        rope_apply_(k, grid_sizes, freqs)
        # Call attention with separate arguments (not a list)
        x = pay_attention(q, k, v, window_size=self.window_size)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, xlist, context, context_lens):
        r"""
        Args:
            x (Tensor): Shape [B, L1, C]
            context (Tensor): Shape [B, L2, C]
            context_lens (Tensor): Shape [B]
        """
        x = xlist[0]
        xlist.clear()
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.q(x)
        del x
        self.norm_q(q)
        q = q.view(b, -1, n, d)
        k = self.k(context)
        self.norm_k(k)
        k = k.view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = pay_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, xlist, context, context_lens):
        r"""
        Args:
            x (Tensor): Shape [B, L1, C]
            context (Tensor): Shape [B, L2, C]
            context_lens (Tensor): Shape [B]
        """
        x = xlist[0]
        xlist.clear()

        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Compute main attention
        q_main = self.q(x)
        del x
        self.norm_q(q_main)
        q_main = q_main.view(b, -1, n, d)
        k = self.k(context)
        self.norm_k(k)
        k = k.view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = pay_attention(q_main, k, v, k_lens=context_lens)

        # Compute image attention using the same query.
        k_img = self.k_img(context_img)
        self.norm_k_img(k_img)
        k_img = k_img.view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        x_img = pay_attention(q_main, k_img, v_img, k_lens=None)
        x = x.flatten(2)
        x_img = x_img.flatten(2)
        x += x_img
        del x_img
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        r"""
        Args:
            x (Tensor): Shape [B, L, C]
            e (Tensor): Shape [B, 6, C]
            seq_lens (Tensor): Shape [B], length of each sequence in batch
            grid_sizes (Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs (Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)
 
        # self-attention
        x_mod = self.norm1(x)
        x_mod *= 1 + e[1]
        x_mod += e[0]
        xlist = [x_mod]
        del x_mod
        y = self.self_attn(xlist, seq_lens, grid_sizes, freqs)
        x.addcmul_(y, e[2])
        del y
        y = self.norm3(x)
        ylist = [y]
        del y
        x += self.cross_attn(ylist, context, context_lens)
        y = self.norm2(x)
        y *= 1 + e[4]
        y += e[3]
        ffn = self.ffn[0]
        gelu = self.ffn[1]
        ffn2 = self.ffn[2]
        y_shape = y.shape
        y = y.view(-1, y_shape[-1])
        chunk_size = int(y_shape[1] / 2.7)
        chunks = torch.split(y, chunk_size)
        for y_chunk in chunks:
            mlp_chunk = ffn(y_chunk)
            mlp_chunk = gelu(mlp_chunk)
            y_chunk[...] = ffn2(mlp_chunk)
            del mlp_chunk 
        y = y.view(y_shape)
        x.addcmul_(y, e[5])
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x (Tensor): Shape [B, L1, C]
            e (Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm(x).to(torch.bfloat16)
        x *= (1 + e[1])
        x += e[0]
        x = self.head(x)
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """
    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """
        super().__init__()
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()
        
        # TeaCache related attributes (for faster inference)
        self.enable_teacache = False
        self.num_steps = 0
        self.rel_l1_thresh = 0.0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual_cond = None
        self.previous_residual_uncond = None
        self.should_calc = True
        self.coefficients = None
        self.cnt = 0

    def get_rope_freqs(self, nb_latent_frames, RIFLEx_k = None):
        dim = self.dim
        num_heads = self.num_heads 
        d = dim // num_heads
        assert (dim % num_heads) == 0 and (d % 2) == 0

        if RIFLEx_k is None:
            temporal_freqs = rope_params(1024, d - 4 * (d // 6))
        else:
            temporal_freqs = rope_params_riflex(1024, dim=d - 4 * (d // 6), L_test=nb_latent_frames, k=RIFLEx_k)
        spatial_freqs1 = rope_params(1024, 2 * (d // 6))
        spatial_freqs2 = rope_params(1024, 2 * (d // 6))
        freqs = torch.cat([temporal_freqs, spatial_freqs1, spatial_freqs2], dim=1)
        return freqs

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None, freqs=None, pipeline=None):
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if freqs.device != device:
            freqs = freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = [list(u.shape[2:]) for u in x]

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).to(torch.bfloat16)

        # context
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=None
        )

        for block in self.blocks:
            if pipeline is not None and getattr(pipeline, '_interrupt', False):
                return [None]
            x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)
