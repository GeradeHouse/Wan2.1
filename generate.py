# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import math

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
import torch.nn as nn

from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

# Import TeaCache functions from the dedicated module instead of duplicating code.
from wan.teacache_generate import t2v_generate, i2v_generate, teacache_forward

# Import RIFLEx patch function
from wan.riflex_patch import apply_riflex_patch


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()),
                        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.")
    parser.add_argument("--frame_num", type=int, default=None, help="How many frames to sample from a image or video. The number should be 4n+1")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
    parser.add_argument("--offload_model", type=str2bool, default=None,
                        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated image or video to.")
    parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the image or video from.")
    parser.add_argument("--use_prompt_extend", action="store_true", default=False, help="Whether to use prompt extend.")
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"],
                        help="The prompt extend method to use.")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    parser.add_argument("--prompt_extend_target_lang", type=str, default="ch", choices=["ch", "en"],
                        help="The target language of prompt extend.")
    parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the image or video.")
    parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="The solver used to sample.")
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="Classifier free guidance scale.")
    parser.add_argument("--teacache_thresh", type=float, default=0.05, help="The size of the ulysses parallelism in DiT.")
    # RIFLEx arguments for extending video length.
    parser.add_argument("--riflex", action="store_true", default=False, help="Enable RIFLEx modification for video extrapolation.")
    parser.add_argument("--riflex_k", type=int, default=None, help="Index for the intrinsic frequency in RoPE for RIFLEx.")
    parser.add_argument("--riflex_L_test", type=int, default=None, help="The number of frames for inference for RIFLEx.")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)s: %(message)s",
                            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


# --- The TeaCache-related generation functions are now imported ---
# t2v_generate, i2v_generate, and teacache_forward have been removed from this file.
# They are imported from wan/teacache_generate.py above.

# The remaining classes and functions below remain unchanged.

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

        q_main = self.q(x)
        del x
        self.norm_q(q_main)
        q_main = q_main.view(b, -1, n, d)
        k = self.k(context)
        self.norm_k(k)
        k = k.view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = pay_attention(q_main, k, v, k_lens=context_lens)

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
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
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
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).to(torch.bfloat16)

        # context
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

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


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    # Apply RIFLEx patch if enabled.
    if args.riflex:
        apply_riflex_patch(args, world_size, rank, local_rank)


    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel, init_distributed_environment)
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(), ring_degree=args.ring_size, ulysses_degree=args.ulysses_size)

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl="i2v" in args.task, device=rank)
        else:
            raise NotImplementedError(f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(args.prompt, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed)
                if not prompt_output.status:
                    logging.info(f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        if "t2v" in args.task:
            logging.info("Creating WanT2V pipeline.")
            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )

            # Override the default generation method with the TeaCache version
            wan_t2v.__class__.generate = t2v_generate
            wan_t2v.model.__class__.enable_teacache = True
            wan_t2v.model.__class__.num_steps = args.sample_steps if args.sample_steps is not None else 50
            wan_t2v.model.__class__.rel_l1_thresh = args.teacache_thresh
            wan_t2v.model.__class__.accumulated_rel_l1_distance = 0
            wan_t2v.model.__class__.previous_modulated_input = None
            wan_t2v.model.__class__.previous_residual_cond = None
            wan_t2v.model.__class__.previous_residual_uncond = None
            wan_t2v.model.__class__.should_calc = True
            if '1.3B' in args.ckpt_dir:
                wan_t2v.model.__class__.coefficients = [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
            if '14B' in args.ckpt_dir:
                wan_t2v.model.__class__.coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
            wan_t2v.model.__class__.forward = teacache_forward

            logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} ...")
            video = wan_t2v.generate(
                args.prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)
        else:
            logging.info("Creating WanI2V pipeline.")
            wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )

            # Override the default generation method with the TeaCache version
            wan_i2v.__class__.generate = i2v_generate
            wan_i2v.model.__class__.enable_teacache = True
            wan_i2v.model.__class__.num_steps = args.sample_steps if args.sample_steps is not None else 40
            wan_i2v.model.__class__.rel_l1_thresh = args.teacache_thresh
            wan_i2v.model.__class__.accumulated_rel_l1_distance = 0
            wan_i2v.model.__class__.previous_modulated_input = None
            wan_i2v.model.__class__.previous_residual_cond = None
            wan_i2v.model.__class__.previous_residual_uncond = None
            wan_i2v.model.__class__.should_calc = True
            if '480P' in args.ckpt_dir:
                wan_i2v.model.__class__.coefficients = [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01]
            if '720P' in args.ckpt_dir:
                wan_i2v.model.__class__.coefficients = [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]
            wan_i2v.model.__class__.forward = teacache_forward

            logging.info("Generating video ...")
            video = wan_i2v.generate(
                args.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                riflex_k=args.riflex_k,
                riflex_L_test=args.riflex_L_test)

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(args.prompt, tar_lang=args.prompt_extend_target_lang, image=img, seed=args.base_seed)
                if not prompt_output.status:
                    logging.info(f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        # Override the default generation method with the TeaCache version
        wan_i2v.__class__.generate = i2v_generate
        wan_i2v.model.__class__.enable_teacache = True
        wan_i2v.model.__class__.num_steps = args.sample_steps if args.sample_steps is not None else 40
        wan_i2v.model.__class__.rel_l1_thresh = args.teacache_thresh
        wan_i2v.model.__class__.accumulated_rel_l1_distance = 0
        wan_i2v.model.__class__.previous_modulated_input = None
        wan_i2v.model.__class__.previous_residual_cond = None
        wan_i2v.model.__class__.previous_residual_uncond = None
        wan_i2v.model.__class__.should_calc = True
        if '480P' in args.ckpt_dir:
            wan_i2v.model.__class__.coefficients = [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01]
        if '720P' in args.ckpt_dir:
            wan_i2v.model.__class__.coefficients = [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]
        wan_i2v.model.__class__.forward = teacache_forward

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            riflex_k=args.riflex_k,
            riflex_L_test=args.riflex_L_test)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x')}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
