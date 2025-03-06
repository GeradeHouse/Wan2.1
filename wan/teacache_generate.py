# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                   get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm

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


def t2v_generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    r"""
    Generates video frames from text prompt using diffusion process.
    Args:
        input_prompt (str):
            Text prompt for content generation
        size (tuple[int], optional, defaults to (1280,720)):
            Controls video resolution, (width,height).
        frame_num (int, optional, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
        shift (float, optional, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
        sample_solver (str, optional, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (int, optional, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        guide_scale (float, optional, defaults to 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity
        n_prompt (str, optional, defaults to ""):
            Negative prompt for content exclusion. If not given, use config.sample_neg_prompt
        seed (int, optional, defaults to -1):
            Random seed for noise generation. If -1, use random seed.
        offload_model (bool, optional, defaults to True):
            If True, offloads models to CPU during generation to save VRAM
    Returns:
        torch.Tensor: Generated video frames tensor with dimensions (C, N, H, W)
    """
    # preprocess
    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])
    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len, 'cond_flag': True, 'freqs': None, 'pipeline': self}
        arg_null = {'context': context_null, 'seq_len': seq_len, 'cond_flag': False, 'freqs': None, 'pipeline': self}

        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = [t]
            timestep = torch.stack(timestep)
            self.model.to(self.device)
            noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            temp_x0 = sample_scheduler.step(noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]
        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def i2v_generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 riflex_k=None,
                 riflex_L_test=None):
    r"""
    Generates video frames from input image and text prompt using diffusion process.
    Args:
        input_prompt (str):
            Text prompt for content generation.
        img (PIL.Image.Image):
            Input image tensor. Shape: [3, H, W]
        max_area (int, optional, defaults to 720*1280):
            Maximum pixel area for latent space calculation. Controls video resolution scaling.
        frame_num (int, optional, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1.
        shift (float, optional, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics.
            [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
        sample_solver (str, optional, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (int, optional, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation.
        guide_scale (float, optional, defaults to 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity.
        n_prompt (str, optional, defaults to ""):
            Negative prompt for content exclusion. If not given, use config.sample_neg_prompt.
        seed (int, optional, defaults to -1):
            Random seed for noise generation. If -1, use a random seed.
        offload_model (bool, optional, defaults to True):
            If True, offloads models to CPU during generation to save VRAM.
        riflex_k (int, optional, defaults to None):
            Index for the intrinsic frequency in RoPE for RIFLEx.
        riflex_L_test (int, optional, defaults to None):
            The number of frames for inference for RIFLEx.
    Returns:
        torch.Tensor: Generated video frames tensor with dimensions (C, N, H, W)
    """
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

    # Compute frequencies for RIFLEx if riflex_k is provided
    if riflex_k is None:
        logging.debug("Generating: riflex_k is not provided; RIFLEx will not be applied.")
    else:
        logging.debug(f"Generating: Using riflex_k = {riflex_k}")
    latent_frames = int((F - 1) / self.vae_stride[0] + 1)
    freqs = self.model.get_rope_freqs(nb_latent_frames=latent_frames, RIFLEx_k=riflex_k)
    logging.debug(f"Generating: freqs computed with shape {freqs.shape}")

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        self.vae.model.z_dim, 
        (F - 1) // self.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=self.device)

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    self.clip.model.to(self.device)
    clip_context = self.clip.visual([img[:, None, :, :]])
    if offload_model:
        self.clip.model.cpu()

    y = self.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
            torch.zeros(3, F-1, h, w)
        ], dim=1).to(self.device)
    ])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # sample videos
        latent = noise

        arg_c = {
            'context': [context[0]],
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
            'cond_flag': True,
            'freqs': freqs,
            'pipeline': self
        }

        arg_null = {
            'context': context_null,
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
            'cond_flag': False,
            'freqs': freqs,
            'pipeline': self
        }

        if offload_model:
            torch.cuda.empty_cache()

        self.model.to(self.device)
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(self.device)]
            timestep = [t]
            timestep = torch.stack(timestep).to(self.device)
            noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0].to(torch.device('cpu') if offload_model else self.device)
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0].to(torch.device('cpu') if offload_model else self.device)
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            latent = latent.to(torch.device('cpu') if offload_model else self.device)
            temp_x0 = sample_scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)
            del latent_model_input, timestep

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.vae.decode([latent.to(self.device)])
    del noise, latent
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def teacache_forward(self,
                     x,
                     t,
                     context,
                     seq_len,
                     clip_fea=None,
                     y=None,
                     freqs=None,
                     cond_flag=False):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]): List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor): Diffusion timesteps tensor of shape [B]
        context (List[Tensor]): List of text embeddings each with shape [L, C]
        seq_len (int): Maximum sequence length for positional encoding
        clip_fea (Tensor, optional): CLIP image features for image-to-video mode
        y (List[Tensor], optional): Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]: List of denoised video tensors with original input shapes [C_out, F, H/8, W/8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if freqs is not None and freqs.device != device:
        freqs = freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        context_lens=context_lens)

    if self.enable_teacache:
        if cond_flag:
            modulated_inp = e
            if self.cnt == 0 or self.cnt == self.num_steps - 1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp - self.previous_modulated_input).abs().mean() /
                                                                  self.previous_modulated_input.abs().mean()).cpu().item())
                should_calc = self.accumulated_rel_l1_distance >= self.rel_l1_thresh
                if should_calc:
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp
            self.cnt = 0 if self.cnt == self.num_steps - 1 else self.cnt + 1
            self.should_calc = should_calc
        else:
            should_calc = self.should_calc

    if self.enable_teacache:
        if not should_calc:
            x = x + self.previous_residual_cond if cond_flag else x + self.previous_residual_uncond
            logging.debug("TeaCache Forward: Skipping block computation; using cached residual.")
        else:
            ori_x = x.clone()
            for block in self.blocks:
                x = block(x, **kwargs)
            if cond_flag:
                self.previous_residual_cond = x - ori_x
            else:
                self.previous_residual_uncond = x - ori_x
            logging.debug("TeaCache Forward: Computed new residual and updated cache.")
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]

