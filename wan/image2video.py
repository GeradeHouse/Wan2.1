# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm
from IPython import get_ipython

# Added import for TeaCache forward function
from wan.modules.model import sinusoidal_embedding_1d

def get_appropriate_tqdm():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            from tqdm.notebook import tqdm as notebook_tqdm
            return notebook_tqdm
        else:
            from tqdm import tqdm as terminal_tqdm
            return terminal_tqdm
    except NameError:
        from tqdm import tqdm as terminal_tqdm
        return terminal_tqdm

tqdm = get_appropriate_tqdm()

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# --- Begin TeaCache forward function ---
def teacache_forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        freqs=None,
        cond_flag=False,
    ):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x
        freqs (Tensor, optional):
            Rotary embedding frequencies (modified by RIFLEx)
        cond_flag (bool, optional):
            Whether this is a conditional pass
    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    # Ensure the passed-in freqs is on the proper device
    if freqs is not None and freqs.device != device:
        freqs = freqs.to(device)
        logging.debug(f"TeaCache Forward (I2V): Moved freqs to device {device}, shape: {freqs.shape}")

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])
    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context
    ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # Use the passed-in freqs
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,  # Changed: using the passed-in freqs
        context=context,
        context_lens=context_lens)
    
    logging.debug(f"TeaCache Forward (I2V): kwargs constructed with freqs shape: {freqs.shape if freqs is not None else 'None'}")
    
    if self.enable_teacache:
        if cond_flag:
            modulated_inp = e
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                rescale_func = np.poly1d(self.coefficients)
                if cond_flag:
                    self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp 
            self.cnt = 0 if self.cnt == self.num_steps-1 else self.cnt + 1
            self.should_calc = should_calc
        else:
            should_calc = self.should_calc
        
    if self.enable_teacache:
        if not should_calc:
            x = x + self.previous_residual_cond if cond_flag else x + self.previous_residual_uncond
            logging.debug("TeaCache Forward (I2V): Skipping block computation; using cached residual.")
        else:
            ori_x = x.clone()
            for block in self.blocks:
                x = block(x, **kwargs)
            if cond_flag:
                self.previous_residual_cond = x - ori_x
            else:
                self.previous_residual_uncond = x - ori_x
            logging.debug("TeaCache Forward (I2V): Computed new residual and updated cache.")
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

        # head
    x = self.head(x, e)

        # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]
# --- End TeaCache forward function ---


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (str):
                Path to directory containing model checkpoints
            device_id (int, optional): Id of target GPU device (default: 0)
            rank (int, optional): Process rank for distributed training (default: 0)
            t5_fsdp (bool, optional): Enable FSDP sharding for T5 model (default: False)
            dit_fsdp (bool, optional): Enable FSDP sharding for DiT model (default: False)
            use_usp (bool, optional): Enable distribution strategy of USP (default: False)
            t5_cpu (bool, optional): Whether to place T5 model on CPU (default: False)
            init_on_cpu (bool, optional): Enable initializing Transformer Model on CPU (default: True)
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import (usp_attn_forward, usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
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
                 riflex_L_test=None,
                 teacache_thresh=0.05):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (str): Text prompt for content generation.
            img (PIL.Image.Image): Input image tensor. Shape: [3, H, W]
            max_area (int, optional): Maximum pixel area for latent space calculation.
            frame_num (int, optional): Base number of frames to sample from a video (used for VAE input).
            shift (float, optional): Noise schedule shift parameter.
            sample_solver (str, optional): Solver used to sample the video.
            sampling_steps (int, optional): Number of diffusion sampling steps.
            guide_scale (float, optional): Classifier-free guidance scale.
            n_prompt (str, optional): Negative prompt.
            seed (int, optional): Random seed for noise generation.
            offload_model (bool, optional): If True, offloads models to CPU during generation.
            riflex_k (int, optional): Intrinsic frequency index for RIFFLEx modification.
            riflex_L_test (int, optional): Extended (target) number of latent frames for RIFFLEx.
            teacache_thresh (float, optional): Threshold for TeaCache (default: 0.05)
        Returns:
            torch.Tensor: Generated video frames tensor of shape (C, N, H, W)
        """
        # Use the extended frame count if provided via riflex_L_test; otherwise use the base frame_num.
        F = riflex_L_test if riflex_L_test is not None else frame_num
        logging.debug(f"Generating: Using extended frame count F = {F}")
        logging.debug(f"Generating: VAE temporal stride = {self.vae_stride[0]}")
        # Compute the number of latent frames using the VAE temporal stride.
        latent_frames = int((F - 1) / self.vae_stride[0] + 1)
        logging.debug(f"Generating: Computed latent frame count = {latent_frames}")
        if riflex_k is None:
            logging.debug("Generating: riflex_k is not provided; RIFLEx will not be applied.")
        else:
            logging.debug(f"Generating: Using riflex_k = {riflex_k}")

        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        # Use F (extended frame count) here, adjusted by the VAE temporal stride.
        noise = torch.randn(
            16,
            latent_frames,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        # Create mask with F frames, using the VAE temporal stride.
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=self.vae_stride[0], dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // self.vae_stride[0], self.vae_stride[0], lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Preprocess text and clip features
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

        # Check that offload.last_offload_obj exists before calling unload_all()
        from mmgp import offload
        if offload.last_offload_obj is not None:
            offload.last_offload_obj.unload_all()

        # VAE encoding using extended frame count F
        enc = torch.concat([
            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
            torch.zeros(3, F - 1, h, w, device="cpu")
        ], dim=1).to(self.device)
        y = self.vae.encode([enc])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # --- TeaCache integration for image-to-video ---
        self.model.__class__.enable_teacache = True
        self.model.__class__.num_steps = sampling_steps
        self.model.__class__.rel_l1_thresh = teacache_thresh
        self.model.__class__.accumulated_rel_l1_distance = 0
        self.model.__class__.previous_modulated_input = None
        self.model.__class__.previous_residual_cond = None
        self.model.__class__.previous_residual_uncond = None
        self.model.__class__.should_calc = True
        if "480P" in self.config.vae_checkpoint:
            self.model.__class__.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
        elif "720P" in self.config.vae_checkpoint:
            self.model.__class__.coefficients = [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
        self.model.__class__.forward = teacache_forward
        # --- End TeaCache integration ---

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

            latent = noise

            freqs = self.model.get_rope_freqs(nb_latent_frames=latent_frames, RIFLEx_k=riflex_k)
            logging.debug(f"Generating: freqs computed with shape {freqs.shape}")
            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                'freqs': freqs,
                'pipeline': self
            }
            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                'freqs': freqs,
                'pipeline': self
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

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