# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V 14B ------------------------#

i2v_14B = EasyDict(__name__='Config: Wan I2V 14B')
i2v_14B.update(wan_shared_cfg)

i2v_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
i2v_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
i2v_14B.clip_dtype = torch.float16
i2v_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
i2v_14B.clip_tokenizer = 'xlm-roberta-large'

# VAE settings
i2v_14B.vae_checkpoint = 'Wan2.1_VAE.pth'         # Path to the pretrained VAE checkpoint
i2v_14B.vae_stride = (4, 8, 8)                    # Stride values in (time, height, width) for video encoding

# Transformer settings
i2v_14B.patch_size = (1, 2, 2)                    # Size of each patch processed (time x height x width)
i2v_14B.dim = 5120                                 # Dimension of the model’s hidden representations
i2v_14B.ffn_dim = 13824                            # Hidden layer dimension inside the feed-forward network
i2v_14B.freq_dim = 256                             # Size of any frequency-based embedding or position encoding
i2v_14B.num_heads = 40                             # Number of attention heads in self- & cross-attention modules
i2v_14B.num_layers = 40                            # Total number of transformer layers
i2v_14B.window_size = (-1, -1)                     # Window-based attention config; (-1, -1) often means “no window”
i2v_14B.qk_norm = True                             # Normalizes QK (query-key) dot products for improved stability
i2v_14B.cross_attn_norm = True                     # Applies normalization in cross-attention blocks
i2v_14B.eps = 1e-6                                  # Epsilon for numerical stability in layer norm operations

# # Transformer settings (modified for speed over quality)
# i2v_14B.patch_size = (1, 4, 4)   # Larger patches -> fewer patches overall -> faster inference, less detail
# i2v_14B.dim = 3072               # Reduced embedding dimension for smaller model
# i2v_14B.ffn_dim = 8192           # Smaller feed-forward dimension
# i2v_14B.freq_dim = 192           # Smaller frequency embedding size
# i2v_14B.num_heads = 24           # Fewer attention heads
# i2v_14B.num_layers = 24          # Fewer transformer layers
# i2v_14B.window_size = (8, 8)     # Local window-based attention, reduces global overhead
# i2v_14B.qk_norm = False          # Disabling QK normalization can simplify calculations
# i2v_14B.cross_attn_norm = False  # Likewise for cross-attention normalization
# i2v_14B.eps = 1e-5  