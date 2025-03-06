# riflex_patch.py
import torch
import logging
import torch.distributed as dist
from wan.modules import model as wan_model

def apply_riflex_patch(args, world_size, rank, local_rank):
    """
    Apply the RIFLEx monkey-patch to modify the rotary positional embeddings.
    This function patches the `rope_params` method in the wan_model module to use RIFLEx modifications.
    Additionally, it initializes the distributed process group if running in a multi-GPU environment.

    Args:
        args: Parsed command-line arguments. Expected to have attributes:
              - riflex (bool)
              - riflex_k (int or None)
              - riflex_L_test (int or None)
              - t5_fsdp, dit_fsdp, ulysses_size, ring_size (for distributed configuration)
        world_size: Total number of processes (GPUs) in the distributed run.
        rank: The rank of the current process.
        local_rank: The local GPU index for the current process.

    Returns:
        The original rope_params function from wan_model, in case restoration is needed.
    """
    # --- RIFLEx monkey-patch ---
    if args.riflex:
        def make_get_1d_rotary_pos_embed_riflex(riflex_k, riflex_L_test):
            def get_1d_rotary_pos_embed_riflex(dim, position, theta=10000.0):
                """Modified version of get_1d_rotary_pos_embed with RIFLEx support."""
                assert dim % 2 == 0
                half = dim // 2
                position = position.float()
                freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=position.device).float() / dim))
                
                # RIFLEx modification: Adjust the intrinsic frequency component
                if riflex_k is not None and riflex_L_test is not None:
                    logging.info(f"Applying RIFLEx with k={riflex_k}, L_test={riflex_L_test}")
                    # Use 0.9 to be conservative and keep within a single period
                    freqs[riflex_k - 1] = 0.9 * 2 * torch.pi / riflex_L_test
                
                freqs = torch.outer(position, freqs)
                emb = torch.zeros(position.shape[0], dim, device=position.device)
                emb[:, 0::2] = freqs.cos()
                emb[:, 1::2] = freqs.sin()
                return emb
            
            return get_1d_rotary_pos_embed_riflex

        # Save the original method for restoration if needed
        original_rope_params = wan_model.rope_params
        
        logging.info(f"Monkey-patching rope_params with RIFLEx (k={args.riflex_k}, L_test={args.riflex_L_test})")
        # Replace the rope_params method with one that calls rope_params_riflex with the RIFLEx parameters
        wan_model.rope_params = lambda max_seq_len, dim, theta=10000: wan_model.rope_params_riflex(
            max_seq_len, dim, theta, L_test=args.riflex_L_test, k=args.riflex_k)
    else:
        original_rope_params = None
    # --- End RIFLEx monkey-patch ---

    # Distributed initialization: if running on multiple GPUs, set the CUDA device and init process group.
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        logging.info(f"Initialized distributed process group: rank={rank}, world_size={world_size}")
    else:
        # For single-GPU runs, ensure that distributed flags are not set
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError("t5_fsdp and dit_fsdp are not supported in non-distributed environments.")
        if args.ulysses_size > 1 or args.ring_size > 1:
            raise ValueError("Context parallelism is not supported in non-distributed environments.")
    
    return original_rope_params
