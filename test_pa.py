# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
import argparse

def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str,
    model_dtype: torch.dtype,
    seed: int = 0,
    device: str = "cuda",
):
    """Create KV cache tensors with specified dtype."""
    torch_dtype = torch.half if cache_dtype == "half" else model_dtype
    
    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches = []
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        k_cache.uniform_(-1, 1)
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches = []
    for _ in range(num_layers):
        v_cache = torch.empty(size=v_cache_shape, dtype=torch_dtype, device=device)
        v_cache.uniform_(-1, 1)
        v_caches.append(v_cache)
    return k_caches, v_caches

def asm_V_shuffle(VC):
    """Shuffle V cache for assembly kernel."""
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC

def run_pa_fwd_asm_with_fp16(decode_len=1, batch_size=128, num_heads=(8, 1), head_size=128, block_size=16, kv_seq_len=4097, device="cuda:6"):
    """Run pa_fwd_asm function with float16 kv cache dtype for paged attention.
    
    Args:
        decode_len: Length of the query sequence 
                   (1 = decode mode, >1 = prefill mode)
        batch_size: Number of sequences to process in parallel
        num_heads: Tuple of (num_query_heads, num_kv_heads) for multi-head attention
        head_size: Dimension of each attention head
        block_size: Size of each memory block in the paged attention
        kv_seq_len: Sequence length for KV cache (context length for decode mode)
        device: Device to run on (e.g., "cuda:6")
    """
    
    # Test parameters
    ctx_lens = kv_seq_len  # KV cache sequence length (context length for decode)
    num_seqs = batch_size
    dtype = torch.half  # fp16
    kv_cache_dtype = "half"  # float16
    
    torch.set_default_device(device)
    
    # Create query tensor (shape depends on decode_len)
    if decode_len == 1:
        # Decode mode: single token per sequence
        query = torch.empty_strided(
            (num_seqs, num_heads[0], head_size),
            ((num_heads[0] + 2 * num_heads[1]) * head_size, head_size, 1),
            dtype=dtype,
        )
    else:
        # Prefill mode: multiple tokens per sequence
        query = torch.empty_strided(
            (num_seqs, decode_len, num_heads[0], head_size),
            ((decode_len * num_heads[0] + 2 * num_heads[1]) * head_size, num_heads[0] * head_size, head_size, 1),
            dtype=dtype,
        )
    query.uniform_(-1, 1)
    
    print(f"[INFO] Paged attention configuration:")
    print(f"  Decode length: {decode_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head size: {head_size}")
    print(f"  Block size: {block_size}")
    print(f"  KV cache sequence length: {kv_seq_len}")
    print(f"  Device: {device}")
    
    # Determine if this is decode or prefill mode
    query_length = decode_len
    is_decode_mode = query_length == 1
    is_prefill_mode = query_length > 1
    
    print(f"\n[INFO] Mode detection:")
    print(f"  Query length: {query_length}")
    print(f"  KV cache sequence length: {ctx_lens}")
    print(f"  Is decode mode: {is_decode_mode} (query_length == 1)")
    print(f"  Is prefill mode: {is_prefill_mode} (query_length > 1)")
    
    if is_decode_mode:
        print(f"  ✓ Running in DECODE mode - single token generation")
    elif is_prefill_mode:
        print(f"  ✓ Running in PREFILL mode - processing multiple tokens")
    else:
        print(f"  ⚠ Unknown mode")
    
    # Create sequence lengths
    seq_lens = torch.tensor([ctx_lens for _ in range(num_seqs)], dtype=torch.int)
    
    # Create block tables
    max_num_blocks_per_seq = (ctx_lens + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs
    
    import random
    block_tables_lst = []
    for _ in range(num_seqs):
        block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)]
        block_tables_lst.append(block_table)
    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)
    
    # Create KV caches
    k_caches, v_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_heads[1], head_size, 
        kv_cache_dtype, dtype, 0, device
    )
    k_cache, v_cache = k_caches[0], v_caches[0]
    
    # Convert to fp16 for pa_fwd_asm
    k_cache_fp16 = k_cache.half()
    v_cache_fp16 = v_cache.half()
    query_fp16 = query.half()
    
    # Debug: Print data types to confirm fp16 usage
    print(f"[DEBUG] pa_fwd_asm input dtypes:")
    print(f"  query.dtype: {query_fp16.dtype}")
    print(f"  k_cache.dtype: {k_cache_fp16.dtype}")
    print(f"  v_cache.dtype: {v_cache_fp16.dtype}")
    print(f"  kv_cache_dtype parameter: {kv_cache_dtype}")
    
    # Verify that k_cache and v_cache are actually fp16
    assert k_cache_fp16.dtype == torch.half, f"k_cache should be fp16, got {k_cache_fp16.dtype}"
    assert v_cache_fp16.dtype == torch.half, f"v_cache should be fp16, got {v_cache_fp16.dtype}"
    
    # Run pa_fwd_asm with float16 kv cache
    output = aiter.pa_fwd_asm(
        query_fp16.contiguous(),
        k_cache_fp16,
        asm_V_shuffle(v_cache_fp16),
        block_tables,
        seq_lens,
        max_num_blocks_per_seq,
        K_QScale=None,
        V_QScale=None,
        out_=None,
        high_precision=0,
    )
    output = aiter.pa_fwd_asm(
        query_fp16.contiguous(),
        k_cache_fp16,
        asm_V_shuffle(v_cache_fp16),
        block_tables,
        seq_lens,
        max_num_blocks_per_seq,
        K_QScale=None,
        V_QScale=None,
        out_=None,
        high_precision=0,
    )
    output = aiter.pa_fwd_asm(
        query_fp16.contiguous(),
        k_cache_fp16,
        asm_V_shuffle(v_cache_fp16),
        block_tables,
        seq_lens,
        max_num_blocks_per_seq,
        K_QScale=None,
        V_QScale=None,
        out_=None,
        high_precision=0,
    )
    output = aiter.pa_fwd_asm(
        query_fp16.contiguous(),
        k_cache_fp16,
        asm_V_shuffle(v_cache_fp16),
        block_tables,
        seq_lens,
        max_num_blocks_per_seq,
        K_QScale=None,
        V_QScale=None,
        out_=None,

        high_precision=0,
    )
    print(f"[SUCCESS] pa_fwd_asm completed with fp16 kv cache!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    
    return output

def parse_args():
    """Parse command line arguments for paged attention."""
    parser = argparse.ArgumentParser(description="Run paged attention with pa_fwd_asm (decode or prefill)")
    parser.add_argument("--query-len", type=int, default=1, 
                       help="Query sequence length (1=decode mode, >1=prefill mode, default: 1)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--num-query-heads", type=int, default=8, help="Number of query heads (default: 8)")
    parser.add_argument("--num-kv-heads", type=int, default=1, help="Number of KV heads (default: 1)")
    parser.add_argument("--head-size", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--block-size", type=int, default=16, help="Block size (default: 16)")
    parser.add_argument("--kv-seq-len", type=int, default=4096, help="KV cache sequence length (default: 4097)")
    parser.add_argument("--device", type=str, default="cuda:6", help="Device to run on (default: cuda:6)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run with command line arguments
    output = run_pa_fwd_asm_with_fp16(
        decode_len=args.query_len,
        batch_size=args.batch_size,
        num_heads=(args.num_query_heads, args.num_kv_heads),
        head_size=args.head_size,
        block_size=args.block_size,
        kv_seq_len=args.kv_seq_len,
        device=args.device
    )


    
import torch
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
import argparse
import time
import random

def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str,
    model_dtype: torch.dtype,
    seed: int = 0,
    device: str = "cuda",
):
    """Create KV cache tensors with specified dtype."""
    torch_dtype = torch.half if cache_dtype == "half" else model_dtype
    
    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches = []
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        k_cache.uniform_(-1, 1)
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches = []
    for _ in range(num_layers):
        v_cache = torch.empty(size=v_cache_shape, dtype=torch_dtype, device=device)
        v_cache.uniform_(-1, 1)
        v_caches.append(v_cache)
    return k_caches, v_caches

def asm_V_shuffle(VC):
    """Shuffle V cache for assembly kernel."""
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC

def run_pa_fwd_asm_with_fp16(decode_len=1, batch_size=128, num_heads=(8, 1), head_size=128, block_size=16, kv_seq_len=4097, device="cuda:6"):
    """Run pa_fwd_asm function with float16 kv cache dtype for paged attention.
    
    Args:
        decode_len: Length of the query sequence 
                    (1 = decode mode, >1 = prefill mode)
        batch_size: Number of sequences to process in parallel
        num_heads: Tuple of (num_query_heads, num_kv_heads) for multi-head attention
        head_size: Dimension of each attention head
        block_size: Size of each memory block in the paged attention
        kv_seq_len: Sequence length for KV cache (context length for decode mode)
        device: Device to run on (e.g., "cuda:6")
    """
    
    # Test parameters
    ctx_lens = kv_seq_len
    num_seqs = batch_size
    dtype = torch.half
    kv_cache_dtype = "half"
    
    torch.set_default_device(device)
    
    # Create sequence lengths
    seq_lens = torch.tensor([ctx_lens for _ in range(num_seqs)], dtype=torch.int)
    
    # Determine if this is decode or prefill mode
    query_length = decode_len
    is_decode_mode = query_length == 1
    
    print(f"[INFO] Paged attention configuration:")
    print(f"  Decode length: {decode_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head size: {head_size}")
    print(f"  Block size: {block_size}")
    print(f"  KV cache sequence length: {kv_seq_len}")
    print(f"  Device: {device}")
    
    print("\n[INFO] Generating data for 5 iterations...")

    # Lists to hold all data for 5 iterations
    all_queries = []
    all_k_caches = []
    all_v_caches = []
    all_block_tables = []
    
    # Generate new random data for each of the 5 iterations
    for _ in range(5):
        # Create query tensor
        if is_decode_mode:
            query = torch.empty_strided(
                (num_seqs, num_heads[0], head_size),
                ((num_heads[0] + 2 * num_heads[1]) * head_size, head_size, 1),
                dtype=dtype,
            )
        else:
            query = torch.empty_strided(
                (num_seqs, decode_len, num_heads[0], head_size),
                ((decode_len * num_heads[0] + 2 * num_heads[1]) * head_size, num_heads[0] * head_size, head_size, 1),
                dtype=dtype,
            )
        query.uniform_(-1, 1)

        # Create block tables
        max_num_blocks_per_seq = (ctx_lens + block_size - 1) // block_size
        num_blocks = max_num_blocks_per_seq * num_seqs
        
        block_tables_lst = []
        for _ in range(num_seqs):
            block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)]
            block_tables_lst.append(block_table)
        block_tables = torch.tensor(block_tables_lst, dtype=torch.int)
        
        # Create KV caches
        k_caches, v_caches = kv_cache_factory(
            num_blocks, block_size, 1, num_heads[1], head_size, 
            kv_cache_dtype, dtype, 0, device
        )
        k_cache, v_cache = k_caches[0], v_caches[0]
        
        all_queries.append(query.half().contiguous())
        all_k_caches.append(k_cache.half())
        all_v_caches.append(v_cache.half())
        all_block_tables.append(block_tables)

    print("\n[INFO] Running benchmark with 2 warm-up iterations and 3 timed iterations.")
    
    # Store timing results
    timings = []
    
    # Total iterations: 2 warm-up + 3 timed
    for i in range(5):
        print(f"\n--- Iteration {i+1} ---")
        
        # Get the pre-generated tensors for this iteration
        query_fp16 = all_queries[i]
        k_cache_fp16 = all_k_caches[i]
        v_cache_fp16 = all_v_caches[i]
        block_tables = all_block_tables[i]

        # Synchronize before timing
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Run the kernel
        output = aiter.pa_fwd_asm(
            query_fp16,
            k_cache_fp16,
            asm_V_shuffle(v_cache_fp16),
            block_tables,
            seq_lens,
            max_num_blocks_per_seq,
            K_QScale=None,
            V_QScale=None,
            out_=None,
            high_precision=0,
        )
        
        # Synchronize after timing
        torch.cuda.synchronize()


