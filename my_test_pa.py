# SPDX-License-Identifier: MIT
# Minimal, fast-path Paged Attention runner
# - Chooses the most efficient kernel automatically
# - Optional per-token FP8 KV-cache quantization
# - No extra test harness or verification

import math
import time
import argparse
import torch

import aiter
from aiter import dtypes
from aiter import paged_attn as ops
from aiter import pertoken_quant
import random
from typing import List, Optional, Tuple, Union
import torch
import aiter
from aiter import dtypes
from aiter import paged_attn as ops
from aiter.test_common import (
    checkAllclose,
    perftest,
    tensor_dump,
    tensor_load,
    benchmark,
)
from aiter import pertoken_quant
import argparse
import pandas as pd
# ---------------------------
# Helpers
# ---------------------------


uniform_range = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": dtypes.bf16,
    "float": dtypes.fp32,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}
ck_naive_quant_algo = [
    "NO",
    "KV_8BIT_PER_HEAD",
    # // FP8/INT8 quant for KVCache, per-token quant
    # // [num_tokens, nhead, hdim] -> [nhead, num_tokens]
    "KV_8BIT_PER_TOKEN",
    # // same as 8bit per token quant but 4 bit
    "KV_4BIT_PER_TOKEN",
    "KV_8BIT_PER_TENSOR",
]

def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            k_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        v_cache = torch.empty(size=v_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            v_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        v_caches.append(v_cache)
    return k_caches, v_caches


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype



@perftest()
def run_aiter(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    k_scale,
    v_scale,
):
    return ops.PagedAttention.forward_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )



@benchmark()
def test_paged_attention(
    ctx_lens: int,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    torch.set_default_device(device)
    # Using default kv_scale
    k_scale = v_scale = torch.tensor(1.0, device=device, dtype=dtypes.fp32)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=dtypes.fp32)
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    max_seq_len = ctx_lens
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs
    # print(f"{debug_mode=}")

    # if debug_mode == VERIFY:
    #     (query, k_cache, v_cache, block_tables, seq_lens, out_golden) = load_input()
    # else:
    query = torch.empty_strided(
        (num_seqs, num_query_heads, head_size),
        ((num_query_heads + 2 * num_kv_heads) * head_size, head_size, 1),
        dtype=dtype,
    )
    query.uniform_(*uniform_range)

    # seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    seq_lens = [ctx_lens for _ in range(num_seqs)]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    block_tables_lst: List[List[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    k_caches, v_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )
    k_cache, v_cache = k_caches[0], v_caches[0]

    out_aiter, time_aiter = run_aiter(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )
    
df = []
l_num_heads = [(8, 1)]
l_ctx_len = [2048, 4097]
l_dtype = ["fp16"]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)

parser.add_argument(
    "-n",
    "--num_heads",
    type=dtypes.str2tuple,
    choices=l_num_heads,
    nargs="?",
    const=None,
    default=None,
    help="""Number of heads (num_query_heads, num_kv_heads)
    e.g.: -n 4,1""",
)

parser.add_argument(
    "-c",
    "--ctx_len",
    type=int,
    choices=l_ctx_len,
    nargs="?",
    const=None,
    default=None,
    help="""Context length.
    e.g. -c 128""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.num_heads is not None:
    l_num_heads = [args.num_heads]
if args.ctx_len is not None:
    l_ctx_len = [args.ctx_len]


for num_heads in l_num_heads:
    for ctx_len in l_ctx_len:
        for dtype in l_dtype:
            ret = test_paged_attention(
                ctx_len, 4, num_heads, 128, False, 16, dtype, "auto", 0, "cuda:0"
            )
            df.append(ret)
df = pd.DataFrame(df)