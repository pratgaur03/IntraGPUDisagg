# run_decode_only_triton_fa.py
import math
import torch
from itertools import accumulate
# from vllm.attention.ops.triton_flash_attention import triton_attention
from triton_flash_attention import triton_attention

def run_decode_only(B=1, kv_lens=[2048], H_Q=8, H_KV=1, D=128, dtype=torch.bfloat16):
    torch.set_default_device("cuda")
    assert H_Q % H_KV == 0, "num_query_heads must be a multiple of num_kv_heads"

    max_k = max(kv_lens)
    sm_scale = 1.0 / math.sqrt(D)

    # Q: one token per sequence -> total_q = B
    q = torch.randn(B, H_Q, D, dtype=dtype)

    # K/V: concatenation of all sequences' past tokens (varlen format)
    total_k = sum(kv_lens) 
    k = torch.randn(total_k, H_KV, D, dtype=dtype)
    v = torch.randn_like(k)

    # cumulative seqlens
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")     # [0,1,2,...,B]
    cu_seqlens_k = torch.tensor([0] + list(accumulate(kv_lens)),
                                dtype=torch.int32, device="cuda")               # [0, L0, L0+L1, ...]

    o, _ = triton_attention(
        q, k, v, None,
        cu_seqlens_q,
        cu_seqlens_k,
        1,                 # decode: one query per seq
        max_k,
        True,                     # autoregressive mask
        sm_scale,
        None,
        None,
        None,
    )
    torch.cuda.synchronize()
    o, _ = triton_attention(
        q, k, v, None,
        cu_seqlens_q,
        cu_seqlens_k,
        1,                 # decode: one query per seq
        max_k,
        True,                     # autoregressive mask
        sm_scale,
        None,
        None,
        None,
    )
    print("out shape:", tuple(o.shape))  # (B, H_Q, D)

if __name__ == "__main__":
    run_decode_only(B=1, kv_lens=[2048])
