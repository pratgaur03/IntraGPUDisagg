import math
import torch
from itertools import accumulate
from vllm.attention.ops.triton_flash_attention import triton_attention
# from triton_flash_attention import triton_attention

def run_prefill_only(B=1, prefill_len=2048, H_Q=8, H_KV=1, D=128, dtype=torch.bfloat16):
    torch.set_default_device("cuda")
    assert H_Q % H_KV == 0, "num_query_heads must be a multiple of num_kv_heads"

    max_k = max(kv_lens)
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B*prefill_len, H_Q, D, dtype=dtype)

    k = torch.randn(B*prefill_len, H_KV, D, dtype=dtype)
    v = torch.randn_like(k)

    # cumulative seqlens
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")     # [0,1,2,...,B]
    cu_seqlens_k = torch.tensor([0] + list(accumulate(prefill_len)),
                                dtype=torch.int32, device="cuda")               # [0, L0, L0+L1, ...]

    o, _ = triton_attention(
        q, k, v, None,
        cu_seqlens_q,
        cu_seqlens_k,
        B,                 # num of prompts
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
        B,                 
        max_k,
        True,                     # autoregressive mask
        sm_scale,
        None,
        None,
        None,
    )
    print("out shape:", tuple(o.shape))  # (B, H_Q, D)

if __name__ == "__main__":
    run_decode_only()
