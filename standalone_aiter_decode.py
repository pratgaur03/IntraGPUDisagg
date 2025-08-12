# # run_aiter_decode_bf16.py
# import torch
# from vllm.attention.ops.paged_attn import PagedAttention
# from vllm.utils import cdiv

# def run_decode_bf16(B=8, L=4096, H_Q=8, H_KV=1, HEAD=128, BLOCK=16):
#     print("Hi")
#     torch.set_default_device("cuda")
#     dtype = torch.float16
#     scale = HEAD ** -0.5
#     assert H_Q % H_KV == 0

#     # ---- Shapes ----
#     # query: [B, H_Q, HEAD]  (1 query token per sequence → decode)
#     q = torch.zeros(B, H_Q, HEAD, dtype=dtype)

#     max_seq_len = L
#     num_blocks_per_seq = cdiv(L, BLOCK)
#     num_blocks = B * num_blocks_per_seq  # plenty

#     # KV cache: [num_blocks, BLOCK, H_KV, HEAD]
#     k_cache = torch.zeros(num_blocks, BLOCK, H_KV, HEAD, dtype=dtype)
#     v_cache = torch.zeros_like(k_cache)

#     # block table: [B, max_num_blocks_per_seq] mapping seq → physical blocks
#     bt = torch.empty(B, num_blocks_per_seq, dtype=torch.int32)
#     for b in range(B):
#         start = b * num_blocks_per_seq
#         bt[b] = torch.arange(start, start + num_blocks_per_seq, dtype=torch.int32)

#     # per-seq effective lengths (int32)
#     seq_lens = torch.full((B,), L, dtype=torch.int32)

#     # Call (non-quantized path)
#     out = PagedAttention.forward_decode(
#         q,
#         k_cache,
#         v_cache,
#         bt,
#         seq_lens,
#         max_seq_len,
#         'bf16',      # anything not in {"int8","fp8","fp8_e4m3"} goes here
#         H_KV,
#         scale,
#         None,
#         None, None, # ignored in non-quantized path
#     )
#     torch.cuda.synchronize()
#     print("ok (bf16), out:", tuple(out.shape))


import torch
from vllm.utils import cdiv
from vllm.attention.ops.rocm_aiter_paged_attn import AITERPagedAttention

@torch.inference_mode()
def decode_only_microbench(B=4, L=2048, HQ=8, HK=1, HEAD=128, PAGE=16, kv_dtype="fp8"):
    torch.set_default_device("cuda")
    assert PAGE == 16, "AITer pa_fwd currently requires PAGE (block size) = 16"
    assert HQ % HK == 0
    dtype = torch.bfloat16
    sm_scale = HEAD ** -0.5

    # 1) One-query-token-per-seq (decode)
    q = torch.randn(B, HQ, HEAD, dtype=dtype, device="cuda")

    # 2) Block tables (contiguous block assignment per sequence)
    blocks_per_seq = cdiv(L, PAGE)
    num_blocks = B * blocks_per_seq
    bt = torch.empty(B, blocks_per_seq, dtype=torch.int32, device="cuda")
    for b in range(B):
        start = b * blocks_per_seq
        bt[b] = torch.arange(start, start + blocks_per_seq, dtype=torch.int32, device="cuda")

    # 3) KV cache buffers (uint8 storage → AITer views as FP8 if kv_dtype has "fp8")
    k_cache = torch.empty(num_blocks, PAGE, HK, HEAD, dtype=torch.uint8, device="cuda").random_(0, 255)
    v_cache = torch.empty_like(k_cache).random_(0, 255)

    # 4) Per-slot scales: [H_KV, num_blocks * PAGE]
    num_slots = num_blocks * PAGE
    k_scale = torch.ones(HK, num_slots, dtype=torch.float32, device="cuda")
    v_scale = torch.ones_like(k_scale)

    # 5) Effective decode context lengths
    seq_lens = torch.full((B,), L, dtype=torch.int32, device="cuda")

    # 6) Decode only
    out = AITERPagedAttention.forward_decode(
        query=q,
        key_cache=k_cache,
        value_cache=v_cache,
        block_tables=bt,
        seq_lens=seq_lens,
        max_seq_len=L,
        kv_cache_dtype=kv_dtype,     # "fp8" or "fp8_e4m3"
        num_kv_heads=HK,
        scale=sm_scale,
        alibi_slopes=None,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    print("OK:", tuple(out.shape))

if __name__ == "__main__":
    decode_only_microbench(B=1, L=2048, HQ=8, HK=1, HEAD=128, PAGE=16, kv_dtype="fp8")
