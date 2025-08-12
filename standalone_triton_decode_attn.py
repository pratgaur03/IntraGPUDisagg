import pytest
import torch
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.utils import cdiv


@pytest.mark.parametrize("B", [4, 8, 16])
@pytest.mark.parametrize("L", [2048, 4096])
@pytest.mark.parametrize("H_Q", [8])   # Simulating TP 8
@pytest.mark.parametrize("H_KV", [1])  # Simulating TP 8
@pytest.mark.parametrize("D_QK", [128])
@pytest.mark.parametrize("D_V", [128])
@pytest.mark.parametrize("CACHE_SIZE", [65536])
@pytest.mark.parametrize("PAGE_SIZE", [16])
def test_decode_attention(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L
    N = 5
    sm_scale = 1.0 / (D_QK ** 0.5)
    num_kv_splits = 8

    # Build mapping
    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device="cuda"
    )

    # Tensors
    # q = torch.randn(B, H_Q, D_QK, dtype=dtype, device="cuda")
    q_list = [torch.randn(B, H_Q, D_QK, dtype=dtype, device="cuda") for _ in range(N)]

    k_buffers = [torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device="cuda") for _ in range(N)]
    v_buffers = [torch.randn(CACHE_SIZE, H_KV, D_V,  dtype=dtype, device="cuda") for _ in range(N)]
    k_pages = [kb.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK) for kb in k_buffers]
    v_pages = [vb.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)  for vb in v_buffers]

    # k_buffer = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device="cuda")
    # v_buffer = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device="cuda")
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")
    b_seq_len = torch.full((B,), seq_len, device="cuda")
    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device="cuda"
    )

    # Paged views
    # k_p = k_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
    # v_p = v_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)

    # --- Warm-up (2 iters) ---
    for i in range(2):
        decode_attention_fwd(
            q_list[i], k_pages[i], v_pages[i], o, req_to_page, b_seq_len,
            attn_logits, num_kv_splits, sm_scale, PAGE_SIZE
        )
    torch.cuda.synchronize()

    # --- Normal runs (3 iters) ---
    for i in range(2,5):
        decode_attention_fwd(
            q_list[i], k_pages[i], v_pages[i], o, req_to_page, b_seq_len,
            attn_logits, num_kv_splits, sm_scale, PAGE_SIZE
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--device", default="0", help="HIP_VISIBLE_DEVICES")
    args = ap.parse_args()

    os.environ["HIP_VISIBLE_DEVICES"] = str(args.device)

    test_decode_attention(
        B=args.B, L=args.L, H_Q=8, H_KV=1, D_QK=128, D_V=128,
        CACHE_SIZE=65536, PAGE_SIZE=16,
    )
