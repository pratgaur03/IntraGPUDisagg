import argparse, torch, torch.nn as nn, ctypes
import argparse
import math, torch, ctypes
from transformers import AutoConfig
# from vllm.attention.ops.triton_unified_attention import unified_attention
from triton_unified_attention_2d import unified_attention
import triton
from triton.runtime.cache import get_cache_manager  

hip = ctypes.CDLL("libamdhip64.so")
DEVICE        = "cuda"          
MODEL_ID      = "amd/Meta-Llama-3.1-70B-Instruct-FP8-KV"

# ---- model amd quantized llama 3.1 70B--------------------------------
cfg       = AutoConfig.from_pretrained(MODEL_ID)

H = cfg.hidden_size        # 8192 
I = cfg.intermediate_size  # 28k

gate_up = nn.Linear(H, 2*I, bias=False, device=DEVICE, dtype=torch.float16)
down    = nn.Linear(2*I, H, bias=False, device=DEVICE, dtype=torch.float16)

@torch.no_grad()
def ffn_no_act(x):
    y = gate_up(x)         # [N_tokens, 2I]
    return down(y)         # [N_tokens, H]

# ------------ helper -------------------------------------------------
def build_tokens(batch, seqlen):
    """Return a [batch*seqlen, H] tensor."""
    return torch.randn(batch * seqlen, H, device=DEVICE, dtype=torch.float16)

# ------------ (optional) CU-masked streams ---------------------------
def int_to_maskarr(mask_int, words):
    return [(mask_int >> (32*i)) & 0xFFFFFFFF for i in range(words)]

def stream_with_cu_mask(mask_bits):
    hip.hipExtStreamCreateWithCUMask.restype  = ctypes.c_int
    hip.hipExtStreamCreateWithCUMask.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]
    raw = ctypes.c_void_p()
    arr = (ctypes.c_uint * len(mask_bits))(*mask_bits)
    ret = hip.hipExtStreamCreateWithCUMask(ctypes.byref(raw), len(mask_bits), arr)
    assert ret == 0
    return torch.cuda.ExternalStream(raw.value)

# ------------ CLI & benchmark loop -----------------------------------
def main():
    ap = argparse.ArgumentParser("FFN prefill vs decode")
    ap.add_argument("--prefill-batch", type=int, default=4)
    ap.add_argument("--prefill-len",  type=int, default=4096)
    ap.add_argument("--decode-batch", type=int, default=200)
    ap.add_argument("--iters",        type=int, default=5)
    ap.add_argument("--masking",      action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decode-mask",  type=int, default=96)
    args = ap.parse_args()

    # Streams ----------------------------------------------------------
    if args.masking:
        N_CU = 304
        decode_mask  = (1 << args.decode_mask) - 1
        prefill_mask = ((1 << N_CU) - 1) ^ decode_mask
        words = (N_CU + 31) // 32
        decode_stream  = stream_with_cu_mask(int_to_maskarr(decode_mask,  words))
        prefill_stream = stream_with_cu_mask(int_to_maskarr(prefill_mask, words))
    else:
        decode_stream, prefill_stream = torch.cuda.Stream(), torch.cuda.Stream()

    # Build token-major inputs once per iteration ----------------------
    xs_prefill = [build_tokens(args.prefill_batch, args.prefill_len)  for _ in range(args.iters+1)]
    xs_decode  = [build_tokens(args.decode_batch, 1)                 for _ in range(args.iters+1)]

    # Warm-up + timed loop --------------------------------------------
    for i in range(1, args.iters+1):
        with torch.cuda.stream(decode_stream):   # DECODE PHASE (1-token each)
            _ = ffn_no_act(xs_decode[i])

        with torch.cuda.stream(prefill_stream):  # PREFILL PHASE (full ctx)
            _ = ffn_no_act(xs_prefill[i])

        decode_stream.synchronize(); prefill_stream.synchronize()

if __name__ == "__main__":
    main()
