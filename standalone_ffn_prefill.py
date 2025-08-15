import argparse, torch, torch.nn as nn, ctypes
import argparse
import math, torch, ctypes
from transformers import AutoConfig
import triton
from triton.runtime.cache import get_cache_manager  
from torch.cuda import nvtx
hip = ctypes.CDLL("libamdhip64.so")
DEVICE        = "cuda"          
MODEL_ID      = "amd/Meta-Llama-3.1-70B-Instruct-FP8-KV"

# ---- model amd quantized llama 3.1 70B--------------------------------
cfg       = AutoConfig.from_pretrained(MODEL_ID)

H = cfg.hidden_size        # 8192 
I = 3584
# cfg.intermediate_size

gate_up = nn.Linear(H, I, bias=False, device=DEVICE, dtype=torch.float16)
down    = nn.Linear(I, H, bias=False, device=DEVICE, dtype=torch.float16)


@torch.no_grad()
def ffn_down_only(A):
    """
    Simulate ONLY the down projection: y = A @ W_d
    A: [M, I]  ->  y: [M, H]
    """
    assert A.dim() == 2 and A.size(1) == I, f"Expected [*, {I}] input to down proj; got {tuple(A.shape)}"
    y = down(A)
    assert y.size(0) == A.size(0) and y.size(1) == H, f"Down output shape mismatch: got {tuple(y.shape)}"
    return y


# ------------ helper -------------------------------------------------
def build_activations(num_tokens):
    """
    Return a [num_tokens, I] tensor that stands in for the gated activation A.
    """
    return torch.randn(num_tokens, I, device=DEVICE, dtype=torch.float16)

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
    ap.add_argument("--prefill-batch", type=int, default=1)
    ap.add_argument("--prefill-len",  type=int, default=2048)
    ap.add_argument("--decode-batch", type=int, default=10)
    ap.add_argument("--decode-len", type=int, default=2048)
    ap.add_argument("--iters",        type=int, default=5)
    ap.add_argument("--masking",      action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decode-mask",  type=int, default=32)
    args = ap.parse_args()

    # Streams ----------------------------------------------------------
    if args.masking:
        print("Using masking")
        N_CU = 304
        decode_mask  = (1 << args.decode_mask) - 1
        prefill_mask = ((1 << N_CU) - 1) ^ decode_mask
        words = (N_CU + 31) // 32
        
        prefill_stream = stream_with_cu_mask(int_to_maskarr(prefill_mask, words))
        decode_stream  = stream_with_cu_mask(int_to_maskarr(decode_mask,  words))
    else:
        prefill_stream,decode_stream = torch.cuda.Stream(), torch.cuda.Stream()

    M_prefill = args.prefill_batch * args.prefill_len     # tokens processed in prefill step
    M_decode  = args.decode_batch                         # tokens processed in decode step (1 per sequence)

    xs_prefill = [build_activations(M_prefill) for _ in range(args.iters + 1)]
    xs_decode  = [build_activations(M_decode)  for _ in range(args.iters + 1)]

    # Warm-up + timed loop --------------------------------------------
    for i in range(1, args.iters+1):
        with torch.cuda.stream(prefill_stream):  # PREFILL PHASE (full ctx)
            _ = ffn_down_only(xs_prefill[i])
           
        # with torch.cuda.stream(decode_stream):   # DECODE PHASE (1-token each)
        #     _ = ffn_down_only(xs_decode[i])
        decode_stream.synchronize(); prefill_stream.synchronize()

if __name__ == "__main__":
    main()
