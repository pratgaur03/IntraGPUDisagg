import argparse
import math, torch, ctypes
from transformers import AutoConfig
from vllm.attention.ops.triton_unified_attention import unified_attention
import triton
from triton.runtime.cache import get_cache_manager  

hip = ctypes.CDLL("libamdhip64.so")
DEVICE        = "cuda"          
MODEL_ID      = "amd/Meta-Llama-3.1-70B-Instruct-FP8-KV"

# ---- model amd quantized llama 3.1 70B--------------------------------
cfg       = AutoConfig.from_pretrained(MODEL_ID)
HEADS_Q   = cfg.num_attention_heads        # 64
HEADS_KV  = cfg.num_key_value_heads        #  8
HEAD_DIM  = cfg.head_dim                   # 128
KV_BLOCK  = 32                             
DTYPE_Q   = torch.float16                  
DTYPE_KV  = torch.float16
assert HEADS_Q % HEADS_KV == 0, "Heads_KV not a multiple of Heads_Q"


def int_to_maskarr(mask_int, length):
    out = []
    for _ in range(length):
        out.append(mask_int & 0xFFFFFFFF)
        mask_int >>= 32
    return out

def stream_with_cu_mask(mask_bits):
    """Return torch.cuda.ExternalStream limited to the given CU mask."""
    hip.hipExtStreamCreateWithCUMask.restype  = ctypes.c_int
    hip.hipExtStreamCreateWithCUMask.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]
    raw_stream = ctypes.c_void_p()
    mask_arr   = (ctypes.c_uint * len(mask_bits))(*mask_bits)
    ret = hip.hipExtStreamCreateWithCUMask(
        ctypes.byref(raw_stream), len(mask_bits), mask_arr
    )
    assert ret == 0, f"HIP err {ret} creating masked stream"
    return torch.cuda.ExternalStream(raw_stream.value)

def make_kv(batch, seqlen, dtype, heads_kv, head_dim, kv_block):
    n_blocks = (seqlen + kv_block - 1)//kv_block
    k = torch.randn(n_blocks, kv_block, heads_kv, head_dim,
                    dtype=dtype, device=DEVICE)
    v = torch.randn_like(k)
    return k, v

def make_q(batch, seqlen, dtype_q, heads_q, head_dim):
    return torch.randn(batch*seqlen, heads_q, head_dim,
                       dtype=dtype_q, device=DEVICE)

def build_cu_seqlens(batch, seqlen):
    return torch.arange(0, (batch+1)*seqlen, seqlen,
                        dtype=torch.int32, device=DEVICE)

def build_block_table(batch, seqlen, kv_block):
    n_blocks = (seqlen + kv_block - 1)//kv_block
    return torch.arange(n_blocks, dtype=torch.int32,
                        device=DEVICE).expand(batch, n_blocks)
def main():
    parser = argparse.ArgumentParser(
        description="Unified-attention benchmark with optional CU masking"
    )
    parser.add_argument("--prefill-batch",  type=int, default=4)
    parser.add_argument("--decode-batch",   type=int, default=200)
    parser.add_argument("--prefill-len",    type=int, default=4096,
                        help="Sequence length for prefill")
    parser.add_argument("--masking", dest="masking", action="store_true",
                        help="Enable CU-mask streams (default)")
    parser.add_argument("--no-masking", dest="masking", action="store_false",
                        help="Disable CU-mask streams")
    parser.set_defaults(masking=True)
    parser.add_argument("--iters", type=int, default=4,
                        help="Number of timed iterations")
    args = parser.parse_args()


    # ---- CU-mask or plain streams -------------------------------------
    if args.masking:
        N_CU       = 304
        N_DECODE   = int(N_CU * 0.30)
        MASK_WORDS = (N_CU + 31) // 32
        decode_mask_int  = (1 << N_DECODE) - 1
        prefill_mask_int = ((1 << N_CU) - 1) ^ decode_mask_int
        mask_bits_decode  = int_to_maskarr(decode_mask_int,  MASK_WORDS)
        mask_bits_prefill = int_to_maskarr(prefill_mask_int, MASK_WORDS)
        prefill_stream = stream_with_cu_mask(mask_bits_prefill)
        decode_stream  = stream_with_cu_mask(mask_bits_decode)
    else:
        prefill_stream = torch.cuda.Stream()
        decode_stream  = torch.cuda.Stream()

    # ---- workload tensors ---------------------------------------------
    ITR = args.iters
    q_prefills=q_decodes=[]  # dummy init to satisfy linter

    q_prefills, k_prefills, v_prefills = [], [], []
    q_decodes,  k_decodes,  v_decodes  = [], [], []

    for _ in range(ITR+1):
        q_prefills.append(
            make_q(args.prefill_batch, args.prefill_len,
                   DTYPE_Q, HEADS_Q, HEAD_DIM)
        )
        k, v = make_kv(args.prefill_batch, args.prefill_len,
                       DTYPE_KV, HEADS_KV, HEAD_DIM, KV_BLOCK)
        k_prefills.append(k); v_prefills.append(v)

        q_decodes.append(
            make_q(args.decode_batch, 1, DTYPE_Q, HEADS_Q, HEAD_DIM)
        )
        k, v = make_kv(args.decode_batch, args.prefill_len,
                       DTYPE_KV, HEADS_KV, HEAD_DIM, KV_BLOCK)
        k_decodes.append(k); v_decodes.append(v)

    cu_seqlens_prefill = build_cu_seqlens(args.prefill_batch, args.prefill_len)
    cu_seqlens_decode  = build_cu_seqlens(args.decode_batch, 1)

    print("cu_seqlens_prefill",cu_seqlens_prefill)

    seq_used_k_prefill = torch.full((args.prefill_batch,), args.prefill_len,
                                    dtype=torch.int32, device=DEVICE)
    seq_used_k_decode  = torch.full((args.decode_batch,),  args.prefill_len,
                                    dtype=torch.int32, device=DEVICE)
    print("seq_used_k_prefill",seq_used_k_prefill)

    block_table_prefill = build_block_table(args.prefill_batch,
                                            args.prefill_len, KV_BLOCK)
    block_table_decode  = build_block_table(args.decode_batch,
                                            args.prefill_len, KV_BLOCK)

    out_prefill = torch.empty_like(q_prefills[0])
    out_decode  = torch.empty_like(q_decodes[0])

    common = dict(softmax_scale=1/HEAD_DIM**0.5,
                  causal=True, window_size=(0,0), softcap=0.0,
                  q_descale=None, k_descale=None, v_descale=None,
                  alibi_slopes=None)

    # ---------- warm-up + timed loop -----------------------------------
    unified_attention(q_prefills[0], k_prefills[0], v_prefills[0], out_prefill,
                      cu_seqlens_prefill, args.prefill_len,
                      seq_used_k_prefill, args.prefill_len,
                      block_table=block_table_prefill, **common)
    torch.cuda.synchronize()

    with torch.cuda.stream(prefill_stream):
        for i in range(1, ITR+1):
            unified_attention(q_prefills[i], k_prefills[i], v_prefills[i],
                              out_prefill,
                              cu_seqlens_prefill, args.prefill_len,
                              seq_used_k_prefill, args.prefill_len,
                              block_table=block_table_prefill, **common)

    with torch.cuda.stream(decode_stream):
        for i in range(1, ITR+1):
            unified_attention(q_decodes[i], k_decodes[i], v_decodes[i],
                              out_decode,
                              cu_seqlens_decode, 1,
                              seq_used_k_decode, args.prefill_len,
                              block_table=block_table_decode, **common)

    prefill_stream.synchronize(); decode_stream.synchronize()
    print("Mean of outputs:",
          out_prefill.mean().item(), out_decode.mean().item())


if __name__ == "__main__":
    main()