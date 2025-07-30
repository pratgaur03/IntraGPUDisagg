import math, torch, ctypes
from transformers import AutoConfig
from vllm.attention.ops.triton_unified_attention import unified_attention
import triton
from triton.runtime.cache import get_cache_manager  


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
# ---- Independent Streams
#prefill_stream, decode_stream = torch.cuda.Stream(), torch.cuda.Stream()
# --- CU Masked Stream
# 30% for decode, 70% Prefill
N_CU       = 304
N_DECODE   = int(N_CU * 0.30)                 # 91 CUs for decode
MASK_WORDS = (N_CU + 31) // 32               

decode_mask_int  = (1 << N_DECODE) - 1
prefill_mask_int = ((1 << N_CU) - 1) ^ decode_mask_int

def int_to_maskarr(mask_int, length):
    out = []
    for _ in range(length):
        out.append(mask_int & 0xFFFFFFFF)
        mask_int >>= 32
    return out

mask_bits_decode  = int_to_maskarr(decode_mask_int,  MASK_WORDS)
mask_bits_prefill = int_to_maskarr(prefill_mask_int, MASK_WORDS)

def stream_with_cu_mask(mask_bits):
    """Return torch.cuda.ExternalStream limited to the given CU mask."""
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipExtStreamCreateWithCUMask.restype  = ctypes.c_int
    hip.hipExtStreamCreateWithCUMask.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]
    raw_stream = ctypes.c_void_p()
    mask_arr   = (ctypes.c_uint * len(mask_bits))(*mask_bits)
    ret = hip.hipExtStreamCreateWithCUMask(
        ctypes.byref(raw_stream), len(mask_bits), mask_arr
    )
    assert ret == 0, f"HIP err {ret} creating masked stream"
    return torch.cuda.ExternalStream(raw_stream.value)

prefill_stream = stream_with_cu_mask(mask_bits_prefill)
decode_stream  = stream_with_cu_mask(mask_bits_decode)

print("Decode CU mask :", [hex(x) for x in mask_bits_decode])
print("Prefill CU mask:", [hex(x) for x in mask_bits_prefill])


def make_kv(batch, seqlen, dtype):
    n_blocks = (seqlen + KV_BLOCK - 1)//KV_BLOCK
    k = torch.randn(n_blocks, KV_BLOCK, HEADS_KV, HEAD_DIM,
                    dtype=dtype, device=DEVICE)
    v = torch.randn_like(k)
    return k, v

def make_q(batch, seqlen):
    return torch.randn(batch*seqlen, HEADS_Q, HEAD_DIM,
                       dtype=DTYPE_Q, device=DEVICE)

def build_cu_seqlens(batch, seqlen):
    return torch.arange(0, (batch+1)*seqlen, seqlen,
                        dtype=torch.int32, device=DEVICE)

def build_block_table(batch, seqlen):
    n_blocks = (seqlen + KV_BLOCK - 1)//KV_BLOCK
    return torch.arange(n_blocks, dtype=torch.int32,
                        device=DEVICE).expand(batch, n_blocks)

# ---- workload ----------------------------------------------
BATCH_PREFILL, BATCH_DECODE   = 4, 4
SEQ_LEN_PREFILL               = 128

q_prefill = make_q(BATCH_PREFILL, SEQ_LEN_PREFILL)
k_prefill, v_prefill = make_kv(BATCH_PREFILL, SEQ_LEN_PREFILL, DTYPE_KV)

q_decode  = make_q(BATCH_DECODE, 1)
k_decode, v_decode  = make_kv(BATCH_DECODE, SEQ_LEN_PREFILL, DTYPE_KV)

cu_seqlens_prefill = build_cu_seqlens(BATCH_PREFILL, SEQ_LEN_PREFILL)
cu_seqlens_decode  = build_cu_seqlens(BATCH_DECODE, 1)

seq_used_k_prefill = torch.full((BATCH_PREFILL,), SEQ_LEN_PREFILL,
                                dtype=torch.int32, device=DEVICE)
seq_used_k_decode  = torch.full((BATCH_DECODE,),  SEQ_LEN_PREFILL,
                                dtype=torch.int32, device=DEVICE)

block_table_prefill = build_block_table(BATCH_PREFILL, SEQ_LEN_PREFILL)
block_table_decode  = build_block_table(BATCH_DECODE,  SEQ_LEN_PREFILL)

out_prefill  = torch.empty_like(q_prefill)
out_decode   = torch.empty_like(q_decode)

common = dict(softmax_scale=1/HEAD_DIM**0.5,
              causal=True, window_size=(0,0), softcap=0.0,
              q_descale=None, k_descale=None, v_descale=None,
              alibi_slopes=None)

with torch.cuda.stream(prefill_stream):
    unified_attention(q_prefill, k_prefill, v_prefill, out_prefill,
                      cu_seqlens_prefill, SEQ_LEN_PREFILL,
                      seq_used_k_prefill, SEQ_LEN_PREFILL,
                      block_table=block_table_prefill, **common)

with torch.cuda.stream(decode_stream):
    unified_attention(q_decode, k_decode, v_decode, out_decode,
                      cu_seqlens_decode, 1,
                      seq_used_k_decode, SEQ_LEN_PREFILL,
                      block_table=block_table_decode, **common)

prefill_stream.synchronize(); decode_stream.synchronize()
print("Mean of output for both", out_prefill.mean().item(), out_decode.mean().item())

