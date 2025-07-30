import torch
from vllm.attention.ops.triton_unified_attention import unified_attention

device = "cuda"
torch.manual_seed(0)

# === Create 2 streams ===
stream_prefill = torch.cuda.Stream(device=device)
stream_decode = torch.cuda.Stream(device=device)

# === Common Config ===
#B = 1
#H_Q = 32
#H_KV = 8
#D = 64
#T_Q = 64
T_D = 1
T_KV_PADDED = 256
dtype = torch.float16
B, H_Q, T_Q, D = 1, 4, 128, 64  # Batch, query heads, seq len, head dim
H_KV = 2  # Number of KV heads (must divide H_Q)
# === Prefill tensors ===
q1 = torch.randn(B * T_Q, H_Q, D, device=device, dtype=dtype)
k1 = torch.randn(B, T_Q, H_KV, D, device=device, dtype=dtype)
v1 = torch.randn(B, T_Q, H_KV, D, device=device, dtype=dtype)
out1 = torch.empty_like(q1)

cu_seqlens_q1 = torch.tensor([0, T_Q], dtype=torch.int32, device=device)
seqused_k1 = torch.tensor([T_Q], dtype=torch.int32, device=device)

# === Decode tensors ===
q2 = torch.randn(B * T_D, H_Q, D, device=device, dtype=dtype)
k2 = torch.zeros(B, T_KV_PADDED, H_KV, D, device=device, dtype=dtype)
v2 = torch.zeros(B, T_KV_PADDED, H_KV, D, device=device, dtype=dtype)

# Fill first (T_Q + T_D) with real data
k2[:, :T_Q + T_D] = torch.randn(B, T_Q + T_D, H_KV, D, device=device, dtype=dtype)
v2[:, :T_Q + T_D] = torch.randn(B, T_Q + T_D, H_KV, D, device=device, dtype=dtype)

out2 = torch.empty_like(q2)
cu_seqlens_q2 = torch.tensor([0, T_D], dtype=torch.int32, device=device)
seqused_k2 = torch.tensor([T_Q + T_D], dtype=torch.int32, device=device)

# === Scales ===
softmax_scale = 1.0 / (D ** 0.5)

k_scale = torch.tensor(1.0, dtype=torch.float16, device=device)
v_scale = torch.tensor(1.0, dtype=torch.float16, device=device)

# === Optional: Use None or int32 block_table ===
#block_table = None
block_table = torch.zeros((B, 1), dtype=torch.int32, device=device)

# === Launch prefill ===
with torch.cuda.stream(stream_prefill):
    unified_attention(
        q=q1, k=k1, v=v1, out=out1,
        cu_seqlens_q=cu_seqlens_q1,
        max_seqlen_q=T_Q,
        seqused_k=seqused_k1,
        max_seqlen_k=T_Q,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(0, 0),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        alibi_slopes=None,
    )

# === Launch decode ===
with torch.cuda.stream(stream_decode):
    unified_attention(
        q=q2, k=k2, v=v2, out=out2,
        cu_seqlens_q=cu_seqlens_q2,
        max_seqlen_q=T_D,
        seqused_k=seqused_k2,
        max_seqlen_k=T_KV_PADDED,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(0, 0),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        alibi_slopes=None,
    )

# === Sync and print ===
torch.cuda.synchronize()
print("Prefill out:", out1.shape)
print("Decode out:", out2.shape)

