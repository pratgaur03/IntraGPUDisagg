import torch
from vllm.attention.ops.triton_unified_attention import unified_attention

# Problem setup
B, H_Q, T_Q, D = 1, 4, 128, 64  # Batch, query heads, seq len, head dim
H_KV = 2  # Number of KV heads (must divide H_Q)
dtype = torch.float16
device = "cuda"

# Construct Q, K, V tensors
q = torch.randn(B * T_Q, H_Q, D, device=device, dtype=dtype)
k = torch.randn(B, T_Q, H_KV, D, device=device, dtype=dtype)
v = torch.randn(B, T_Q, H_KV, D, device=device, dtype=dtype)
# Output tensor
out = torch.empty_like(q)

# Sequence metadata
cu_seqlens_q = torch.tensor([0, T_Q], dtype=torch.int32, device=device)  # Shape [B+1]
seqused_k = torch.tensor([T_Q], dtype=torch.int32, device=device)        # Shape [B]

# Softmax and scale parameters
softmax_scale = 1.0 / (D ** 0.5)
q_descale = None
k_descale = 1.0
v_descale = 1.0
softcap = 0.0
causal = True
window_size = (0, 0)  # No sliding window

# Optional args
block_table = torch.zeros((B, 1), dtype=torch.int32, device=device)  # Dummy
alibi_slopes = None  # Or use actual slopes if needed

# Call kernel
unified_attention(
    q=q,
    k=k,
    v=v,
    out=out,
    cu_seqlens_q=cu_seqlens_q,
    max_seqlen_q=T_Q,
    seqused_k=seqused_k,
    max_seqlen_k=T_Q,
    softmax_scale=softmax_scale,
    causal=causal,
    window_size=window_size,
    block_table=block_table,
    softcap=softcap,
    q_descale=q_descale,
    k_descale=k_descale,
    v_descale=v_descale,
    alibi_slopes=alibi_slopes,
)

# Final shape is same as q: (B*T_Q, H_Q, D)
print("Output shape:", out.shape)

