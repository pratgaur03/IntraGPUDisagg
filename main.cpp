// main.cpp – Llama‑3 70 B, FP8 KV cache  -------------------------------
#include <torch/torch.h>
#include <iostream>

/*  ───────────  paged_attention forward decl  ───────────
   (leave this as‑is: it is already defined in attention.cu)            */
void paged_attention( torch::Tensor& out,
                      torch::Tensor& exp_sums,
                      torch::Tensor& max_logits,
                      torch::Tensor& tmp_out,
                      torch::Tensor& query,
                      torch::Tensor& key_cache,
                      torch::Tensor& value_cache,
                      int64_t       num_kv_heads,
                      double        scale,
                      torch::Tensor& block_tables,
                      torch::Tensor& context_lens,
                      c10::optional<torch::Tensor> query_start_loc,
                      int64_t       block_size,
                      int64_t       max_context_len,
                      c10::optional<torch::Tensor> alibi_slopes,
                      const std::string& kv_cache_dtype,
                      torch::Tensor& k_scale,
                      torch::Tensor& v_scale,
                      c10::optional<torch::Tensor> fp8_out_scale );

int main() {
    /* ──────  Model constants for Llama‑3 70 B  ────── */
    const int B  = 1;          // decoding one prompt
    const int H  = 64;         // total *query* heads
    const int KV = 8;          // KV heads (group‑query attention ratio 8)
    const int D  = 128;        // dk = dv = 128
    const int L  = 1;          // one new token – single‑step decode
    const int BLOCK = 16;      // vLLM uses 16‑token KV blocks
    const int MAX_CTX = 4096;  // set to your prompt length upper‑bound

    /* ──────  Tensor options  ────── */
    auto dev      = torch::Device(torch::kCUDA, 0);
    auto fp16_opt = torch::TensorOptions().dtype(torch::kHalf    ).device(dev);
    auto fp8_opt  = torch::TensorOptions().dtype(torch::kUInt8   ).device(dev);
    auto f32_opt  = torch::TensorOptions().dtype(torch::kFloat32 ).device(dev);
    auto i32_opt  = torch::TensorOptions().dtype(torch::kInt32   ).device(dev);

    /* ──────  Query for current token  (B, H, D)  ────── */
    torch::Tensor query = torch::randn({B, H, D}, fp16_opt);

    /* ──────  Allocate FP8 KV cache  ──────
       Layout is [blocks, kv_heads, head_size / x, block, x]
       where x = 16 bytes / sizeof(uint8) = 16                               */
    const int x = 16;                                  // bytes per tile
    int blocks_needed = (MAX_CTX + BLOCK - 1) / BLOCK; // worst case
    torch::Tensor key_cache   = torch::empty({blocks_needed, KV, D / x, BLOCK, x}, fp8_opt);
    torch::Tensor value_cache = torch::empty({blocks_needed, KV, D,       BLOCK   }, fp8_opt);

    /* ──────  Per‑block KV scale tensors – 1 scale for demo  ────── */
    torch::Tensor k_scale = torch::full({1}, 1.0f, f32_opt);
    torch::Tensor v_scale = torch::full({1}, 1.0f, f32_opt);

    /* ──────  vLLM book‑keeping    ────── */
    torch::Tensor block_tables = torch::arange(blocks_needed, i32_opt).unsqueeze(0); // (B, max_blocks)
    torch::Tensor context_lens = torch::tensor({0}, i32_opt);   // empty KV at t 0

    /* exp_sums / max_logits / tmp_out scratch */
    int max_parts = (MAX_CTX + 255) / 256;
    torch::Tensor exp_sums   = torch::empty({B, H, max_parts}, f32_opt);
    torch::Tensor max_logits = torch::empty_like(exp_sums);
    torch::Tensor tmp_out    = torch::empty({B, H, max_parts, D}, fp16_opt);
    torch::Tensor out        = torch::empty({B, H, D}, fp16_opt);

    /* ──────  Call the kernel group  ────── */
    paged_attention(out, exp_sums, max_logits, tmp_out,
                    query, key_cache, value_cache,
                    /*num_kv_heads=*/KV,
                    /*scale*/       1.0,
                    block_tables, context_lens,
                    /*query_start_loc*/ c10::nullopt,
                    /*block_size*/  BLOCK,
                    /*max_ctx*/     MAX_CTX,
                    /*alibi*/       c10::nullopt,
                    /*kv_dtype*/    "fp8",          // important!
                    k_scale, v_scale,
                    /*fp8_out_scale*/ c10::nullopt   // keeping output in fp16
    );

    torch::cuda::synchronize();
    std::cout << "decode‑token mean = " << out.mean().item<float>() << '\n';
    return 0;
}
