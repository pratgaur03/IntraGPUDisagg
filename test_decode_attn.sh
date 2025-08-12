#!/usr/bin/env bash
set -euo pipefail

# Edit these lists as you like
B_LIST=(1 2)
L_LIST=(2048 4096)

PY=python3
SCRIPT=standalone_triton_decode_attn.py
DEVICE=0  # HIP device index
OUT_DIR="test_decode_attention"
for B in "${B_LIST[@]}"; do
  for L in "${L_LIST[@]}"; do
    TRACE=test_decode_attention_${B}_${L}
    echo "Profiling B=${B} L=${L} -> ${OUT_DIR}/"
    HIP_VISIBLE_DEVICES="${DEVICE}" \
    rocprofv3 \
      --kernel-trace \
      -d "${OUT_DIR}" \
      -o "${TRACE}" \
      -- ${PY} "${SCRIPT}" --B "${B}" --L "${L}" --device "${DEVICE}"
  done
done

