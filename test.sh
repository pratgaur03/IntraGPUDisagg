# In the same directory that holds attention.cu and main.cpp
hipcc -O3 -std=c++17 \
      --amdgpu-target=gfx90a,gfx942,gfx1100,gfx1200 \
      $(python - <<'PY'
import torch, sys, pathlib
print(" -I" + str(pathlib.Path(torch.__file__).parent / "include"), end=" ")
print(" -I" + str(pathlib.Path(torch.__file__).parent / "include/torch/csrc/api/include"), end=" ")
print(" -L" + str(pathlib.Path(torch.__file__).parent.parent), end="")
PY) \
      -ltorch -lc10 -ltorch_cpu -ltorch_cuda \
      attention.cu main.cpp -o demo_attn

# run
./demo_attn
