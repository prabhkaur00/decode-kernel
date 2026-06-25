#!/usr/bin/env bash
# Detects CUDA version, installs matching FlashInfer wheel,
# and prints GPU name + compute capability.
set -euo pipefail

# ── CUDA version detection ─────────────────────────────────────────────────
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/')
else
    # Fall back to nvidia-smi
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | tr -d '.')
    # Keep only major.minor digits, e.g. "12.1" -> "121"
    CUDA_VER=$(echo "$CUDA_VER" | sed 's/\([0-9]*\)\.\([0-9]*\).*/\1\2/')
fi
echo "Detected CUDA: ${CUDA_VER}"

# ── Torch version detection ────────────────────────────────────────────────
TORCH_VER=$(python - <<'EOF'
import torch
v = torch.__version__.split("+")[0]          # strip +cu121 suffix
print(".".join(v.split(".")[:2]))            # major.minor only
EOF
)
echo "Detected torch: ${TORCH_VER}"

# ── FlashInfer installation ────────────────────────────────────────────────
FLASHINFER_INDEX="https://flashinfer.ai/whl/cu${CUDA_VER}/torch${TORCH_VER}/"
echo "FlashInfer index: ${FLASHINFER_INDEX}"
pip install flashinfer==0.1.6 --extra-index-url "${FLASHINFER_INDEX}" --quiet

# ── Remaining deps ─────────────────────────────────────────────────────────
pip install transformers==4.44.0 accelerate==0.32.0 \
            numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 \
            matplotlib==3.9.0 prettytable==3.10.2 --quiet

# ── GPU info ───────────────────────────────────────────────────────────────
python - <<'EOF'
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available")
else:
    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    print(f"GPU: {props.name}")
    print(f"Compute capability: {cc}")
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    if props.major < 7:
        print("WARNING: compute capability < 7.0; fp16 Tensor Core support may be limited")
    if props.major < 8:
        print("NOTE: Part 2 (SGLang) requires compute capability >= 8.0 (Ampere)")
EOF

echo "Setup complete."
