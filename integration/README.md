# SGLang Integration (Part 2) — A100 Only

This integration requires:
- **Ampere GPU (compute capability ≥ 8.0)**: sm_80 (A100), sm_86 (RTX 3090), sm_89 (RTX 4090)
- SGLang ≥ 0.2.13 (pinned commit: `v0.2.13`)
- `sgl_kernel` built from source for sm_80

---

## 1  Build `sgl_kernel` for sm_80

The prebuilt `sgl_kernel` wheels do not always include sm_80 binaries.
Build from source:

```bash
# Inside your Python venv / Colab environment
git clone https://github.com/sgl-project/sglang.git /tmp/sglang_src
cd /tmp/sglang_src && git checkout v0.2.13

# Build just sgl_kernel (no full server needed)
cd sgl-kernel
pip install ninja cmake
TORCH_CUDA_ARCH_LIST="8.0" pip install -e . --no-build-isolation
```

Verify:
```python
import sgl_kernel
print(sgl_kernel.__version__)
```

---

## 2  Install SGLang

```bash
pip install "sglang[all]==0.2.13" --find-links https://flashinfer.ai/whl/cu121/torch2.3/
```

---

## 3  Register the backend

Option A — environment variable (recommended):

```bash
export SGLANG_ATTENTION_BACKEND=split_kv_triton
export SPLIT_KV=8          # optional; default is 8
python -c "import integration.sglang_backend"  # triggers auto-registration
```

Option B — programmatic:

```python
import sys
sys.path.insert(0, "/path/to/triton-flashinfer")
from integration.sglang_backend import register_backend
register_backend()
```

---

## 4  Launch the SGLang server on Colab

```bash
# In a Colab cell (background subprocess)
import subprocess, os

env = os.environ.copy()
env["SGLANG_ATTENTION_BACKEND"] = "split_kv_triton"
env["SPLIT_KV"] = "8"

proc = subprocess.Popen([
    "python", "-m", "sglang.launch_server",
    "--model-path",  "meta-llama/Llama-3.2-1B",
    "--port",        "30000",
    "--tp",          "1",
    "--dtype",       "float16",
    "--mem-fraction-static", "0.80",
], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for "Server is ready" in stdout
import time
for _ in range(60):
    line = proc.stdout.readline().decode()
    if "ready" in line.lower():
        print("Server ready on port 30000")
        break
    time.sleep(1)
```

Ping the server:
```bash
curl http://localhost:30000/health
```

---

## 5  Pinned versions

| Package     | Version  | Note |
|-------------|----------|------|
| sglang      | 0.2.13   | Last tested commit |
| sgl_kernel  | source   | Must build with `TORCH_CUDA_ARCH_LIST=8.0` |
| flashinfer  | 0.1.6    | Used for prefill path |
| torch       | 2.3.0    |      |
| triton      | 2.3.0    |      |

---

## 6  Known issues

- `sgl_kernel` source build requires `ninja` and `cmake ≥ 3.18`.
- On Colab free tier (T4, sm_75) the build succeeds but the server will warn
  about missing sm_80 optimisations; the kernel itself will still run.
- The `bench_e2e.py` script assumes the server is running on `localhost:30000`.
  Change `SERVER_URL` at the top of that file if needed.
