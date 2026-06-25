"""Imports every dependency, prints versions, exits non-zero on failure."""
import sys

REQUIRED = [
    ("torch",        "torch"),
    ("flashinfer",   "flashinfer"),
    ("transformers", "transformers"),
    ("numpy",        "numpy"),
    ("scipy",        "scipy"),
    ("pandas",       "pandas"),
    ("matplotlib",   "matplotlib"),
]

errors = []
rows = []

for friendly, import_name in REQUIRED:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "unknown")
        rows.append((friendly, ver, "OK"))
    except ImportError as e:
        rows.append((friendly, "—", f"MISSING: {e}"))
        errors.append(friendly)

# Print as a table
col_w = max(len(r[0]) for r in rows) + 2
print(f"\n{'Package':<{col_w}}{'Version':<16}Status")
print("-" * (col_w + 30))
for name, ver, status in rows:
    print(f"{name:<{col_w}}{ver:<16}{status}")
print()

# CUDA check
try:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA device : {props.name}  (sm_{props.major}{props.minor})")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available; kernels will not run")
except Exception as e:
    print(f"WARNING: torch CUDA check failed: {e}")

if errors:
    print(f"\nFAIL: missing packages: {', '.join(errors)}")
    sys.exit(1)

print("All dependencies verified OK.")
