# %% [markdown]
# # PMRA Hessian-sketch scorer — Kaggle CPU kernel
#
# Runs `scripts/hessian_scorer.py` at 4B on a free Kaggle CPU kernel (30 GB
# RAM) and validates the analytic scores against empirical probe rows.
# The HF model (~8 GB) downloads kernel-side; nothing heavy touches the
# launching machine. `--capture-passes 4` keeps peak RAM ~24 GB at 4B fp32.

# %%
# ===================== CONFIG =====================
HF_MODEL = "Qwen/Qwen3.5-4B"
MODEL = {
    "gguf_prefix": "Qwen_Qwen3.5-4B",
    "gguf_dataset_slug": "asystemoffields/qwen35-4b-ggufs",
    "low": "iq2_m",
    "highs": "q3_k_s,q3_k_m,q3_k_l,iq4_xs,q4_k_m",
}
FAMILIES = "mlp"        # empirical 4B probe set is all-MLP; DeltaNet layers have no self_attn.*
CAPTURE_PASSES = 3      # cov RAM ~4.5 GB per 12-layer pass; checkpointing bounds the graph
DTYPE = "float32"       # bf16 matmul is ~250x slower on CPUs without AVX512-BF16/AMX
CTX, CHUNKS = 512, 12   # rank validation tolerates sqrt(2) more SE; halves capture time
# v1 postmortem: fp32 with no checkpointing built a full-depth graph on pass 1
# and swapped the 30 GB kernel to death (7.6h for one chunk).
# ==================================================

import glob, os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
PMRA = WORK / "PMRA"
OUT = WORK / "output"
OUT.mkdir(parents=True, exist_ok=True)

# %%
# Repo + deps (torch CPU build keeps the download small)
if not PMRA.exists():
    subprocess.run(["git", "clone", "--depth=1", "https://github.com/Asystemoffields/PMRA.git", str(PMRA)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "--index-url",
                "https://download.pytorch.org/whl/cpu"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers", "gguf", "huggingface_hub", "datasets", "safetensors"], check=True)

# %%
# HF reference model (~8 GB, kernel-side download)
from huggingface_hub import snapshot_download

model_dir = snapshot_download(HF_MODEL, local_dir=str(WORK / "hf-model"))
print("model:", model_dir)

# %%
# GGUF sources from the mounted Dataset (low + highs only; no ref needed)
labels = sorted({MODEL["low"], *MODEL["highs"].split(",")})
user, name = MODEL["gguf_dataset_slug"].split("/")
roots = [Path("/kaggle/input") / name, Path("/kaggle/input/datasets") / user / name]
dataset_root = next(r for r in roots if r.exists())
source_args = []
for label in labels:
    path = dataset_root / f"{MODEL['gguf_prefix']}-{label.upper()}.gguf"
    assert path.exists(), path
    source_args += ["--source", f"{label}={path}"]
    print(f"  {label:8s} <- {path}")

# %%
# Calibration text — identical recipe to the prober run
from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
out, size = [], 0
for row in ds:
    text = row["text"].strip()
    if not text:
        continue
    out.append(text)
    size += len(text) + 1
    if size > 256 * 1024:
        break
(WORK / "calib.txt").write_text("\n".join(out))
print(f"calib.txt {size/1024:.0f} KB")

# %%
# Empirical rows for the validation ladder (salvaged-rows Dataset)
rows = sorted(glob.glob("/kaggle/input/**/allocation_rows.jsonl", recursive=True))
assert rows, "no allocation_rows.jsonl mounted"
print("rows:", rows[0])

# %%
cmd = [
    sys.executable, str(PMRA / "scripts" / "hessian_scorer.py"),
    "--model-dir", model_dir,
    *source_args,
    "--low-source", MODEL["low"], "--high-sources", MODEL["highs"],
    "--calib-text", str(WORK / "calib.txt"),
    "--ctx", str(CTX), "--chunks", str(CHUNKS),
    "--threads", str(os.cpu_count() or 4),
    "--families", FAMILIES, "--capture-passes", str(CAPTURE_PASSES),
    "--dtype", DTYPE, "--gradient-checkpointing", "--drop-vision",
    "--output", str(OUT / "hessian_scores.json"),
    "--validate-rows", rows[0],
]
print(" \\\n  ".join(cmd))
result = subprocess.run(cmd, cwd=str(PMRA / "scripts"))
print(f"scorer exit: {result.returncode}")
assert result.returncode == 0

# %%
# Shrink kernel output to deliverables
import shutil

for path in [PMRA, WORK / "hf-model"]:
    shutil.rmtree(path, ignore_errors=True)
print("Final output files:")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        print(f"  {f} ({f.stat().st_size/1e6:.1f} MB)")
