# %% [markdown]
# # PMRA factorial probing — Kaggle CPU kernel
#
# Runs `scripts/factorial_probe.py` against the Qwen3.5-4B singles rows:
# a PB20 design over the 19 probed MLP groups (+ all_cheap / level_high /
# level_knap rows) = 23 multi-promotion evals, validating whether factorial
# probing can replace one-at-a-time tier-2 (and testing the knapsack's
# additivity assumption directly). Shard with SHARD = "k/3" (~8 evals each).

# %%
# ===================== CONFIG =====================
MODEL = {
    "gguf_prefix": "Qwen_Qwen3.5-4B",
    "gguf_dataset_slug": "asystemoffields/qwen35-4b-ggufs",
    "low": "iq2_m",
    "sources": "q3_k_s,q3_k_m,q3_k_l,iq4_xs,q4_k_m",  # any level a design row may assign
}
SHARD = None         # "k/N" for a shard worker, None to run all 23 evals
CTX, CHUNKS = 512, 24   # match the singles probes
# ==================================================

import glob, json, os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
PMRA = WORK / "PMRA"
OUT = WORK / "output"
OUT.mkdir(parents=True, exist_ok=True)

# %%
# Repo + deps
if not PMRA.exists():
    subprocess.run(["git", "clone", "--depth=1", "https://github.com/Asystemoffields/PMRA.git", str(PMRA)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "huggingface_hub", "datasets"], check=True)

# %%
# llama.cpp CPU binaries: prebuilt release zip (fast), fallback to cmake build
LLAMA_BIN = None
try:
    import urllib.request
    api = json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"))
    url = next(a["browser_download_url"] for a in api["assets"] if "bin-ubuntu-x64" in a["name"])
    subprocess.run(["curl", "-sL", url, "-o", "/tmp/llama.zip"], check=True)
    subprocess.run(["unzip", "-qo", "/tmp/llama.zip", "-d", str(WORK / "llama-bin")], check=True)
    LLAMA_BIN = next(p for p in (WORK / "llama-bin").rglob("llama-perplexity")).parent
    subprocess.run([str(LLAMA_BIN / "llama-perplexity"), "--version"], check=True, capture_output=True)
    print(f"prebuilt llama.cpp ok: {LLAMA_BIN}")
except Exception as e:
    print(f"prebuilt failed ({e!r}); building from source")
    src = WORK / "llama.cpp"
    if not src.exists():
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/ggml-org/llama.cpp.git", str(src)], check=True)
    subprocess.run(["cmake", "-S", str(src), "-B", str(src / "build"), "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(src / "build"), "-j", str(os.cpu_count() or 4),
                    "--target", "llama-perplexity"], check=True)
    LLAMA_BIN = src / "build" / "bin"

# %%
# GGUF sources from the mounted Dataset
labels = sorted({MODEL["low"], *MODEL["sources"].split(",")})
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
# Calibration text — identical recipe to the singles run
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
# Singles rows (design input + validation truth) and the base eval with
# per-chunk NLLs, both from the salvaged-rows Dataset
rows = sorted(glob.glob("/kaggle/input/**/allocation_rows.jsonl", recursive=True))
assert rows, "no allocation_rows.jsonl mounted"
scalar = None
for cand in sorted(glob.glob("/kaggle/input/**/scalar_evals.jsonl", recursive=True)):
    for line in Path(cand).read_text().splitlines():
        if line.strip() and "chunk_nlls" in json.loads(line):
            scalar = cand
            break
    if scalar:
        break
print("rows:", rows[0])
print("scalar evals (chunked):", scalar)

# %%
# Merge any predecessor factorial evals (resume after session cap)
prev = glob.glob("/kaggle/input/**/factorial_evals.jsonl", recursive=True)
if prev:
    seen = set()
    with (OUT / "factorial_evals.jsonl").open("w") as fh:
        for p in prev:
            for line in Path(p).read_text().splitlines():
                if line.strip() and json.loads(line)["id"] not in seen:
                    seen.add(json.loads(line)["id"])
                    fh.write(line + "\n")
    print(f"carried {len(seen)} prior evals")

# %%
cmd = [
    sys.executable, str(PMRA / "scripts" / "factorial_probe.py"),
    *source_args,
    "--low-source", MODEL["low"],
    "--rows", rows[0],
    "--calib-text", str(WORK / "calib.txt"),
    "--llama-bin", str(LLAMA_BIN),
    "--output-dir", str(OUT),
    "--ctx", str(CTX), "--chunks", str(CHUNKS),
    "--threads", str(os.cpu_count() or 4),
]
if scalar:
    cmd += ["--scalar-evals", scalar]
if SHARD:
    cmd += ["--shard", SHARD]
print(" \\\n  ".join(cmd))
result = subprocess.run(cmd)
print(f"factorial exit: {result.returncode}")
assert result.returncode == 0

# %%
# Fit report (informative on the final/merged kernel; partial shards skip recovery)
subprocess.run(cmd + ["--fit"])

# %%
# Shrink kernel output to deliverables
import shutil

for path in [PMRA, WORK / "llama.cpp", WORK / "llama-bin"]:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
(OUT / "factorial_mix.gguf").unlink(missing_ok=True)
print("Final output files:")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        print(f"  {f} ({f.stat().st_size/1e6:.1f} MB)")
