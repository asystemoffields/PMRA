# %% [markdown]
# # PMRA CPU Prober — Kaggle CPU Kernel
#
# Runs the two-tier llama.cpp CPU prober (`scripts/cpu_prober.py`) on a free
# Kaggle **CPU** kernel. CPU kernels have **no weekly quota** (unlike the 30h
# GPU cap), ~100 GB working disk via mounted Datasets, and allow concurrent
# sessions — so tier-2 probing shards horizontally across kernels.
#
# **Sharding:** set `SHARD = "k/N"` (e.g. "0/8"; 8 shards is the sweet spot —
# Kaggle allows ~10 concurrent CPU sessions) and launch N copies of this
# kernel (Save Version → Save & Run All on N forks). Each writes a partial
# `checkpoints/allocation_rows.jsonl`. A final kernel (or local run) with
# `SHARD = None, STAGES = "finalize"` mounts the shard outputs via
# kernel_sources and merges them — same dedup as the b1/b2 layer-split resume.

# %%
# ===================== CONFIG =====================
MODELS = {
    "qwen35_4b": {
        "gguf_repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "gguf_prefix": "Qwen_Qwen3.5-4B",
        "gguf_dataset_slug": "asystemoffields/qwen35-4b-ggufs",  # mounted Kaggle Dataset (preferred over HF download)
        "low": "iq2_m",
        "target": "iq3_xs",
        "highs": "q3_k_s,q3_k_m,q3_k_l,iq4_xs,q4_k_m",
        "ref": "q8_0",          # near-lossless tier-1 reference; q8_0 keeps imatrix CPU time sane at 4B
        "imatrix_url": None,
        "tensor_profile": "qwen35",
        "group_mode": "layer_family",
    },
    "nemotron_4b": {
        "gguf_repo": "bartowski/nvidia_Nemotron-3-Nano-4B-GGUF",
        "gguf_prefix": "nvidia_Nemotron-3-Nano-4B",
        "gguf_dataset_slug": "asystemoffields/nemotron3-nano-4b-ggufs",  # reserved, never built; HF fallback is fine
        "low": "iq2_m",
        "target": "iq3_xs",
        "highs": "q3_k_s,q3_k_m,iq4_xs",
        # no ref quant: bartowski ships the imatrix, so we skip the 4.2GB q8_0
        # (disk: 5 quants ~12.3GB + probe gguf must fit in 19.5GB /kaggle/working)
        "ref": None,
        "imatrix_url": "https://huggingface.co/bartowski/nvidia_Nemotron-3-Nano-4B-GGUF/resolve/main/nvidia_Nemotron-3-Nano-4B-imatrix.gguf",
        "tensor_profile": "nemotron_h",
        "group_mode": "tensor",
    },
}
MODEL_KEY = "nemotron_4b"
MODEL = MODELS[MODEL_KEY]
SHARD = None         # "k/N" for a shard worker, None for single-kernel run
STAGES = "all"       # "tier1" | "tier2" | "finalize" | "all"
MAX_PROBES = 32      # 64 does not fit a 12h CPU session at 4B (see PREREG_NEMOTRON.md)
CTX, CHUNKS = 512, 24
CODE_GUARDRAIL_EPS = 0.02  # None = skip code eval entirely; 0.0 = record-only; >0 = enforce at finalize
CODE_CHUNKS = 48           # code evals get more chunks: only 3 of them, and the guardrail SE matters
PROBE_SE_STOP = 0.004      # paired SEs below this are unresolvable at 24 chunks — don't chase them
TIER2_TIME_BUDGET_MIN = 390  # stop probing at ~6.5h so finalize + artifact always fit the 12h cap
MERGE_INPUT_CHECKPOINTS = False  # True ONLY for a deliberate shard/resume kernel with pinned sources
LLAMA_RELEASE = "b9859"    # pinned: verdicts must not depend on whichever llama.cpp shipped today
# ==================================================

import json, os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
PMRA = WORK / "PMRA"
OUT = WORK / "output"
GGUFS = WORK / "ggufs"
for d in [OUT, GGUFS]:
    d.mkdir(parents=True, exist_ok=True)

# %%
# Repo + deps (no torch needed for probing; CPU kernels preinstall numpy)
if not PMRA.exists():
    subprocess.run(["git", "clone", "--depth=1", "https://github.com/Asystemoffields/PMRA.git", str(PMRA)], check=True)
sha = subprocess.run(["git", "-C", str(PMRA), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"PMRA @ {sha}")
# capability probe: fail NOW, not after an hour of downloads, if the clone
# predates the profile-aware grouping + code guardrail
prober_src = (PMRA / "scripts" / "cpu_prober.py").read_text()
assert "NEMOTRON_H_TAILS" in prober_src and "--code-text" in prober_src, \
    "cloned PMRA lacks profile-aware grouping / code guardrail — push the current main first"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "huggingface_hub", "datasets"], check=True)

# %%
# llama.cpp CPU binaries: PINNED prebuilt release (the verdict must not depend
# on whichever llama.cpp shipped today), fallback to cmake build of the same tag
LLAMA_BIN = None
try:
    url = (f"https://github.com/ggml-org/llama.cpp/releases/download/"
           f"{LLAMA_RELEASE}/llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz")
    subprocess.run(["curl", "-sL", url, "-o", "/tmp/llama.tar.gz"], check=True)
    (WORK / "llama-bin").mkdir(exist_ok=True)
    subprocess.run(["tar", "-xzf", "/tmp/llama.tar.gz", "-C", str(WORK / "llama-bin")], check=True)
    LLAMA_BIN = next(p for p in (WORK / "llama-bin").rglob("llama-perplexity")).parent
    subprocess.run([str(LLAMA_BIN / "llama-perplexity"), "--version"], check=True, capture_output=True)
    print(f"prebuilt llama.cpp {LLAMA_RELEASE} ok: {LLAMA_BIN}")
except Exception as e:  # glibc mismatch etc -> build the SAME tag from source
    print(f"prebuilt failed ({e!r}); building {LLAMA_RELEASE} from source")
    src = WORK / "llama.cpp"
    if not src.exists():
        subprocess.run(["git", "clone", "--depth=1", "--branch", LLAMA_RELEASE,
                        "https://github.com/ggml-org/llama.cpp.git", str(src)], check=True)
    subprocess.run(["cmake", "-S", str(src), "-B", str(src / "build"), "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(src / "build"), "-j", str(os.cpu_count() or 4),
                    "--target", "llama-perplexity", "llama-imatrix"], check=True)
    LLAMA_BIN = src / "build" / "bin"

# %%
# GGUF sources: mounted Kaggle Dataset preferred, HF fallback
from huggingface_hub import hf_hub_download

labels = sorted({MODEL["low"], MODEL["target"], *([MODEL["ref"]] if MODEL["ref"] else []), *MODEL["highs"].split(",")})
slug = MODEL.get("gguf_dataset_slug", "")
roots = []
if slug:
    user, name = slug.split("/")
    roots = [Path("/kaggle/input") / name, Path("/kaggle/input/datasets") / user / name]
dataset_root = next((r for r in roots if r.exists()), None)

source_args = []
for label in labels:
    suffix = label if label == "f16" else label.upper()
    filename = f"{MODEL['gguf_prefix']}-{suffix}.gguf"
    if dataset_root and (dataset_root / filename).exists():
        path = dataset_root / filename
    else:
        path = hf_hub_download(repo_id=MODEL["gguf_repo"], filename=filename, local_dir=str(GGUFS))
    source_args += ["--source", f"{label}={path}"]
    print(f"  {label:8s} <- {path}")

# %%
# Corpus
from datasets import load_dataset

for split, name, max_kb in [("train", "calib.txt", 256), ("test", "eval.txt", 256)]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    out, size = [], 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        out.append(text)
        size += len(text) + 1
        if size > max_kb * 1024:
            break
    (WORK / name).write_text("\n".join(out))
    print(name, f"{size/1024:.0f} KB")

# Code corpus for the release guardrail (MBPP-sanitized + HumanEval, ungated)
if CODE_GUARDRAIL_EPS is not None and STAGES in {"all", "finalize"}:
    subprocess.run([sys.executable, str(PMRA / "tools" / "build_code_corpus.py"),
                    "--output", str(WORK / "code.txt")], check=True)

# %%
# Pre-built imatrix (skips the in-kernel llama-imatrix run and the ref quant)
IMATRIX = None
if MODEL.get("imatrix_url"):
    IMATRIX = WORK / "imatrix.gguf"
    if not IMATRIX.exists():
        subprocess.run(["curl", "-sL", MODEL["imatrix_url"], "-o", str(IMATRIX)], check=True)
    assert IMATRIX.stat().st_size > 1_000_000 and IMATRIX.read_bytes()[:4] == b"GGUF", \
        f"imatrix download looks broken ({IMATRIX.stat().st_size} bytes)"
    print(f"imatrix: {IMATRIX} ({IMATRIX.stat().st_size/1e6:.1f} MB)")

# %%
# Merge predecessor checkpoints mounted via kernel_sources (shard fan-in,
# resume after session cap). Dedup key (group, source) — b1/b2 pattern.
import glob

ckpt = OUT / "checkpoints"
ckpt.mkdir(parents=True, exist_ok=True)
if MERGE_INPUT_CHECKPOINTS:
    # match anywhere under input: kernel_sources mount under .../output/checkpoints/,
    # but Dataset zip extraction may flatten the carrying folder
    shard_rows = glob.glob("/kaggle/input/**/allocation_rows.jsonl", recursive=True)
    if shard_rows:
        subprocess.run([sys.executable, str(PMRA / "scripts" / "merge_allocation_rows.py"),
                        *shard_rows, "--output", str(ckpt / "allocation_rows.jsonl")], check=True)
    for aux in ["tier1_scores.json", "scalar_evals.jsonl"]:
        if not (ckpt / aux).exists():
            prev = glob.glob(f"/kaggle/input/**/checkpoints/{aux}", recursive=True)
            if prev:
                import shutil
                shutil.copy(prev[0], ckpt / aux)
                print(f"carried {aux} from {prev[0]}")
    imatrix_prev = glob.glob("/kaggle/input/**/work/imatrix.gguf", recursive=True)
    if imatrix_prev:
        (OUT / "work").mkdir(exist_ok=True)
        import shutil
        shutil.copy(imatrix_prev[0], OUT / "work" / "imatrix.gguf")
else:
    # a fresh verdict run must not inherit rows from stale kernel attachments
    strays = glob.glob("/kaggle/input/**/allocation_rows.jsonl", recursive=True)
    assert not strays, f"unexpected checkpoint attachments on a fresh run: {strays}"

# %%
# Run the prober
cmd = [
    sys.executable, str(PMRA / "scripts" / "cpu_prober.py"),
    *source_args,
    "--low-source", MODEL["low"], "--target-source", MODEL["target"],
    "--high-sources", MODEL["highs"],
    "--calib-text", str(WORK / "calib.txt"), "--eval-text", str(WORK / "eval.txt"),
    "--output-dir", str(OUT), "--llama-bin", str(LLAMA_BIN),
    "--group-mode", MODEL["group_mode"], "--tensor-profile", MODEL["tensor_profile"],
    "--ctx", str(CTX), "--chunks", str(CHUNKS), "--max-probes", str(MAX_PROBES),
    "--threads", str(os.cpu_count() or 4),   # prober's default leaves 2 of 4 vCPUs idle
    "--probe-se-stop", str(PROBE_SE_STOP),
    "--tier2-time-budget-min", str(TIER2_TIME_BUDGET_MIN),
    "--stages", STAGES,
]
if MODEL["ref"]:
    cmd += ["--ref-source", MODEL["ref"]]
if IMATRIX:
    cmd += ["--imatrix", str(IMATRIX)]
if CODE_GUARDRAIL_EPS is not None and STAGES in {"all", "finalize"}:
    cmd += ["--code-text", str(WORK / "code.txt"), "--code-no-regress", str(CODE_GUARDRAIL_EPS),
            "--code-chunks", str(CODE_CHUNKS)]
if SHARD:
    cmd += ["--shard", SHARD]
print(" \\\n  ".join(cmd))
result = subprocess.run(cmd)
print(f"prober exit: {result.returncode}")
assert result.returncode == 0, "cpu_prober failed — do not let the kernel report success"

# %%
# Build the artifact when finalizing (needs torch for the gate-spec import)
if STAGES in {"all", "finalize"} and (OUT / "result.json").exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "--index-url",
                    "https://download.pytorch.org/whl/cpu"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "safetensors"], check=True)
    artifact = subprocess.run(
        [sys.executable, str(PMRA / "scripts" / "build_mixed_gguf_artifact.py"),
         "--result-json", str(OUT / "result.json"),
         "--output-dir", str(OUT / "artifact"),
         "--output-gguf", f"{MODEL['gguf_prefix'].lower()}-pmra-cpu.gguf",
         "--variant", "c2_calib_knapsack_mixed",
         *source_args],
        cwd=str(PMRA / "scripts"),
    )
    print(f"artifact exit: {artifact.returncode}")

# %%
# Shrink kernel output to deliverables
import shutil

for path in [PMRA, GGUFS, WORK / "llama.cpp", WORK / "llama-bin"]:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
print("Final output files:")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        print(f"  {f} ({f.stat().st_size/1e6:.1f} MB)")
