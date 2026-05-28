# %% [markdown]
# # PMRA Free-GPU Pipeline
#
# Run Production Mixed-Rate Allocation on free Kaggle/Colab T4 GPUs.
#
# **Supported models:**
# - NVIDIA Nemotron-3-Nano-4B (Mamba2-Transformer hybrid, `nemotron_h` profile)
# - Qwen3.5-4B (DeltaNet-Transformer hybrid, `qwen35` profile)
#
# **What this does:**
# 1. Downloads pre-quantized GGUF sources from HuggingFace
# 2. Downloads the HF model weights for forward evaluation
# 3. Runs the PMRA probing loop (swap each tensor, measure NLL delta)
# 4. Solves a knapsack to pick the best tensor promotions under a byte budget
# 5. Assembles the mixed-precision GGUF artifact
# 6. Validates with perplexity on held-out data
# 7. (Optional) Pushes the result to HuggingFace Hub

# %% [markdown]
# ## 1. Configuration
#
# Pick your model and set parameters. Everything else flows from this cell.

# %%
# ===================== CHOOSE YOUR MODEL =====================
MODEL_KEY = "nemotron_4b"  # "nemotron_4b" or "qwen35_4b"
# =============================================================

# HuggingFace token for pushing results (optional — set in Kaggle secrets or paste here)
import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ------------- Model configs ------------------------------------------------
MODEL_CONFIGS = {
    "nemotron_4b": {
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "gguf_repo": "bartowski/nvidia_Nemotron-3-Nano-4B-GGUF",
        "gguf_prefix": "nvidia_Nemotron-3-Nano-4B",
        "tensor_profile": "nemotron_h",
        "num_layers": 42,
        "group_mode": "tensor",           # tensor-level granularity
        "low_source": "iq2_m",
        "target_source": "iq3_xs",
        "high_sources": ["q3_k_s", "q3_k_m", "iq4_xs"],
        # Map from label used by the pipeline → actual GGUF filename suffix
        "source_quants": {
            "iq2_m":  "IQ2_M",
            "iq3_xs": "IQ3_XS",
            "q3_k_s": "Q3_K_S",
            "q3_k_m": "Q3_K_M",
            "iq4_xs": "IQ4_XS",
        },
        "calib_prompts": 12,
        "eval_prompts": 64,
        "calib_max_length": 96,
        "eval_max_length": 128,
        "hf_file_name": "model.safetensors",   # single file, not sharded
        "output_name": "nemotron3-nano-4b-pmra",
    },
    "qwen35_4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "gguf_repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "gguf_prefix": "Qwen_Qwen3.5-4B",
        "tensor_profile": "qwen35",
        "num_layers": 32,
        "group_mode": "layer_family",      # coarser grouping (matches abliterated run)
        "low_source": "iq2_m",
        "target_source": "iq3_xs",
        "high_sources": ["q3_k_s", "q3_k_m", "q3_k_l", "iq4_xs", "q4_k_m"],
        "source_quants": {
            "iq2_m":  "IQ2_M",
            "iq3_xs": "IQ3_XS",
            "q3_k_s": "Q3_K_S",
            "q3_k_m": "Q3_K_M",
            "q3_k_l": "Q3_K_L",
            "iq4_xs": "IQ4_XS",
            "q4_k_m": "Q4_K_M",
        },
        "calib_prompts": 12,
        "eval_prompts": 64,
        "calib_max_length": 96,
        "eval_max_length": 128,
        "hf_file_name": "model.safetensors.index.json",  # sharded
        "output_name": "qwen35-4b-pmra",
    },
}

cfg = MODEL_CONFIGS[MODEL_KEY]
print(f"Selected model: {MODEL_KEY}")
print(f"  HF model:        {cfg['model_id']}")
print(f"  GGUF source:     {cfg['gguf_repo']}")
print(f"  Tensor profile:  {cfg['tensor_profile']}")
print(f"  Layers:          {cfg['num_layers']}")
print(f"  Low source:      {cfg['low_source']}")
print(f"  Target:          {cfg['target_source']}")
print(f"  High sources:    {cfg['high_sources']}")

# %% [markdown]
# ## 2. Install & Setup
#
# Clone the PMRA repo and install dependencies.
# On Kaggle/Colab the base image already has torch + CUDA.

# %%
import subprocess, sys, os
from pathlib import Path

# Workspace paths
WORK_DIR = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content")
PMRA_DIR = WORK_DIR / "PMRA"
MODEL_DIR = WORK_DIR / "model"
GGUF_DIR = WORK_DIR / "ggufs"
OUTPUT_DIR = WORK_DIR / "output"

for d in [MODEL_DIR, GGUF_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Clone repo (or pull latest)
if not PMRA_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/Asystemoffields/PMRA.git", str(PMRA_DIR)],
        check=True,
    )
    print("Cloned PMRA repo")
else:
    subprocess.run(["git", "-C", str(PMRA_DIR), "pull", "--ff-only"], check=False)
    print("PMRA repo already present, pulled latest")

# Install Python deps (skip torch — already installed on Kaggle/Colab)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "gguf", "safetensors", "datasets", "huggingface_hub", "accelerate", "sentencepiece"],
    check=True,
)
# mamba-ssm + causal-conv1d needed for NemotronH (compiles CUDA kernels — takes a few minutes)
if cfg["tensor_profile"] == "nemotron_h":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mamba-ssm", "causal-conv1d"],
        check=True,
    )
print("Dependencies installed")

# %% [markdown]
# ## 3. Download GGUF Sources
#
# Pull the pre-quantized GGUF variants we need from HuggingFace.
# bartowski's quants are imatrix-calibrated, so we skip generating our own.

# %%
from huggingface_hub import hf_hub_download

source_paths = {}
all_sources = {cfg["low_source"], cfg["target_source"]} | set(cfg["high_sources"])

print(f"Downloading {len(all_sources)} GGUF variants from {cfg['gguf_repo']}...")
for label in sorted(all_sources):
    quant_suffix = cfg["source_quants"][label]
    filename = f"{cfg['gguf_prefix']}-{quant_suffix}.gguf"
    print(f"  {label:10s} -> {filename}")
    local_path = hf_hub_download(
        repo_id=cfg["gguf_repo"],
        filename=filename,
        local_dir=str(GGUF_DIR),
        local_dir_use_symlinks=False,
    )
    source_paths[label] = str(local_path)
    print(f"             saved to {local_path}")

print(f"\nAll GGUF sources downloaded ({len(source_paths)} files)")

# %% [markdown]
# ## 4. Download HF Model
#
# We need the original HF weights as the reference for forward evaluation.
# The model is loaded in bfloat16 — fits easily in T4's 16GB VRAM at 4B params (~8GB).

# %%
from huggingface_hub import snapshot_download

print(f"Downloading HF model: {cfg['model_id']}...")
snapshot_download(
    repo_id=cfg["model_id"],
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False,
    ignore_patterns=["*.md", "*.txt", "*.jinja", "*.png", "*.jpg"],
)
print(f"HF model downloaded to {MODEL_DIR}")

# Locate the safetensors reference file
hf_file = MODEL_DIR / cfg["hf_file_name"]
if not hf_file.exists():
    # Fall back: search for any .safetensors file
    candidates = list(MODEL_DIR.glob("*.safetensors"))
    if candidates:
        hf_file = candidates[0]
        print(f"  Using fallback HF file: {hf_file}")
    else:
        raise FileNotFoundError(f"No safetensors file found in {MODEL_DIR}")
print(f"  HF reference file: {hf_file}")

# %% [markdown]
# ## 5. Run PMRA Probing
#
# This is the core step — the long pole. For each tensor (or tensor group),
# we swap it from the low source to each higher source, measure the NLL delta,
# and record the cost/benefit ratio.
#
# **Expected runtime:** ~3–6 hours for a 4B model on T4, depending on
# the number of source variants and tensor count.
#
# The script checkpoints progress, so if the session is interrupted,
# you can restart and it will resume from the last completed probe.

# %%
import time

layers_str = ",".join(str(i) for i in range(cfg["num_layers"]))
high_sources_str = ",".join(cfg["high_sources"])

# Build --source arguments
source_args = []
for label, path in source_paths.items():
    source_args.extend(["--source", f"{label}={path}"])

cmd = [
    sys.executable,
    str(PMRA_DIR / "scripts" / "production_mixed_rate_transcoder_gate.py"),
    "--model-dir", str(MODEL_DIR),
    "--hf", str(hf_file),
    "--output-dir", str(OUTPUT_DIR),
    "--low-source", cfg["low_source"],
    "--target-source", cfg["target_source"],
    "--high-sources", high_sources_str,
    "--layers", layers_str,
    "--group-mode", cfg["group_mode"],
    "--tensor-profile", cfg["tensor_profile"],
    "--calib-prompts", str(cfg["calib_prompts"]),
    "--eval-prompts", str(cfg["eval_prompts"]),
    "--calib-max-length", str(cfg["calib_max_length"]),
    "--eval-max-length", str(cfg["eval_max_length"]),
    "--prompt-source", "public",
    "--dataset", "wikitext",
    "--dataset-config", "wikitext-2-raw-v1",
    "--calib-split", "train",
    "--eval-split", "validation",
    "--candidate-variant", "c2_calib_greedy_mixed",
    *source_args,
]

print("=" * 72)
print("PMRA PROBING COMMAND")
print("=" * 72)
print(" \\\n  ".join(cmd))
print("=" * 72)

t0 = time.time()
result = subprocess.run(cmd, cwd=str(PMRA_DIR / "scripts"))
elapsed = time.time() - t0

if result.returncode != 0:
    print(f"\n*** PROBING FAILED (exit code {result.returncode}) after {elapsed/60:.1f} min ***")
    print("Check output above for errors.")
else:
    print(f"\n*** PROBING COMPLETE in {elapsed/60:.1f} min ***")

# %% [markdown]
# ## 6. Inspect Results
#
# Load the result.json and show the key metrics before building the artifact.

# %%
import json

result_json_path = OUTPUT_DIR / "result.json"
if not result_json_path.exists():
    print("No result.json found — probing may not have completed.")
else:
    with open(result_json_path) as f:
        result_data = json.load(f)

    print("=" * 60)
    print("PMRA PROBING RESULTS")
    print("=" * 60)

    # Show variant NLL comparisons
    variants = result_data.get("variants", {})
    for name, v in sorted(variants.items()):
        nll = v.get("nll", "?")
        bpw = v.get("payload_bpw", "?")
        payload_mb = v.get("payload_bytes", 0) / 1e6
        print(f"  {name:40s}  NLL={nll:<10}  bpw={bpw:<8}  payload={payload_mb:.1f} MB")

    # Show selection summary
    selections = result_data.get("selections", {})
    for variant_name, sel_rows in selections.items():
        from collections import Counter
        src_counts = Counter(row["source"] for row in sel_rows)
        total_improvement = sum(row.get("calib_nll_improvement", 0) for row in sel_rows)
        print(f"\n  Selection '{variant_name}': {len(sel_rows)} groups promoted")
        for src, count in src_counts.most_common():
            print(f"    {src}: {count} groups")
        print(f"    Total NLL improvement: {total_improvement:.6f}")

    verdict = result_data.get("verdict", "UNKNOWN")
    print(f"\n  Verdict: {verdict}")

# %% [markdown]
# ## 7. Build Mixed GGUF Artifact
#
# Assemble the final mixed-precision GGUF by copying selected tensor payloads
# from the appropriate source GGUFs.

# %%
if result_json_path.exists():
    artifact_dir = OUTPUT_DIR / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_cmd = [
        sys.executable,
        str(PMRA_DIR / "scripts" / "build_mixed_gguf_artifact.py"),
        "--result-json", str(result_json_path),
        "--output-dir", str(artifact_dir),
        "--output-gguf", f"{cfg['output_name']}.gguf",
        "--variant", "c2_calib_greedy_mixed",
        "--metadata-source", cfg["low_source"],
        "--layers", layers_str,
        "--group-mode", cfg["group_mode"],
        "--tensor-profile", cfg["tensor_profile"],
        *source_args,
    ]

    print("Building mixed GGUF artifact...")
    t0 = time.time()
    result = subprocess.run(artifact_cmd, cwd=str(PMRA_DIR / "scripts"))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n*** ARTIFACT BUILD FAILED (exit code {result.returncode}) ***")
    else:
        print(f"\n*** ARTIFACT BUILT in {elapsed:.1f}s ***")
        # Show artifact details
        artifact_file = artifact_dir / f"{cfg['output_name']}.gguf"
        if artifact_file.exists():
            size_gb = artifact_file.stat().st_size / 1e9
            print(f"  Output: {artifact_file}")
            print(f"  Size:   {size_gb:.2f} GB")

        report_file = artifact_dir / "artifact_report.md"
        if report_file.exists():
            print(f"\n--- Artifact Report ---")
            print(report_file.read_text(encoding="utf-8")[:2000])
else:
    print("Skipping artifact build — no result.json")

# %% [markdown]
# ## 8. Validate (Perplexity)
#
# Run llama-perplexity on the mixed GGUF to confirm quality.
# This step requires llama.cpp compiled with CUDA.
# If compilation fails on your platform, you can skip this and validate locally.

# %%
LLAMA_CPP_DIR = WORK_DIR / "llama.cpp"
COMPILE_LLAMA = True  # Set to False to skip compilation

if COMPILE_LLAMA:
    if not LLAMA_CPP_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_CPP_DIR)],
            check=True,
        )
    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["cmake", "..", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75"],  # 75 for T4
        cwd=str(build_dir), check=True,
    )
    subprocess.run(
        ["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count() or 4)],
        cwd=str(build_dir), check=True,
    )
    perplexity_bin = build_dir / "bin" / "llama-perplexity"
    print(f"llama.cpp compiled, binary: {perplexity_bin}")

# %%
# Download wikitext-2 test set for validation
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_text_path = WORK_DIR / "wikitext2_test.txt"
with open(eval_text_path, "w", encoding="utf-8") as f:
    for row in ds:
        text = row.get("text", "").strip()
        if text:
            f.write(text + "\n")
print(f"Wikitext-2 test text: {eval_text_path} ({eval_text_path.stat().st_size / 1024:.1f} KB)")

# %%
# Run perplexity eval on the mixed GGUF and the target baseline for comparison
artifact_file = artifact_dir / f"{cfg['output_name']}.gguf" if 'artifact_dir' in dir() else None
target_gguf = source_paths.get(cfg["target_source"])
perplexity_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-perplexity"

if artifact_file and artifact_file.exists() and perplexity_bin.exists():
    for label, gguf_path in [("PMRA_MIX", str(artifact_file)), ("TARGET_BASELINE", target_gguf)]:
        print(f"\n{'='*60}")
        print(f"Perplexity eval: {label}")
        print(f"  GGUF: {gguf_path}")
        print(f"{'='*60}")
        ppl_cmd = [
            str(perplexity_bin),
            "-m", gguf_path,
            "-f", str(eval_text_path),
            "-ngl", "999",         # offload all layers to GPU
            "--chunks", "64",      # limit for faster eval
        ]
        subprocess.run(ppl_cmd)
else:
    missing = []
    if not artifact_file or not artifact_file.exists():
        missing.append("mixed GGUF artifact")
    if not perplexity_bin.exists():
        missing.append("llama-perplexity binary")
    print(f"Skipping perplexity eval — missing: {', '.join(missing)}")

# %% [markdown]
# ## 9. Push to HuggingFace Hub (Optional)
#
# Upload the mixed GGUF, result.json, and artifact report to a new HF repo.

# %%
PUSH_TO_HUB = False  # Set to True when ready
HUB_REPO_ID = f"YOUR_USERNAME/{cfg['output_name']}"  # Change this!

if PUSH_TO_HUB and HF_TOKEN:
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HUB_REPO_ID, repo_type="model", exist_ok=True)

    upload_files = []
    if artifact_file and artifact_file.exists():
        upload_files.append((str(artifact_file), artifact_file.name))
    if result_json_path.exists():
        upload_files.append((str(result_json_path), "result.json"))
    report_file = artifact_dir / "artifact_report.md" if 'artifact_dir' in dir() else None
    if report_file and report_file.exists():
        upload_files.append((str(report_file), "artifact_report.md"))
    report_json = artifact_dir / "artifact_report.json" if 'artifact_dir' in dir() else None
    if report_json and report_json.exists():
        upload_files.append((str(report_json), "artifact_report.json"))

    for local_path, repo_path in upload_files:
        print(f"Uploading {repo_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=HUB_REPO_ID,
            repo_type="model",
        )

    print(f"\nUploaded to https://huggingface.co/{HUB_REPO_ID}")
elif PUSH_TO_HUB and not HF_TOKEN:
    print("Set HF_TOKEN to push to the Hub.")
else:
    print("Push to Hub disabled. Set PUSH_TO_HUB = True when ready.")

# %% [markdown]
# ## Notes
#
# ### Kaggle-specific tips
# - Use **2x T4** accelerator (Settings → Accelerator)
# - Enable **Internet access** (Settings → Internet → On)
# - Set `HF_TOKEN` in Kaggle Secrets, then access via `os.environ.get("HF_TOKEN")`
# - Kaggle gives **30h/week** of GPU time; a 4B model run takes ~3-6h
# - If the session dies mid-probe, the script checkpoints progress
#   (restart from the probing cell — it resumes automatically)
#
# ### Colab-specific tips
# - Use **T4 GPU** runtime (Runtime → Change runtime type)
# - Colab free sessions last ~12h max with activity
# - Mount Google Drive for persistent storage between sessions:
#   ```python
#   from google.colab import drive
#   drive.mount('/content/drive')
#   WORK_DIR = Path("/content/drive/MyDrive/PMRA")
#   ```
#
# ### Resource usage
# - **VRAM:** ~8 GB for 4B model in bfloat16 (T4 has 16 GB — plenty of room)
# - **Disk:** ~15-20 GB for model + all GGUF sources + artifact
# - **RAM:** ~10 GB (Kaggle provides 13 GB)
#
# ### If the model doesn't fit
# If you're adapting this for a larger model (7B+), you may need to:
# - Use `device_map="auto"` for CPU offloading during the HF model load
# - Reduce `calib_prompts` and `eval_prompts`
# - Use `layer_family` group mode (fewer probes) instead of `tensor` mode
