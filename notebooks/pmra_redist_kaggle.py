# %% [markdown]
# # PMRA attention + redistribution probes — Kaggle CPU kernel
#
# Applies the Qwen3.5-4B GRAY lesson. Two custom factorial_probe designs:
#
# **From the iq2_m base** (was the mix's blind spot — the imatrix proxy scores
# every attn group 0.0 on the DeltaNet path, so no attn candidate was ever
# probed and the knapsack couldn't buy attention):
#   attn_all / attn_early / attn_mid / attn_late — promote attn groups to q3_k_s
#   attn_plus_knap — the shipped 17-promotion mix PLUS all-attn (does attention
#   close the 0.033-nat gap to uniform iq3_xs?)
#
# **From the iq3_xs (uniform target) base** — demote-to-fund-promote:
#   demote_only — drop measured-cold L8/L16 MLP to iq2_m (harvestable slack?)
#   redist_cons / redist_aggr — spend the freed bytes on the steep groups
#   (L31/L22/L29 at q4_k_m), net bytes <= 0 vs uniform.

# %%
# ===================== CONFIG =====================
MODEL = {
    "gguf_prefix": "Qwen_Qwen3.5-4B",
    "gguf_dataset_slug": "asystemoffields/qwen35-4b-ggufs",
    "low": "iq2_m",
    "target": "iq3_xs",
    "sources": "q3_k_s,q3_k_m,q3_k_l,iq4_xs,q4_k_m",
}
# the c2_calib_knapsack_mixed selection from result.json (2026-06-12 run)
KNAP_17 = {
    "L6:mlp": "q3_k_s", "L7:mlp": "q3_k_s", "L17:mlp": "q3_k_s", "L18:mlp": "q3_k_s",
    "L19:mlp": "q3_k_s", "L20:mlp": "iq4_xs", "L21:mlp": "q3_k_m", "L22:mlp": "q4_k_m",
    "L23:mlp": "q3_k_l", "L24:mlp": "iq4_xs", "L25:mlp": "q3_k_l", "L26:mlp": "iq4_xs",
    "L27:mlp": "q3_k_s", "L28:mlp": "q3_k_l", "L29:mlp": "q4_k_m", "L30:mlp": "iq4_xs",
    "L31:mlp": "q4_k_m",
}
CTX, CHUNKS = 512, 24
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
sys.path.insert(0, str(PMRA / "scripts"))

# %%
# llama.cpp CPU binaries
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
# GGUF sources
labels = sorted({MODEL["low"], MODEL["target"], *MODEL["sources"].split(",")})
user, name = MODEL["gguf_dataset_slug"].split("/")
roots = [Path("/kaggle/input") / name, Path("/kaggle/input/datasets") / user / name]
dataset_root = next(r for r in roots if r.exists())
source_args = []
paths = {}
for label in labels:
    path = dataset_root / f"{MODEL['gguf_prefix']}-{label.upper()}.gguf"
    assert path.exists(), path
    paths[label] = path
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
# Inputs from the salvaged-rows Dataset
rows = sorted(glob.glob("/kaggle/input/**/allocation_rows.jsonl", recursive=True))[0]
scalar = None
for cand in sorted(glob.glob("/kaggle/input/**/scalar_evals.jsonl", recursive=True)):
    if any("chunk_nlls" in l for l in Path(cand).read_text().splitlines()):
        scalar = cand
        break
print("rows:", rows, "\nscalar:", scalar)

# %%
# Build the two custom designs with exact per-group byte accounting
from cpu_prober import open_sources, build_groups

readers = open_sources(paths)
groups = build_groups(readers[MODEL["low"]], "layer_family")
tensors = {lab: {t.name: int(t.n_bytes) for t in r.tensors} for lab, r in readers.items()}

def gbytes(group, label):
    return sum(tensors[label][n] for n in groups[group])

attn_groups = sorted((g for g in groups if g.endswith(":attn")),
                     key=lambda g: int(g.split(":")[0][1:]))  # L<n>:attn only; skip global:*
print(f"{len(attn_groups)} non-mlp groups: {attn_groups[:4]} ... {attn_groups[-2:]}")
n = len(attn_groups)
bands = {"attn_early": attn_groups[: n // 3],
         "attn_mid": attn_groups[n // 3 : 2 * n // 3],
         "attn_late": attn_groups[2 * n // 3 :]}

design_low = [
    {"id": "attn_all", "kind": "attn", "assignment": {g: "q3_k_s" for g in attn_groups}},
    *[{"id": k, "kind": "attn", "assignment": {g: "q3_k_s" for g in v}} for k, v in bands.items()],
    {"id": "attn_plus_knap", "kind": "attn",
     "assignment": {**KNAP_17, **{g: "q3_k_s" for g in attn_groups}}},
]

# demote-to-fund-promote from the uniform target base
def redist(demote, promote_ladder):
    assignment = {g: MODEL["low"] for g in demote}
    freed = sum(gbytes(g, MODEL["target"]) - gbytes(g, MODEL["low"]) for g in demote)
    spent = 0
    for g, s in promote_ladder:
        cost = gbytes(g, s) - gbytes(g, MODEL["target"])
        if spent + cost <= freed:
            assignment[g] = s
            spent += cost
    print(f"  freed {freed/1e6:.1f}MB, spent {spent/1e6:.1f}MB on {sum(1 for g in assignment if assignment[g] != MODEL['low'])} promotions")
    return assignment

LADDER = [("L31:mlp", "q4_k_m"), ("L22:mlp", "q4_k_m"), ("L29:mlp", "q4_k_m"),
          ("L27:mlp", "q4_k_m"), ("L23:mlp", "q3_k_l"), ("L26:mlp", "iq4_xs")]
design_target = [
    {"id": "demote_only", "kind": "redist",
     "assignment": {"L8:mlp": MODEL["low"], "L16:mlp": MODEL["low"]}},
    {"id": "redist_cons", "kind": "redist",
     "assignment": redist(["L8:mlp", "L16:mlp"], LADDER)},
    {"id": "redist_aggr", "kind": "redist",
     "assignment": redist(["L8:mlp", "L16:mlp", "L17:mlp", "L25:mlp", "L29:mlp"], LADDER)},
]
(WORK / "design_low.json").write_text(json.dumps(design_low))
(WORK / "design_target.json").write_text(json.dumps(design_target))

# %%
# Run both designs (paired vs their own bases)
common = [sys.executable, str(PMRA / "scripts" / "factorial_probe.py"), *source_args,
          "--low-source", MODEL["low"], "--rows", rows,
          "--calib-text", str(WORK / "calib.txt"), "--llama-bin", str(LLAMA_BIN),
          "--ctx", str(CTX), "--chunks", str(CHUNKS), "--threads", str(os.cpu_count() or 4)]
r1 = subprocess.run(common + ["--custom-design", str(WORK / "design_low.json"),
                              "--output-dir", str(OUT / "attn"),
                              *(["--scalar-evals", scalar] if scalar else [])])
r2 = subprocess.run(common + ["--custom-design", str(WORK / "design_target.json"),
                              "--output-dir", str(OUT / "redist"),
                              "--base-source", MODEL["target"]])
print(f"exits: attn={r1.returncode} redist={r2.returncode}")
assert r1.returncode == 0 and r2.returncode == 0

# %%
# Shrink kernel output to deliverables
import shutil

for path in [PMRA, WORK / "llama.cpp", WORK / "llama-bin"]:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
for sub in ["attn", "redist"]:
    (OUT / sub / "factorial_mix.gguf").unlink(missing_ok=True)
print("Final output files:")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        print(f"  {f} ({f.stat().st_size/1e6:.1f} MB)")
