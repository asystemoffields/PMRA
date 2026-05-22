# Reproduce PMRA Mixes

This document describes the portable PMRA workflow. Modal was used for many of
the published runs because it is convenient for long GPU jobs, but PMRA itself
does not depend on Modal. The same scripts can run locally, in Colab, on a rented
GPU box, or in any other Python environment with enough RAM/VRAM for the model.

## Inputs

Every PMRA mix needs:

- the base model weights in a Transformers-readable directory
- a matching HF tensor source, usually the same directory or a safetensors index
- two or more existing GGUF source files from the same base checkpoint
- a tensor profile matching the model family
- a writable output directory

Source labels are important. The labels passed with `--source label=path` must
match `--low-source`, `--target-source`, and every item in `--high-sources`.

Supported tensor profiles:

```text
qwen, qwen35, gemma4, mistral3, granite, olmo2, olmo3
```

## Workflow

PMRA has two main steps.

1. Run the selector/evaluator:

```text
scripts/production_mixed_rate_transcoder_gate.py
```

This scores tensor-group promotions, selects a mix under the byte budget, and
writes `result.json` plus `result.md`.

2. Build the mixed GGUF:

```text
scripts/build_mixed_gguf_artifact.py
```

This reads `result.json`, copies selected production GGUF tensor payloads into
one GGUF, adds `pmra.*` metadata, and writes an artifact report.

## Local Python

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set paths for your machine. These are placeholders; use your own model and GGUF
locations.

```powershell
$MODEL_DIR = "C:\models\base-model"
$HF_SOURCE = "$MODEL_DIR\model.safetensors.index.json"
$GGUF_DIR = "C:\models\base-model-gguf"
$OUT = ".\results\local_pmra_run"
```

For single-file checkpoints, set `$HF_SOURCE` to the `.safetensors` file. For
sharded checkpoints, set it to the `.safetensors.index.json` file.

Run a small public-data selector. Increase prompt counts after the smoke run is
working.

```powershell
python .\scripts\production_mixed_rate_transcoder_gate.py `
  --model-dir $MODEL_DIR `
  --hf $HF_SOURCE `
  --source iq2_m="$GGUF_DIR\model.IQ2_M.gguf" `
  --source iq3_xs="$GGUF_DIR\model.IQ3_XS.gguf" `
  --source q3_k_s="$GGUF_DIR\model.Q3_K_S.gguf" `
  --source q3_k_m="$GGUF_DIR\model.Q3_K_M.gguf" `
  --source iq4_xs="$GGUF_DIR\model.IQ4_XS.gguf" `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q3_k_s,q3_k_m,iq4_xs `
  --tensor-profile <profile> `
  --layers <comma-separated-layer-ids> `
  --group-mode tensor `
  --prompt-source public `
  --dataset wikitext `
  --dataset-config wikitext-2-raw-v1 `
  --calib-split train `
  --eval-split validation `
  --calib-prompts 12 `
  --eval-prompts 32 `
  --calib-max-length 128 `
  --eval-max-length 192 `
  --candidate-variant c2_calib_knapsack_mixed `
  --output-dir $OUT
```

Build the GGUF from the selected variant:

```powershell
python .\scripts\build_mixed_gguf_artifact.py `
  --result-json "$OUT\result.json" `
  --source iq2_m="$GGUF_DIR\model.IQ2_M.gguf" `
  --source iq3_xs="$GGUF_DIR\model.IQ3_XS.gguf" `
  --source q3_k_s="$GGUF_DIR\model.Q3_K_S.gguf" `
  --source q3_k_m="$GGUF_DIR\model.Q3_K_M.gguf" `
  --source iq4_xs="$GGUF_DIR\model.IQ4_XS.gguf" `
  --variant c2_calib_knapsack_mixed `
  --output-dir ".\artifacts\local_pmra_mix" `
  --output-gguf "local_pmra_mix.gguf"
```

Check the artifact report:

```powershell
Get-Content .\artifacts\local_pmra_mix\artifact_report.md
```

## Colab

Colab works well for smoke runs and medium-sized mixes when the model fits in
the selected runtime. Use a GPU runtime for selector runs. The artifact build
step is mostly file I/O and CPU work, but it needs enough disk space for every
source GGUF plus the output GGUF.

Basic Colab setup:

```bash
git clone https://github.com/asystemoffields/PMRA.git
cd PMRA
python -m pip install -r requirements.txt
```

Store model and GGUF files in one of these places:

- the Colab VM filesystem for temporary work
- Google Drive for persistence across sessions
- Hugging Face cache, downloaded during the notebook

For public GGUF files, `huggingface_hub` can download sources directly:

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_dir = snapshot_download(
    repo_id="<base-model-repo>",
    local_dir="/content/models/base-model",
)

gguf_dir = "/content/models/base-model-gguf"
sources = {
    "iq2_m": hf_hub_download("<gguf-repo>", "<filename-IQ2_M.gguf>", local_dir=gguf_dir),
    "iq3_xs": hf_hub_download("<gguf-repo>", "<filename-IQ3_XS.gguf>", local_dir=gguf_dir),
    "q3_k_s": hf_hub_download("<gguf-repo>", "<filename-Q3_K_S.gguf>", local_dir=gguf_dir),
    "q3_k_m": hf_hub_download("<gguf-repo>", "<filename-Q3_K_M.gguf>", local_dir=gguf_dir),
    "iq4_xs": hf_hub_download("<gguf-repo>", "<filename-IQ4_XS.gguf>", local_dir=gguf_dir),
}
```

For gated models, use Colab Secrets or an environment variable for your Hugging
Face token. Do not paste tokens into notebooks that will be committed or shared.
For single-file checkpoints, pass the `.safetensors` file to `--hf`; for sharded
checkpoints, pass the `.safetensors.index.json` file.

Run the selector in a Colab shell cell:

```bash
python scripts/production_mixed_rate_transcoder_gate.py \
  --model-dir /content/models/base-model \
  --hf /content/models/base-model/model.safetensors.index.json \
  --source iq2_m=/content/models/base-model-gguf/<filename-IQ2_M.gguf> \
  --source iq3_xs=/content/models/base-model-gguf/<filename-IQ3_XS.gguf> \
  --source q3_k_s=/content/models/base-model-gguf/<filename-Q3_K_S.gguf> \
  --source q3_k_m=/content/models/base-model-gguf/<filename-Q3_K_M.gguf> \
  --source iq4_xs=/content/models/base-model-gguf/<filename-IQ4_XS.gguf> \
  --low-source iq2_m \
  --target-source iq3_xs \
  --high-sources q3_k_s,q3_k_m,iq4_xs \
  --tensor-profile <profile> \
  --layers <comma-separated-layer-ids> \
  --group-mode tensor \
  --prompt-source public \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --calib-split train \
  --eval-split validation \
  --calib-prompts 12 \
  --eval-prompts 32 \
  --calib-max-length 128 \
  --eval-max-length 192 \
  --candidate-variant c2_calib_knapsack_mixed \
  --output-dir /content/pmra-results/local-run
```

Then build the mixed GGUF:

```bash
python scripts/build_mixed_gguf_artifact.py \
  --result-json /content/pmra-results/local-run/result.json \
  --source iq2_m=/content/models/base-model-gguf/<filename-IQ2_M.gguf> \
  --source iq3_xs=/content/models/base-model-gguf/<filename-IQ3_XS.gguf> \
  --source q3_k_s=/content/models/base-model-gguf/<filename-Q3_K_S.gguf> \
  --source q3_k_m=/content/models/base-model-gguf/<filename-Q3_K_M.gguf> \
  --source iq4_xs=/content/models/base-model-gguf/<filename-IQ4_XS.gguf> \
  --variant c2_calib_knapsack_mixed \
  --output-dir /content/pmra-artifacts/local-mix \
  --output-gguf local_pmra_mix.gguf
```

## Optional Modal Runner

Modal is one supported runner, not part of the PMRA method. The repository keeps
`modal/modal_sprint.py` as a convenience harness for reproducible GPU jobs and
batch artifact builds.

Use Modal if you want managed GPUs and persistent remote storage. Keep any
provider-specific volume names, account names, and tokens in your own local
configuration rather than in committed docs.

The Modal entrypoints call the same two scripts described above:

- selector/evaluator: `scripts/production_mixed_rate_transcoder_gate.py`
- GGUF builder: `scripts/build_mixed_gguf_artifact.py`

## Verification

The builder writes:

```text
artifact_report.json
artifact_report.md
<output>.gguf
```

The report should show:

- `Status` is `GO`
- mismatched tensors is `0`
- the selected PMRA variant is the one you intended to build
- payload bytes are within the intended target/control budget

Smoke-load the GGUF with any llama.cpp-compatible frontend:

```powershell
<path-to-llama.cpp>\build\bin\llama-cli.exe `
  -m .\artifacts\local_pmra_mix\local_pmra_mix.gguf `
  -p "The compression method works because" `
  -n 16 `
  --no-warmup
```

## Scaling Up

After a smoke run passes:

- increase `--calib-prompts` and `--eval-prompts`
- increase `--calib-max-length` and `--eval-max-length`
- use `--group-mode layer_family` when the model family benefits from coarser
  allocation groups
- run a held-out public evaluation before publishing a mix
- compare against the uniform target/control and a same-budget random mix

Keep release-specific commands and result names in release docs or local run
logs. This file should stay portable and should not contain private paths,
service volume names, tokens, or account-specific cache locations.
