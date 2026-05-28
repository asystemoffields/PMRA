# PMRA Free-GPU Kaggle Handoff

_Last updated: 2026-05-28 evening._

## What we're doing

Running PMRA on two models via free Kaggle T4 GPUs:

1. **NVIDIA Nemotron-3-Nano-4B** — Mamba2-Transformer hybrid, `nemotron_h` tensor profile
2. **Qwen3.5-4B** — DeltaNet-Transformer hybrid, `qwen35` tensor profile

Two Kaggle notebooks live under `asystemoffields`:
- https://www.kaggle.com/code/asystemoffields/pmra-qwen35-4b
- https://www.kaggle.com/code/asystemoffields/pmra-nemotron-4b

GGUF source quants live in private Kaggle Datasets (separate quota from `/kaggle/working`):
- ✅ `asystemoffields/qwen35-4b-ggufs` — built, 7 GGUFs
- ⏳ `asystemoffields/nemotron3-nano-4b-ggufs` — slug reserved in config, not built yet

## Current state (2026-05-28)

### Qwen3.5 — probing in progress
- **Status:** cell 12 (PMRA probing) actively running, restarted with `calib_prompts=4` (was 12). Fresh JSONL checkpoint.
- **Why reduced calib_prompts:** original 12 prompts × ~25s per forward pass × Qwen3.5's slow DeltaNet path = ~5 min/probe = 24h+ total. Doesn't fit Kaggle's 12h session cap. Dropping to 4 prompts → ~1.7 min/probe → ~9.2h projected. May still need a 2nd session via checkpoint resume.
- **Slow DeltaNet path:** we *tried* installing `flash-linear-attention` to enable the fast path. fla on its own works, but it pulls `causal-conv1d` as a dependency, and causal-conv1d's CUDA fast path raises `Expected x.is_cuda() to be true` on Kaggle T4. Net result: we have to live with the torch fallback. The notebook source (`pmra_free_gpu.py`) still attempts `pip install flash-linear-attention` but skips `causal-conv1d`; if fla pulls it in transitively, uninstall manually before running cell 12.
- **When the user returns:** open the notebook URL, check whether cell 12 completed. If yes, cells 14-22 auto-ran (inspect, build artifact, perplexity, optional HF push). If no, click Run All to resume from checkpoint.

### Nemotron — prepped, not started
- Kaggle tab open at https://www.kaggle.com/code/asystemoffields/pmra-nemotron-4b/edit
- Notebook already re-imported with the latest source.
- `MODEL_KEY = "nemotron_4b"` (default), `BUILD_DATASET = True` (so first run will build the dataset).
- Blocked on the single-GPU-session constraint: needs Qwen3.5 to stop before Nemotron can start.

## Code changes shipped today

All on `main` of `github.com/asystemoffields/PMRA`. Most-recent first:

1. `Auto-install flash-linear-attention for qwen35 fast DeltaNet` — adds fla install for qwen35 profile (and explicit comment that we skip causal-conv1d).
2. `Resolve both Kaggle dataset mount layouts` — `/kaggle/input/<name>/` *and* `/kaggle/input/datasets/<user>/<name>/`. The newer layout is what Kaggle gave us today.
3. `Use Kaggle Datasets for GGUF sources, restore full high_sources` — adds `BUILD_DATASET=True` cell that downloads + pushes the GGUFs as a private Kaggle Dataset, then SystemExits before the HF download (so the GGUFs and HF model never need to coexist in /kaggle/working). Run-mode reads from the mounted dataset.
4. `Trust remote code for Qwen3.5; trim free-Kaggle high_sources` — superseded by the Datasets pivot but the `trust_remote_code=True` for qwen35 stays. Qwen3.5-4B has model_type `qwen3_5` which transformers doesn't know about until you upgrade.
5. `Upgrade transformers in setup cell` — cell 4 runs `pip install --upgrade transformers` because Kaggle's preinstalled version is too old for qwen3_5.
6. `Harden mamba-ssm install for Kaggle T4` — Nemotron path: `--no-build-isolation` and verify-import before continuing. Cell 4's silent install of mamba-ssm bit us in the morning; this stops it failing only at cell 5.

## Kaggle constraints we hit

- **1 concurrent GPU session** on free tier. Mix runs must be sequential, not parallel.
- **19.5 GiB `/kaggle/working`** — Qwen3.5's 7 GGUFs (~18GB) + HF weights (~9GB) overflow. Solved by hosting GGUFs in a Kaggle Dataset (separate ~100GB quota), mounted read-only.
- **12h max GPU session.** Probing takes longer than that in some configs; PMRA checkpoints to JSONL so restarting cell 12 resumes seamlessly.
- **30h/week of GPU quota.** Used ~5h on the failed-and-restarted Qwen3.5 attempts. Plenty left.
- **Console is IPython, not bash.** Use `!` prefix for shell.
- **Dataset mount path is** `/kaggle/input/datasets/<user>/<slug>/`, not the older `/kaggle/input/<slug>/`. Resolver tries both.

## When you return

The fast check:

1. Open https://www.kaggle.com/code/asystemoffields/pmra-qwen35-4b/edit
2. Look at cell 12's last `[c2] scoring group N/330` print. Latest checkpoint count is also in `/kaggle/working/output/checkpoints/allocation_rows.jsonl` (line count = probes done).
3. If 330 reached, cells 14-22 should have run automatically — find the artifact at `/kaggle/working/output/artifact/qwen35-4b-pmra.gguf`.
4. If still running, leave it.
5. If session timed out, click Run All. Cell 12 will skip already-done probes (`[c2] checkpoint hit group N/330`).

Then for Nemotron:

1. Stop the Qwen3.5 session (Run → Stop session) so the GPU slot frees.
2. Switch to the Nemotron tab. `BUILD_DATASET=True` is already set.
3. Run All. The dataset builder downloads 5 GGUFs (~13 GB), pushes a private dataset, exits.
4. Add Input → Datasets → search `nemotron3-nano-4b-ggufs` → add.
5. Flip `BUILD_DATASET=False` (top of cell 2).
6. Run All again. Reads GGUFs from the mount, downloads HF weights, runs probing.

## Local-side / harness notes

- The `cb` browser harness lives at `C:\Users\power\Documents\cb`. Patched today to honor `$env:CB_TAB_URL` — set it to a unique substring of the target tab's URL before any command, so multi-tab work is deterministic.
- Local Kaggle CLI is installed but `~/.kaggle/kaggle.json` is empty. Drop your API token there if you want `kaggle kernels output …` to work from this machine.
- All artifacts will live in `/kaggle/working/output/artifact/`. Download via Kaggle's UI file browser or `kaggle kernels output asystemoffields/pmra-qwen35-4b -p ./downloads/` once the CLI is configured.

## Key files

| File | Purpose |
|------|---------|
| `scripts/production_mixed_rate_transcoder_gate.py` | Core PMRA engine (now: `trust_remote_code=True` for qwen35 + nemotron_h) |
| `scripts/build_mixed_gguf_artifact.py` | Assembles mixed GGUF from selection result |
| `notebooks/pmra_free_gpu.py` | Notebook source (cell markers, single source of truth) |
| `notebooks/pmra_free_gpu.ipynb` | Kaggle-ready notebook, regenerated from the `.py` whenever the `.py` changes |
| `HANDOFF.md` | This file |
