# CPU Prober — GPU-free PMRA probing on the llama.cpp inference path

`scripts/cpu_prober.py` replaces the GPU probing loop for producing PMRA
mixes. It needs no torch, no HF model download, and no GPU — just numpy,
gguf-py, and llama.cpp CPU binaries. A 135M-model run takes ~15 minutes on a
desktop CPU; a 4B run fits comfortably in free Kaggle CPU kernels (which have
no weekly quota, unlike the 30h GPU cap).

## Why probe on CPU at all?

1. **Compute.** The v1 loop swaps each tensor group into a torch model and
   runs calibration forwards — O(groups × sources) full passes, hours of GPU
   time per model. That was the publishing bottleneck.
2. **Fidelity.** v1 measures NLL through torch-side dequantization on the HF
   model graph. The artifact ships to llama.cpp. For hybrid architectures
   (DeltaNet/Mamba paths) the two disagree — the Qwen3.5-4B b2 run showed
   every quant at NLL ~13.5 vs fp16 at 2.8, an eval-path artifact that
   contributed to a GRAY verdict. The CPU prober evaluates real spliced GGUFs
   with `llama-perplexity`: it measures exactly what users run.

## Two-tier design

**Tier 1 (free):** every (group, source) candidate is scored with an
imatrix-weighted SSE proxy. One `llama-imatrix` run on the calibration text
captures per-channel mean squared activations E[x_j^2]; the proxy for a
candidate is `sum_j E[x_j^2] * sum_i dW_ij^2` against a near-lossless
reference (f16 or Q8_0), computed with numpy dequantization only. This is the
activation-aware (AWQ-style) upgrade of the v2 `--triage weight_sse`
pre-rank.

**Tier 2 (cheap, parallel):** a proxy knapsack picks a provisional selection;
the candidates in and near the budget boundary (triage-style top-fraction +
band, capped by `--max-probes`) get empirical probes: a real single-promotion
GGUF is assembled by payload splicing and measured with `llama-perplexity`
on the calibration text. The final knapsack runs on empirical improvements
only; the unprobed tail is proxy-zeroed, mirroring `--triage` semantics.

Outputs are drop-in compatible with the existing toolchain:

- `checkpoints/allocation_rows.jsonl` — same row keys and the same
  `(group, source)` dedup key as the gate script and the b1/b2 Kaggle
  layer-split merge.
- `result.json` — consumed unchanged by `build_mixed_gguf_artifact.py`.
- Group names are byte-identical to `build_tensor_specs()` output for the
  matching profile (validated against the gate for `qwen`/`layer_family`).

## Single-machine usage

```bash
python scripts/cpu_prober.py \
  --source q2_k=... --source q4_k_m=... --source q8_0=... --source f16=... \
  --low-source q2_k --target-source q4_k_m --high-sources iq3_m,q3_k_m,q8_0 \
  --ref-source f16 \
  --calib-text calib.txt --eval-text eval.txt \
  --output-dir out --llama-bin /path/to/llama.cpp/build/bin
```

Then build the artifact exactly as before:

```bash
cd scripts && python build_mixed_gguf_artifact.py \
  --result-json ../out/result.json --output-dir ../out/artifact \
  --variant c2_calib_knapsack_mixed --source q2_k=... --source q8_0=... ...
```

## Distributed usage

Tier-2 probes shard by stable hash: `--stages tier2 --shard k/N`. Each worker
emits a partial `allocation_rows.jsonl`; merge with
`scripts/merge_allocation_rows.py` and run `--stages finalize`.

- **GitHub Actions** (`.github/workflows/cpu-prober.yml`): prepare →
  N-way probe matrix → finalize, all on free public-repo runners. Sized for
  ≤1B models (runners have ~14 GB disk).
- **Kaggle CPU kernels** (`notebooks/pmra_cpu_prober_kaggle.py`): no GPU
  quota consumed, big disk via mounted GGUF Datasets, concurrent sessions.
  This is the path for 4B+ models. Shard fan-in reuses the kernel_sources
  checkpoint-merge pattern from the b1/b2 runs.

## Knobs

| Flag | Default | Meaning |
|---|---|---|
| `--probe-fraction` | 0.15 | Tier-2 probes the top fraction of candidates by proxy score/MB |
| `--boundary-band` | 0.10 | plus this band past the greedy budget cutoff |
| `--max-probes` | 48 | hard cap on tier-2 probes (proxy-selection members first) |
| `--ctx` / `--chunks` | 512 / 24 | llama-perplexity window: 12k tokens per probe |
| `--ref-source` | f16 if present | tier-1 reference; use q8_0 at 4B+ to keep the imatrix run fast |

## Verdict semantics

`GO` — mix beats the target uniform quant NLL **and** the random-same-budget
control. `GRAY` — beats target but not the control (allocation isn't adding
value over budget alone). `NO-GO` — doesn't beat the target.
