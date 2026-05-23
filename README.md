# Production Mixed-Rate Allocation

Production Mixed-Rate Allocation (PMRA) is a method for mixing existing GGUF
quantizations of the same model into one standard GGUF.

To mix a model with PMRA, start with a small production GGUF, then promote
selected tensor payloads to stronger production GGUF formats where calibration
shows the extra bytes buy quality. The mixer writes the selected payloads into a
single GGUF that loads in normal GGUF runtimes.

PMRA is not a new quantizer and does not need a custom runtime. It is a
selection-and-materialization method over production GGUF payloads. Each mix
records its source allocation in `pmra.*` metadata and in the artifact report.

Released model mixes, metrics, and upstream attribution are tracked separately
from this method overview. The Hugging Face collection for public PMRA GGUF
mixes is [PMRA](https://huggingface.co/collections/Asystemoffields/pmra-6a1067359be8a5f82021efe5).

## Vocabulary

- mix, verb: run PMRA selection and build the resulting mixed GGUF.
- mix, noun: the resulting GGUF artifact with tensor payloads from more than
  one source quantization.
- source: one existing GGUF quantization used as a tensor-payload donor.
- target/control: the uniform GGUF budget or quality point the mix is compared
  against.
- selector: the allocation strategy, usually calibration knapsack or a recorded
  comparison selector.
- payload budget: the tensor-payload byte ceiling the mix must fit under.

## How PMRA Works

PMRA treats mixed quantization as a byte-budgeted allocation problem.

1. Pick a low-bit source GGUF, such as `IQ2_M`.
2. Pick a target/control budget, such as `IQ3_XS`.
3. Load stronger GGUF sources from the same checkpoint.
4. For each tensor group, temporarily promote it to each stronger source.
5. Measure calibration NLL improvement and added payload bytes.
6. Select the best set of promotions under the byte budget.
7. Mix the selected tensor payloads into one standard GGUF.
8. Evaluate the full mix against uniform controls and random same-budget mixes.

Uniform quantization spends one format choice broadly across the model, even
though tensors are not equally sensitive. PMRA spends stronger formats only
where calibration shows the bytes matter.

## Why Knapsack

The default selector is a multiple-choice knapsack:

```text
maximize total calibration NLL improvement
subject to total extra bytes <= payload budget
and at most one source choice per tensor group
```

Each candidate promotion has a value, which is measured calibration improvement,
and a cost, which is added tensor payload bytes. Knapsack is a better fit than a
pure greedy ratio because the whole byte budget matters: several modest tensor
promotions can beat one large promotion even if the large one has a tempting
single-tensor score.

When the byte state space is compact, PMRA uses an exact scaled dynamic program.
When it is too large, it keeps a Pareto-pruned frontier so the run stays
practical. The selected mix still has to pass held-out evaluation because tensor
interactions are real and calibration is the selection objective, not the final
claim.

The repo also supports search refinements around that default: seeded genetic
search, direct genetic search, seeded simulated annealing, and direct simulated
annealing. These are tested as candidate finders, then judged by the same
held-out controls as knapsack.

## Prior Art And Positioning

PMRA is new in this repo as a GGUF-native mixing workflow, but it sits inside a
longer line of mixed-precision and sensitivity-aware compression work.

Relevant predecessors include:

- [HAQ](https://arxiv.org/abs/1811.08886), which used hardware feedback to
  choose mixed-precision quantization policies.
- [HAWQ-V2](https://arxiv.org/abs/1911.03852), which used Hessian-aware
  sensitivity analysis for mixed-precision quantization.
- [LLM.int8()](https://arxiv.org/abs/2208.07339), which used mixed-precision
  decomposition to preserve transformer outlier dimensions.
- [GPTQ](https://arxiv.org/abs/2210.17323), which made post-training LLM weight
  quantization practical at large scale.
- [AWQ](https://arxiv.org/abs/2306.00978), which protected salient weights using
  activation-aware calibration.
- [SpQR](https://arxiv.org/abs/2306.03078) and
  [SqueezeLLM](https://arxiv.org/abs/2306.07629), which combined dense
  low-bit quantization with special handling for sensitive or outlier weights.
- [OmniQuant](https://arxiv.org/abs/2308.13137), which used calibration to
  optimize quantization parameters across LLM settings.

## Applications

PMRA is useful anywhere a deployable GGUF has to balance quality, file size,
memory budget, and runtime compatibility without introducing a custom inference
path.

PMRA is useful when you want to:

- hit a local memory budget more precisely than one uniform preset allows
- recover quality at the same size by protecting sensitive tensors
- publish one normal GGUF instead of a custom runtime path
- reuse public quantization ladders rather than recomputing every quant
- expose exactly where the bytes went through `pmra.*` metadata and reports
- create practical local-model deployment points for laptops, small GPUs, and apps

## How To Read This Repo

For a quick pass through the repo, use this order:

1. This README - the showcase-friendly explanation of what PMRA is, how mixing
   works, and why knapsack is the default selector.
2. [PMRA Hugging Face Collection](https://huggingface.co/collections/Asystemoffields/pmra-6a1067359be8a5f82021efe5) -
   public GGUF mixes and model cards.
3. [Artifact Index](docs/ARTIFACT_INDEX.md) - released mixes, reports, and
   metrics.
4. [Method](docs/METHOD.md) - implementation-level method notes.
5. [Reproduce](docs/REPRODUCE.md) - local, Colab, and optional Modal paths for
   rebuilding selector results and GGUF artifacts.
6. [Evidence Ledger](docs/EVIDENCE.md) - full research trail, including failed
   and superseded gates.

## Repository Layout

```text
docs/       Human-facing release notes, method notes, evidence, reproduction.
scripts/    PMRA selector, GGUF mixer, public evaluators, helpers.
modal/      Modal A100 harness used for the public-calibrated runs.
results/    Result cards and JSON reports for released and validation mixes.
artifacts/  Earlier artifact reports, not the GGUF files themselves.
tools/      Hugging Face upload and verification helpers.
```

## Install

```bash
pip install -r requirements.txt
```

Full selector runs expect access to the base model weights and matching GGUF
source files. They can run locally, in Colab, on rented GPU machines, or through
the optional Modal harness. Local CPU runs are useful for small checks, but the
larger selector runs are GPU-heavy.

## Minimal Reproduction Shape

1. Run public-calibrated PMRA mix selection on Wikitext-2 raw train/validation.
2. Evaluate the frozen selection on held-out public text.
3. Mix selected tensor payloads into one GGUF.
4. Load and smoke-test the GGUF with llama.cpp.

Exact commands are in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Attribution

PMRA code and docs in this repo are released under Apache-2.0. Individual mixes
inherit the licensing and attribution requirements of their base models and GGUF
source quantizations; release-specific attribution is tracked in the model
release docs.
