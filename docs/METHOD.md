# Method - Production Mixed-Rate Allocation

Production Mixed-Rate Allocation (PMRA) is a method for mixing tensor payloads
from existing GGUF production quantizations of the same model.

To mix a model, PMRA starts from a low-bit GGUF source, measures which tensor
promotions buy the most calibration loss improvement per byte, solves the
allocation under a payload budget, and materializes the selected tensor payloads
as one standard GGUF. The result is a PMRA mix.

## Current Recipes

The current frontier recipe is Gemma 4 E2B-it with a knapsack selector:

- low source: `Q2_K`
- byte target/control: `Q3_K_S`
- stronger controls/sources: `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- allocation unit: full tensor
- selector: calibration NLL improvement under a multiple-choice knapsack budget
- calibration prompts: `24`
- public calibration: Wikitext-2 raw train
- public validation: Wikitext-2 raw validation, disjoint from calibration

The earlier Qwen3 public-calibrated recipe remains supporting evidence:

- low source: `IQ2_M`
- byte target/control: `IQ3_XS`
- stronger controls/sources: `Q2_K_L`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- allocation unit: full tensor
- selector: calibration forward-loss improvement per added byte, greedy ratio
- calibration prompts: `48`
- public calibration: Wikitext-2 raw train
- public validation/test: Wikitext-2 raw validation/test, disjoint from calibration
- cross-corpus checks: TinyStories validation and LAMBADA English test

The released Huihui Qwen3.5 4B abliterated recipe is a completed `qwen35`
profile example:

- low source: `IQ2_M`
- byte target/control: `IQ3_XS`
- stronger controls/sources: `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- allocation unit: layer-family tensor groups
- selector: calibration/weight-rank blended score under the target payload budget
- calibration prompts: `48`
- public calibration: Wikitext-2 raw train
- public validation: Wikitext-2 raw validation, disjoint from calibration
- released variant: `c2_calib_weight_blend_mixed`

The Granite 4.1 8B Heretic release is a completed `granite` profile example:

- upstream model: `heretic-org/IBM-granite-4.1-8b-heretic`
- low source: `IQ2_M`
- byte target/control: `IQ3_XS`
- stronger controls/sources: `Q2_K_S`, `Q2_K`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- allocation unit: layer-family tensor groups
- selector: multiple-choice knapsack under the target payload budget
- calibration prompts: `12`
- public calibration: Wikitext-2 raw train
- public validation/test: Wikitext-2 raw validation/test, disjoint from calibration
- released variant: `c2_calib_knapsack_mixed`

## Mix Algorithm

1. Load a full-precision HF reference model.
2. Load several GGUF versions of the same model.
3. Start from the low-source GGUF tensor set.
4. For each candidate tensor/source promotion:
   - patch the model with that promoted tensor
   - run calibration forward loss
   - record quality gain and added bytes
5. Choose promotions while staying under the target payload budget:
   - greedy ratio selector for the earlier Qwen mix
   - exact scaled multiple-choice knapsack when the byte budget is compact
   - Pareto-pruned knapsack frontier when the exact state space is larger
   - optional genetic search and simulated annealing refinements around scored
     candidates, plus direct search controls that skip per-tensor scoring
6. Evaluate the resulting full mix on held-out prompts.
7. Compare against:
   - target uniform source
   - stronger uniform production control
   - random same-budget mixed allocation
   - weight-only selector controls
8. Mix the selected tensor payloads into one standard GGUF.
9. Reuse the frozen selection on a public dataset eval before making any broad
   quality claim.
10. If the frozen selector fails public transfer, rerun the same allocation
    algorithm with public calibration data and held-out public evaluation.

## Current Next Work

- Current strongest mix is the Gemma 4 E2B-it knapsack GGUF.
- Huihui Qwen3.5 4B abliterated is published as a `qwen35` layer-family
  weight-blend PMRA GGUF.
- Granite 4.1 8B Heretic is published as a `granite` layer-family knapsack PMRA
  GGUF with Heretic upstream attribution.
- Qwen3 public-calibrated results remain supporting evidence across Wikitext,
  TinyStories, and LAMBADA.
- Broader public benchmark coverage comes next for Gemma, especially repeating
  the knapsack selector across seeds and model families.
- Tensor-level allocation is coarse; page/block allocation may improve the
  frontier but needs careful metadata and runtime accounting.
- `general.file_type` cannot express mixed tensor allocation, so PMRA mixes
  include explicit `pmra.*` metadata.
