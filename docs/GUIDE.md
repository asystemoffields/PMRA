# How PMRA Works (and Why It Makes Better GGUFs)

## The one-sentence version

PMRA makes smaller, better GGUF files by giving more bits to the parts of a
model that matter and fewer bits to the parts that don't.

---

## The problem with uniform quantization

When you download a Q4_K_M GGUF from HuggingFace, every weight tensor in the
model gets roughly the same treatment: about 4.5 bits per weight. The first
layer, the middle layers, the last layer, the attention weights, the MLP
weights -- all 4.5 bits.

This is like packing a suitcase by giving every item the same amount of space.
Your passport gets the same room as your socks. It works, but it's wasteful --
you could fit more if you were smarter about what goes where.

Neural networks have the same property. Not all parts of a model contribute
equally to its output:

- **Edge layers** (the first few and last few) are the bottleneck -- they
  transform between the token vocabulary and the model's internal
  representations. Damage here cascades through everything.
- **Middle layers** are highly redundant. In some models, middle layers can be
  compressed 700x with barely measurable quality loss. They're doing important
  work, but they're doing it with enormous overkill.
- **Attention output projections** (`o_proj`) have no normalization layer before
  them in most architectures, which means quantization errors there can't be
  absorbed or corrected by anything downstream.
- **MLP gate and up projections** are the most robust to quantization -- their
  errors get filtered through activation functions that naturally clip noise.

Uniform quantization ignores all of this. It gives the same 4.5 bits to a
middle-layer gate projection (which could survive at 2 bits) and to a
first-layer attention output (which really wants 6 bits).

## What PMRA does differently

PMRA treats quantization as a **budget allocation problem**.

Given a file size budget (say, "fit in 3 GB"), PMRA asks: what's the best way
to distribute bits across every tensor in the model?

The answer is a **mixed-precision GGUF** -- a single, standard GGUF file where
different tensors use different quantization types. The first layer's attention
output might be Q6_K (6.5 bits), the middle MLP gates might be IQ2_XXS (2
bits), and the embeddings might be Q4_K (4.5 bits). The file is the same total
size as a uniform Q3_K_M, but the quality is better because the bits went where
they're needed.

This is a standard GGUF file. It loads in Ollama, LM Studio, llama.cpp --
anything that reads GGUF. No custom runtime, no patches, no hacks. GGUF has
always supported per-tensor quantization types; PMRA just uses that capability
deliberately.

## Two ways to know what matters

The core question in PMRA is: **how do you know which tensors need more bits?**

There are two approaches, and they complement each other.

### Approach 1: Probing (standard PMRA)

The direct approach: try it and measure.

1. Start with a cheap quantization (say, IQ2_M) as the baseline.
2. For each tensor, temporarily upgrade it to a better format (Q4_K, Q6_K, etc.).
3. Run the model on calibration text and measure how much the output improves.
4. Record the improvement and the byte cost of the upgrade.
5. Use a knapsack solver to pick the best set of upgrades under your byte budget.

This is empirical -- you're directly measuring what each tensor upgrade buys
you. The downside is cost: you need to run inference once per tensor per
candidate format. For a 7B model with 226 tensors and 5 candidate formats,
that's over 1,000 forward passes.

### Approach 2: Profiling (Fisher-guided PMRA)

The structural approach: analyze the model's weight matrices to predict
importance without running all those forward passes.

For each weight matrix in the model, compute:

- **Stable rank** -- how many effective dimensions of information the matrix
  carries. A matrix with stable rank 40 has its information concentrated in ~40
  dimensions; one with stable rank 400 is spread across 400. Higher rank means
  more bits needed to faithfully represent the information.

- **Condition number** -- how sensitive the matrix is to small perturbations.
  High condition number means quantization errors get amplified.

- **Weight distribution** -- how many outlier weights exist, how heavy the tails
  are. Matrices with lots of outliers are harder to quantize uniformly.

- **Entropy utilization** -- how efficiently the matrix uses N-bit quantization.
  If a matrix only uses 28% of its 4-bit quantization grid, it could probably
  survive at 3 bits.

These metrics, combined with structural knowledge about which layer positions
and matrix roles are critical, produce a per-tensor importance score. A knapsack
solver then allocates quantization types exactly like the probing approach, but
using the profiling scores instead of measured NLL improvements.

The advantage: profiling runs once (a few minutes on GPU) and works for any
target file size. You don't re-run inference for each byte budget. The
importance map is a property of the model, not of a particular quantization
target.

### Where Fisher information comes in

The name "Fisher-guided" refers to the [Fisher information
matrix](https://en.wikipedia.org/wiki/Fisher_information), a concept from
statistics that measures how sensitive a model's output is to changes in its
parameters. In plain terms: the Fisher diagonal tells you "if you wiggle this
weight slightly, how much does the model's output change?"

Weights with high Fisher information are load-bearing -- quantization errors
there actually affect what the model says. Weights with low Fisher information
are redundant -- you can round them aggressively and the model barely notices.

A 2025 paper ([arXiv:2505.12988](https://arxiv.org/abs/2505.12988)) proved
mathematically that Fisher information gives the **optimal** per-tensor bit
allocation: it minimizes the distance between the original and quantized model's
outputs for any given file size. PMRA's profiling approach approximates this
through the SVD and distribution metrics described above, plus direct Fisher
diagonal computation when calibration data is available.

## What the output looks like

A PMRA recipe is a list of tensor-name-to-quantization-type assignments:

```
blk.0.attn_output.weight=q6_k
blk.0.attn_q.weight=q5_k
blk.0.ffn_gate.weight=q3_k
blk.0.ffn_down.weight=q4_k
...
blk.15.ffn_gate.weight=iq2_xxs
blk.15.ffn_up.weight=iq2_xs
...
blk.31.attn_output.weight=q5_k
token_embd.weight=q4_k
output.weight=q5_k
```

This file feeds directly into llama.cpp's quantizer:

```bash
llama-quantize \
  --imatrix imatrix.gguf \
  --tensor-type-file recipe.txt \
  model-f16.gguf model-pmra.gguf Q4_K_M
```

The result is one `.gguf` file you can use like any other:

```bash
ollama run model-pmra.gguf
```

## The bathtub curve

One of the most consistent findings across models is the **bathtub curve** of
layer importance:

```
Importance
  ^
  |##                                                          ##
  |###                                                        ###
  |####                                                      ####
  |#####                                                    #####
  |######                                                  ######
  |########                                              ########
  |###########                                        ###########
  |################                              ################
  |#########################        #########################
  |###########################################################
  +-----------------------------------------------------------> Layer
   0  1  2  3  4  5  ...  middle  ...  28  29  30  31
```

The first and last few layers are critical. The middle layers are remarkably
compressible. This pattern appears in every transformer we've profiled (SmolLM,
OLMo, LLaMA, Qwen) -- it seems to be a fundamental property of how these models
organize information.

PMRA exploits this by giving edge layers 5-6 bits and middle layers 2-3 bits,
achieving the same quality as a uniform 4-bit quantization at 30-40% less space.

## What you need to try it

**For probing (standard PMRA):**
- The model weights (HuggingFace format)
- Pre-quantized GGUF files at multiple levels (available on HF for most models)
- GPU (A10G or better for 7B models)
- ~1-2 hours of compute

**For profiling (Fisher-guided PMRA):**
- The model weights (HuggingFace format)
- A profiling run (~10 minutes on GPU, produces `per_tensor.json`)
- No pre-quantized GGUFs needed
- llama.cpp (for final quantization and evaluation)

See [REPRODUCE.md](REPRODUCE.md) for step-by-step instructions.

## Frequently asked questions

**Does this actually help? Aren't existing quants already good?**

It depends on the model. For some models (especially those with strong bathtub
curves), PMRA can match Q4_K_M quality at Q3_K_M file size. For others where
the existing quants are already near-optimal, the gains are smaller. The
profiling data tells you upfront whether there's room to improve.

**How is this different from what Unsloth does?**

Unsloth's Dynamic 2.0 does essentially the same thing -- mixed-precision GGUF
with per-tensor type selection. Their approach uses brute-force KL divergence
benchmarking (150+ configurations per model) to find recipes. PMRA's probing
approach uses calibration NLL. Fisher-guided PMRA uses structural profiling.
All three are trying to solve the same allocation problem with different
importance signals.

**Can I use this with any model?**

Yes. The recipe generator works with any transformer-family model that uses the
standard HuggingFace naming conventions (LLaMA, Mistral, OLMo, Qwen, Gemma,
Phi, etc.). The profiling and allocation are model-agnostic -- you just need
`per_tensor.json` for your model.

**What about MoE models?**

MoE models are where mixed-precision quantization helps *most*. Router weights
need high precision (they control which expert runs), but individual expert
weights can often be aggressive. PMRA handles this through its role-sensitivity
system.

**Does the output work with Ollama / LM Studio / llama.cpp?**

Yes, unconditionally. The output is a standard GGUF file with per-tensor type
metadata. Every GGUF consumer has supported this since the GGUF format was
introduced. There is nothing custom about a PMRA GGUF.

## Further reading

- [README.md](../README.md) -- implementation-level details, knapsack solver, repo layout
- [METHOD.md](METHOD.md) -- technical method notes
- [REPRODUCE.md](REPRODUCE.md) -- reproduction instructions
- [PMRA on HuggingFace](https://huggingface.co/collections/Asystemoffields/pmra-6a1067359be8a5f82021efe5) -- released model mixes
- [arXiv:2505.12988](https://arxiv.org/abs/2505.12988) -- "Optimal Formats for Weight Quantisation" (Fisher-optimal bit allocation proof)
