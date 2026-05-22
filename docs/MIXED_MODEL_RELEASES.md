# Mixed Model Releases

This is the chronological PMRA release list, newest to oldest. The README
describes the method; this file records released mixes, source checkpoints, and
where to find their reports.

Public GGUF mixes are also grouped in the Hugging Face
[PMRA collection](https://huggingface.co/collections/Asystemoffields/pmra-6a1067359be8a5f82021efe5).

## Granite 4.1 8B Heretic

- release repo:
  `https://huggingface.co/Asystemoffields/IBM-granite-4.1-8b-heretic-PMRA-GGUF`
- release mix: `granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget.gguf`
- base model: `heretic-org/IBM-granite-4.1-8b-heretic`
- GGUF source repo: `mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF`
- selector: `c2_calib_knapsack_mixed`
- target/control budget: `IQ3_XS`
- report: [Granite 4.1 8B Heretic PMRA](GRANITE4_1_8B_HERETIC_PMRA.md)
- model card source: [Granite HF Model Card](GRANITE4_1_8B_HERETIC_HF_MODEL_CARD.md)

## Ministral 3 8B Instruct

- release repo:
  `https://huggingface.co/Asystemoffields/Ministral-3-8B-Instruct-PMRA-GGUF`
- release mixes:
  `ministral3_8b_pmra_knapsack_iq3xs_budget.gguf`,
  `ministral3_8b_pmra_knapsack_3p2.gguf`
- base model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- GGUF source repo: `bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF`
- selector: `c2_calib_knapsack_mixed`
- target/control budget: `IQ3_XS`
- report: [Ministral 3 8B Instruct PMRA](MINISTRAL3_8B_INSTRUCT_PMRA.md)
- model card source: [Ministral HF Model Card](MINISTRAL3_8B_INSTRUCT_HF_MODEL_CARD.md)

## Huihui Qwen3.5 4B Abliterated

- release repo:
  `https://huggingface.co/Asystemoffields/Huihui-Qwen3.5-4B-Abliterated-PMRA-GGUF`
- release mix: `huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf`
- base model: `huihui-ai/Huihui-Qwen3.5-4B-abliterated`
- GGUF source repo: `mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF`
- selector: `c2_calib_weight_blend_mixed`
- target/control budget: `IQ3_XS`
- report: [Qwen3.5 Abliterated PMRA](QWEN35_ABLITERATED_PMRA.md)
- model card source: [Qwen3.5 HF Model Card](HUIHUI_QWEN35_4B_ABLITERATED_HF_MODEL_CARD.md)

## Gemma 4 E2B-it

- release repo:
  `https://huggingface.co/Asystemoffields/gemma-4-E2B-it-PMRA-GGUF`
- release mix: `gemma4_e2b_it_pmra_calib_knapsack.gguf`
- base model: `google/gemma-4-E2B-it`
- GGUF source repo: `mradermacher/gemma-4-E2B-it-GGUF`
- selector: `c2_calib_knapsack_mixed`
- target/control budget: `Q3_K_S`
- report: [Gemma 4 E2B-it Release](GEMMA4_E2B_IT_RELEASE.md)
- model card source: [Gemma HF Model Card](GEMMA4_E2B_IT_HF_MODEL_CARD.md)

## Qwen3 1.7B

- release repo:
  `https://huggingface.co/Asystemoffields/Qwen3-1.7B-PMRA-IQ3XS-budget-GGUF`
- release mix: `qwen17_publiccal_pmra_calib_greedy.gguf`
- base model: `Qwen/Qwen3-1.7B`
- GGUF source repo: `bartowski/Qwen_Qwen3-1.7B-GGUF`
- selector: `c2_calib_greedy_mixed`
- target/control budget: `IQ3_XS`
- model card source: [Qwen3 1.7B HF Model Card](HF_MODEL_CARD.md)
- broader evidence: [Evidence Ledger](EVIDENCE.md)
