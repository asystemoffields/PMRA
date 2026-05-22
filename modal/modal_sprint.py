from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = os.environ.get("PMRA_MODAL_APP", "pmra-sprint")
VOLUME_NAME = os.environ.get("PMRA_MODAL_VOLUME", "pmra-cache")
MODEL_ID = "Qwen/Qwen3-1.7B-Base"
MODEL_DIR = "/cache/models/qwen3-1.7b-base"
HF_FILE = f"{MODEL_DIR}/model.safetensors"
IQ4_FILE = "/cache/models/Qwen_Qwen3-1.7B-IQ4_XS.gguf"
RESULT_ROOT = os.environ.get("PMRA_RESULT_ROOT", "/cache/results/pmra")
BASELINE_REPO = "bartowski/Qwen_Qwen3-1.7B-GGUF"
BASELINE_GGUFS = {
    "iq3_m": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-IQ3_M.gguf",
    },
    "iq3_xs": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-IQ3_XS.gguf",
    },
    "iq2_m": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-IQ2_M.gguf",
    },
    "iq4_xs": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-IQ4_XS.gguf",
    },
    "q2_k": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-Q2_K.gguf",
    },
    "q2_k_l": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-Q2_K_L.gguf",
    },
    "q3_k_m": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-Q3_K_M.gguf",
    },
    "q3_k_s": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-Q3_K_S.gguf",
    },
    "q4_k_m": {
        "repo_id": BASELINE_REPO,
        "filename": "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
    },
}
MODEL_CONFIGS = {
    "qwen3_1p7b": {
        "model_id": "Qwen/Qwen3-1.7B",
        "model_dir": "/cache/models/qwen3-1.7b",
        "hf_file": "/cache/models/qwen3-1.7b/model.safetensors",
        "baseline_repo": "bartowski/Qwen_Qwen3-1.7B-GGUF",
        "layers": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        "baseline_ggufs": {
            "iq3_m": "Qwen_Qwen3-1.7B-IQ3_M.gguf",
            "iq3_xs": "Qwen_Qwen3-1.7B-IQ3_XS.gguf",
            "iq2_m": "Qwen_Qwen3-1.7B-IQ2_M.gguf",
            "iq4_xs": "Qwen_Qwen3-1.7B-IQ4_XS.gguf",
            "q2_k": "Qwen_Qwen3-1.7B-Q2_K.gguf",
            "q2_k_l": "Qwen_Qwen3-1.7B-Q2_K_L.gguf",
            "q3_k_m": "Qwen_Qwen3-1.7B-Q3_K_M.gguf",
            "q3_k_s": "Qwen_Qwen3-1.7B-Q3_K_S.gguf",
            "q4_k_m": "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
        },
    },
    "qwen3_0p6b_base": {
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "model_dir": "/cache/models/qwen3-0.6b-base",
        "hf_file": "/cache/models/qwen3-0.6b-base/model.safetensors",
        "baseline_repo": "bartowski/Qwen_Qwen3-0.6B-GGUF",
        "layers": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        "baseline_ggufs": {
            "iq3_m": "Qwen_Qwen3-0.6B-IQ3_M.gguf",
            "iq3_xs": "Qwen_Qwen3-0.6B-IQ3_XS.gguf",
            "iq2_m": "Qwen_Qwen3-0.6B-IQ2_M.gguf",
            "iq4_xs": "Qwen_Qwen3-0.6B-IQ4_XS.gguf",
            "q2_k": "Qwen_Qwen3-0.6B-Q2_K.gguf",
            "q2_k_l": "Qwen_Qwen3-0.6B-Q2_K_L.gguf",
            "q3_k_m": "Qwen_Qwen3-0.6B-Q3_K_M.gguf",
            "q3_k_s": "Qwen_Qwen3-0.6B-Q3_K_S.gguf",
            "q4_k_m": "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
        },
    },
    "huihui_qwen35_4b_abliterated": {
        "model_id": "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        "model_dir": "/cache/models/huihui-qwen35-4b-abliterated",
        "hf_file": "/cache/models/huihui-qwen35-4b-abliterated/model.safetensors.index.json",
        "baseline_repo": "mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF",
        "tensor_profile": "qwen35",
        "layers": ",".join(str(i) for i in range(32)),
        "baseline_ggufs": {
            "iq1_m": "Huihui-Qwen3.5-4B-abliterated.i1-IQ1_M.gguf",
            "iq1_s": "Huihui-Qwen3.5-4B-abliterated.i1-IQ1_S.gguf",
            "iq2_m": "Huihui-Qwen3.5-4B-abliterated.i1-IQ2_M.gguf",
            "iq2_s": "Huihui-Qwen3.5-4B-abliterated.i1-IQ2_S.gguf",
            "iq2_xs": "Huihui-Qwen3.5-4B-abliterated.i1-IQ2_XS.gguf",
            "iq2_xxs": "Huihui-Qwen3.5-4B-abliterated.i1-IQ2_XXS.gguf",
            "iq3_m": "Huihui-Qwen3.5-4B-abliterated.i1-IQ3_M.gguf",
            "iq3_s": "Huihui-Qwen3.5-4B-abliterated.i1-IQ3_S.gguf",
            "iq3_xs": "Huihui-Qwen3.5-4B-abliterated.i1-IQ3_XS.gguf",
            "iq3_xxs": "Huihui-Qwen3.5-4B-abliterated.i1-IQ3_XXS.gguf",
            "iq4_nl": "Huihui-Qwen3.5-4B-abliterated.i1-IQ4_NL.gguf",
            "iq4_xs": "Huihui-Qwen3.5-4B-abliterated.i1-IQ4_XS.gguf",
            "q2_k": "Huihui-Qwen3.5-4B-abliterated.i1-Q2_K.gguf",
            "q2_k_s": "Huihui-Qwen3.5-4B-abliterated.i1-Q2_K_S.gguf",
            "q3_k_s": "Huihui-Qwen3.5-4B-abliterated.i1-Q3_K_S.gguf",
            "q3_k_m": "Huihui-Qwen3.5-4B-abliterated.i1-Q3_K_M.gguf",
            "q3_k_l": "Huihui-Qwen3.5-4B-abliterated.i1-Q3_K_L.gguf",
            "q4_0": "Huihui-Qwen3.5-4B-abliterated.i1-Q4_0.gguf",
            "q4_1": "Huihui-Qwen3.5-4B-abliterated.i1-Q4_1.gguf",
            "q4_k_s": "Huihui-Qwen3.5-4B-abliterated.i1-Q4_K_S.gguf",
            "q4_k_m": "Huihui-Qwen3.5-4B-abliterated.i1-Q4_K_M.gguf",
            "q5_k_s": "Huihui-Qwen3.5-4B-abliterated.i1-Q5_K_S.gguf",
            "q5_k_m": "Huihui-Qwen3.5-4B-abliterated.i1-Q5_K_M.gguf",
            "q6_k": "Huihui-Qwen3.5-4B-abliterated.i1-Q6_K.gguf",
        },
    },
    "ministral3_3b_instruct": {
        "model_id": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "model_dir": "/cache/models/ministral3-3b-instruct-2512-bf16",
        "hf_file": "/cache/models/ministral3-3b-instruct-2512-bf16/model.safetensors",
        "baseline_repo": "bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF",
        "tensor_profile": "mistral3",
        "layers": ",".join(str(i) for i in range(26)),
        "baseline_ggufs": {
            "iq2_m": "mistralai_Ministral-3-3B-Instruct-2512-IQ2_M.gguf",
            "iq3_m": "mistralai_Ministral-3-3B-Instruct-2512-IQ3_M.gguf",
            "iq3_xs": "mistralai_Ministral-3-3B-Instruct-2512-IQ3_XS.gguf",
            "iq3_xxs": "mistralai_Ministral-3-3B-Instruct-2512-IQ3_XXS.gguf",
            "iq4_nl": "mistralai_Ministral-3-3B-Instruct-2512-IQ4_NL.gguf",
            "iq4_xs": "mistralai_Ministral-3-3B-Instruct-2512-IQ4_XS.gguf",
            "q2_k": "mistralai_Ministral-3-3B-Instruct-2512-Q2_K.gguf",
            "q2_k_l": "mistralai_Ministral-3-3B-Instruct-2512-Q2_K_L.gguf",
            "q3_k_s": "mistralai_Ministral-3-3B-Instruct-2512-Q3_K_S.gguf",
            "q3_k_m": "mistralai_Ministral-3-3B-Instruct-2512-Q3_K_M.gguf",
            "q3_k_l": "mistralai_Ministral-3-3B-Instruct-2512-Q3_K_L.gguf",
            "q3_k_xl": "mistralai_Ministral-3-3B-Instruct-2512-Q3_K_XL.gguf",
            "q4_k_s": "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_S.gguf",
            "q4_k_m": "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
            "q4_k_l": "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_L.gguf",
        },
    },
    "ministral3_8b_instruct": {
        "model_id": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "model_dir": "/cache/models/ministral3-8b-instruct-2512-bf16",
        "hf_file": "/cache/models/ministral3-8b-instruct-2512-bf16/model.safetensors.index.json",
        "baseline_repo": "bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF",
        "tensor_profile": "mistral3",
        "layers": ",".join(str(i) for i in range(34)),
        "baseline_ggufs": {
            "iq2_m": "mistralai_Ministral-3-8B-Instruct-2512-IQ2_M.gguf",
            "iq3_m": "mistralai_Ministral-3-8B-Instruct-2512-IQ3_M.gguf",
            "iq3_xs": "mistralai_Ministral-3-8B-Instruct-2512-IQ3_XS.gguf",
            "iq3_xxs": "mistralai_Ministral-3-8B-Instruct-2512-IQ3_XXS.gguf",
            "iq4_nl": "mistralai_Ministral-3-8B-Instruct-2512-IQ4_NL.gguf",
            "iq4_xs": "mistralai_Ministral-3-8B-Instruct-2512-IQ4_XS.gguf",
            "q2_k": "mistralai_Ministral-3-8B-Instruct-2512-Q2_K.gguf",
            "q2_k_l": "mistralai_Ministral-3-8B-Instruct-2512-Q2_K_L.gguf",
            "q3_k_s": "mistralai_Ministral-3-8B-Instruct-2512-Q3_K_S.gguf",
            "q3_k_m": "mistralai_Ministral-3-8B-Instruct-2512-Q3_K_M.gguf",
            "q3_k_l": "mistralai_Ministral-3-8B-Instruct-2512-Q3_K_L.gguf",
            "q3_k_xl": "mistralai_Ministral-3-8B-Instruct-2512-Q3_K_XL.gguf",
            "q4_k_s": "mistralai_Ministral-3-8B-Instruct-2512-Q4_K_S.gguf",
            "q4_k_m": "mistralai_Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
            "q4_k_l": "mistralai_Ministral-3-8B-Instruct-2512-Q4_K_L.gguf",
        },
    },
    "ministral3_14b_instruct": {
        "model_id": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "model_dir": "/cache/models/ministral3-14b-instruct-2512-bf16",
        "hf_file": "/cache/models/ministral3-14b-instruct-2512-bf16/model.safetensors.index.json",
        "baseline_repo": "bartowski/mistralai_Ministral-3-14B-Instruct-2512-GGUF",
        "tensor_profile": "mistral3",
        "layers": ",".join(str(i) for i in range(40)),
        "baseline_ggufs": {
            "iq2_m": "mistralai_Ministral-3-14B-Instruct-2512-IQ2_M.gguf",
            "iq2_s": "mistralai_Ministral-3-14B-Instruct-2512-IQ2_S.gguf",
            "iq3_m": "mistralai_Ministral-3-14B-Instruct-2512-IQ3_M.gguf",
            "iq3_xs": "mistralai_Ministral-3-14B-Instruct-2512-IQ3_XS.gguf",
            "iq3_xxs": "mistralai_Ministral-3-14B-Instruct-2512-IQ3_XXS.gguf",
            "iq4_nl": "mistralai_Ministral-3-14B-Instruct-2512-IQ4_NL.gguf",
            "iq4_xs": "mistralai_Ministral-3-14B-Instruct-2512-IQ4_XS.gguf",
            "q2_k": "mistralai_Ministral-3-14B-Instruct-2512-Q2_K.gguf",
            "q2_k_l": "mistralai_Ministral-3-14B-Instruct-2512-Q2_K_L.gguf",
            "q3_k_s": "mistralai_Ministral-3-14B-Instruct-2512-Q3_K_S.gguf",
            "q3_k_m": "mistralai_Ministral-3-14B-Instruct-2512-Q3_K_M.gguf",
            "q3_k_l": "mistralai_Ministral-3-14B-Instruct-2512-Q3_K_L.gguf",
            "q3_k_xl": "mistralai_Ministral-3-14B-Instruct-2512-Q3_K_XL.gguf",
            "q4_0": "mistralai_Ministral-3-14B-Instruct-2512-Q4_0.gguf",
            "q4_1": "mistralai_Ministral-3-14B-Instruct-2512-Q4_1.gguf",
            "q4_k_s": "mistralai_Ministral-3-14B-Instruct-2512-Q4_K_S.gguf",
            "q4_k_m": "mistralai_Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
            "q4_k_l": "mistralai_Ministral-3-14B-Instruct-2512-Q4_K_L.gguf",
        },
    },
    "granite4_1_8b_heretic": {
        "model_id": "heretic-org/IBM-granite-4.1-8b-heretic",
        "model_dir": "/cache/models/ibm-granite-4.1-8b-heretic",
        "hf_file": "/cache/models/ibm-granite-4.1-8b-heretic/model.safetensors.index.json",
        "baseline_repo": "mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF",
        "tensor_profile": "granite",
        "layers": ",".join(str(i) for i in range(40)),
        "baseline_ggufs": {
            "iq1_m": "IBM-granite-4.1-8b-heretic.i1-IQ1_M.gguf",
            "iq1_s": "IBM-granite-4.1-8b-heretic.i1-IQ1_S.gguf",
            "iq2_m": "IBM-granite-4.1-8b-heretic.i1-IQ2_M.gguf",
            "iq2_s": "IBM-granite-4.1-8b-heretic.i1-IQ2_S.gguf",
            "iq2_xs": "IBM-granite-4.1-8b-heretic.i1-IQ2_XS.gguf",
            "iq2_xxs": "IBM-granite-4.1-8b-heretic.i1-IQ2_XXS.gguf",
            "iq3_m": "IBM-granite-4.1-8b-heretic.i1-IQ3_M.gguf",
            "iq3_s": "IBM-granite-4.1-8b-heretic.i1-IQ3_S.gguf",
            "iq3_xs": "IBM-granite-4.1-8b-heretic.i1-IQ3_XS.gguf",
            "iq3_xxs": "IBM-granite-4.1-8b-heretic.i1-IQ3_XXS.gguf",
            "iq4_nl": "IBM-granite-4.1-8b-heretic.i1-IQ4_NL.gguf",
            "iq4_xs": "IBM-granite-4.1-8b-heretic.i1-IQ4_XS.gguf",
            "q2_k": "IBM-granite-4.1-8b-heretic.i1-Q2_K.gguf",
            "q2_k_s": "IBM-granite-4.1-8b-heretic.i1-Q2_K_S.gguf",
            "q3_k_s": "IBM-granite-4.1-8b-heretic.i1-Q3_K_S.gguf",
            "q3_k_m": "IBM-granite-4.1-8b-heretic.i1-Q3_K_M.gguf",
            "q3_k_l": "IBM-granite-4.1-8b-heretic.i1-Q3_K_L.gguf",
            "q4_0": "IBM-granite-4.1-8b-heretic.i1-Q4_0.gguf",
            "q4_1": "IBM-granite-4.1-8b-heretic.i1-Q4_1.gguf",
            "q4_k_s": "IBM-granite-4.1-8b-heretic.i1-Q4_K_S.gguf",
            "q4_k_m": "IBM-granite-4.1-8b-heretic.i1-Q4_K_M.gguf",
            "q5_k_s": "IBM-granite-4.1-8b-heretic.i1-Q5_K_S.gguf",
            "q5_k_m": "IBM-granite-4.1-8b-heretic.i1-Q5_K_M.gguf",
            "q6_k": "IBM-granite-4.1-8b-heretic.i1-Q6_K.gguf",
        },
    },
    "olmo2_1124_7b_instruct": {
        "model_id": "allenai/OLMo-2-1124-7B-Instruct",
        "model_dir": "/cache/models/olmo2-1124-7b-instruct",
        "hf_file": "/cache/models/olmo2-1124-7b-instruct/model.safetensors.index.json",
        "baseline_repo": "mradermacher/OLMo-2-1124-7B-Instruct-i1-GGUF",
        "tensor_profile": "olmo2",
        "layers": ",".join(str(i) for i in range(32)),
        "baseline_ggufs": {
            "iq1_m": "OLMo-2-1124-7B-Instruct.i1-IQ1_M.gguf",
            "iq1_s": "OLMo-2-1124-7B-Instruct.i1-IQ1_S.gguf",
            "iq2_m": "OLMo-2-1124-7B-Instruct.i1-IQ2_M.gguf",
            "iq2_s": "OLMo-2-1124-7B-Instruct.i1-IQ2_S.gguf",
            "iq2_xs": "OLMo-2-1124-7B-Instruct.i1-IQ2_XS.gguf",
            "iq2_xxs": "OLMo-2-1124-7B-Instruct.i1-IQ2_XXS.gguf",
            "iq3_m": "OLMo-2-1124-7B-Instruct.i1-IQ3_M.gguf",
            "iq3_s": "OLMo-2-1124-7B-Instruct.i1-IQ3_S.gguf",
            "iq3_xs": "OLMo-2-1124-7B-Instruct.i1-IQ3_XS.gguf",
            "iq3_xxs": "OLMo-2-1124-7B-Instruct.i1-IQ3_XXS.gguf",
            "iq4_nl": "OLMo-2-1124-7B-Instruct.i1-IQ4_NL.gguf",
            "iq4_xs": "OLMo-2-1124-7B-Instruct.i1-IQ4_XS.gguf",
            "q2_k": "OLMo-2-1124-7B-Instruct.i1-Q2_K.gguf",
            "q2_k_s": "OLMo-2-1124-7B-Instruct.i1-Q2_K_S.gguf",
            "q3_k_s": "OLMo-2-1124-7B-Instruct.i1-Q3_K_S.gguf",
            "q3_k_m": "OLMo-2-1124-7B-Instruct.i1-Q3_K_M.gguf",
            "q3_k_l": "OLMo-2-1124-7B-Instruct.i1-Q3_K_L.gguf",
            "q4_0": "OLMo-2-1124-7B-Instruct.i1-Q4_0.gguf",
            "q4_1": "OLMo-2-1124-7B-Instruct.i1-Q4_1.gguf",
            "q4_k_s": "OLMo-2-1124-7B-Instruct.i1-Q4_K_S.gguf",
            "q4_k_m": "OLMo-2-1124-7B-Instruct.i1-Q4_K_M.gguf",
            "q5_k_s": "OLMo-2-1124-7B-Instruct.i1-Q5_K_S.gguf",
            "q5_k_m": "OLMo-2-1124-7B-Instruct.i1-Q5_K_M.gguf",
            "q6_k": "OLMo-2-1124-7B-Instruct.i1-Q6_K.gguf",
        },
    },
    "olmo2_0425_1b_instruct": {
        "model_id": "allenai/OLMo-2-0425-1B-Instruct",
        "model_dir": "/cache/models/olmo2-0425-1b-instruct",
        "hf_file": "/cache/models/olmo2-0425-1b-instruct/model.safetensors",
        "baseline_repo": "mradermacher/OLMo-2-0425-1B-Instruct-i1-GGUF",
        "tensor_profile": "olmo2",
        "layers": ",".join(str(i) for i in range(16)),
        "baseline_ggufs": {
            "iq1_m": "OLMo-2-0425-1B-Instruct.i1-IQ1_M.gguf",
            "iq1_s": "OLMo-2-0425-1B-Instruct.i1-IQ1_S.gguf",
            "iq2_m": "OLMo-2-0425-1B-Instruct.i1-IQ2_M.gguf",
            "iq2_s": "OLMo-2-0425-1B-Instruct.i1-IQ2_S.gguf",
            "iq2_xs": "OLMo-2-0425-1B-Instruct.i1-IQ2_XS.gguf",
            "iq2_xxs": "OLMo-2-0425-1B-Instruct.i1-IQ2_XXS.gguf",
            "iq3_m": "OLMo-2-0425-1B-Instruct.i1-IQ3_M.gguf",
            "iq3_s": "OLMo-2-0425-1B-Instruct.i1-IQ3_S.gguf",
            "iq3_xs": "OLMo-2-0425-1B-Instruct.i1-IQ3_XS.gguf",
            "iq3_xxs": "OLMo-2-0425-1B-Instruct.i1-IQ3_XXS.gguf",
            "iq4_nl": "OLMo-2-0425-1B-Instruct.i1-IQ4_NL.gguf",
            "iq4_xs": "OLMo-2-0425-1B-Instruct.i1-IQ4_XS.gguf",
            "q2_k": "OLMo-2-0425-1B-Instruct.i1-Q2_K.gguf",
            "q2_k_s": "OLMo-2-0425-1B-Instruct.i1-Q2_K_S.gguf",
            "q3_k_s": "OLMo-2-0425-1B-Instruct.i1-Q3_K_S.gguf",
            "q3_k_m": "OLMo-2-0425-1B-Instruct.i1-Q3_K_M.gguf",
            "q3_k_l": "OLMo-2-0425-1B-Instruct.i1-Q3_K_L.gguf",
            "q4_0": "OLMo-2-0425-1B-Instruct.i1-Q4_0.gguf",
            "q4_1": "OLMo-2-0425-1B-Instruct.i1-Q4_1.gguf",
            "q4_k_s": "OLMo-2-0425-1B-Instruct.i1-Q4_K_S.gguf",
            "q4_k_m": "OLMo-2-0425-1B-Instruct.i1-Q4_K_M.gguf",
            "q5_k_s": "OLMo-2-0425-1B-Instruct.i1-Q5_K_S.gguf",
            "q5_k_m": "OLMo-2-0425-1B-Instruct.i1-Q5_K_M.gguf",
            "q6_k": "OLMo-2-0425-1B-Instruct.i1-Q6_K.gguf",
        },
    },
    "olmo3_7b_think": {
        "model_id": "allenai/Olmo-3-7B-Think",
        "model_dir": "/cache/models/olmo3-7b-think",
        "hf_file": "/cache/models/olmo3-7b-think/model.safetensors.index.json",
        "baseline_repo": "mradermacher/Olmo-3-7B-Think-i1-GGUF",
        "tensor_profile": "olmo3",
        "layers": ",".join(str(i) for i in range(32)),
        "baseline_ggufs": {
            "iq1_m": "Olmo-3-7B-Think.i1-IQ1_M.gguf",
            "iq1_s": "Olmo-3-7B-Think.i1-IQ1_S.gguf",
            "iq2_m": "Olmo-3-7B-Think.i1-IQ2_M.gguf",
            "iq2_s": "Olmo-3-7B-Think.i1-IQ2_S.gguf",
            "iq2_xs": "Olmo-3-7B-Think.i1-IQ2_XS.gguf",
            "iq2_xxs": "Olmo-3-7B-Think.i1-IQ2_XXS.gguf",
            "iq3_m": "Olmo-3-7B-Think.i1-IQ3_M.gguf",
            "iq3_s": "Olmo-3-7B-Think.i1-IQ3_S.gguf",
            "iq3_xs": "Olmo-3-7B-Think.i1-IQ3_XS.gguf",
            "iq3_xxs": "Olmo-3-7B-Think.i1-IQ3_XXS.gguf",
            "iq4_nl": "Olmo-3-7B-Think.i1-IQ4_NL.gguf",
            "iq4_xs": "Olmo-3-7B-Think.i1-IQ4_XS.gguf",
            "q2_k": "Olmo-3-7B-Think.i1-Q2_K.gguf",
            "q2_k_s": "Olmo-3-7B-Think.i1-Q2_K_S.gguf",
            "q3_k_s": "Olmo-3-7B-Think.i1-Q3_K_S.gguf",
            "q3_k_m": "Olmo-3-7B-Think.i1-Q3_K_M.gguf",
            "q3_k_l": "Olmo-3-7B-Think.i1-Q3_K_L.gguf",
            "q4_0": "Olmo-3-7B-Think.i1-Q4_0.gguf",
            "q4_1": "Olmo-3-7B-Think.i1-Q4_1.gguf",
            "q4_k_s": "Olmo-3-7B-Think.i1-Q4_K_S.gguf",
            "q4_k_m": "Olmo-3-7B-Think.i1-Q4_K_M.gguf",
            "q5_k_s": "Olmo-3-7B-Think.i1-Q5_K_S.gguf",
            "q5_k_m": "Olmo-3-7B-Think.i1-Q5_K_M.gguf",
            "q6_k": "Olmo-3-7B-Think.i1-Q6_K.gguf",
        },
    },
    "gemma4_e2b": {
        "model_id": "google/gemma-4-E2B",
        "model_dir": "/cache/models/gemma4-e2b",
        "hf_file": "/cache/models/gemma4-e2b/model.safetensors",
        "baseline_repo": "mradermacher/gemma-4-E2B-GGUF",
        "tensor_profile": "gemma4",
        "layers": ",".join(str(i) for i in range(35)),
        "baseline_ggufs": {
            "q2_k": "gemma-4-E2B.Q2_K.gguf",
            "q3_k_s": "gemma-4-E2B.Q3_K_S.gguf",
            "q3_k_m": "gemma-4-E2B.Q3_K_M.gguf",
            "q3_k_l": "gemma-4-E2B.Q3_K_L.gguf",
            "iq4_xs": "gemma-4-E2B.IQ4_XS.gguf",
            "q4_k_m": "gemma-4-E2B.Q4_K_M.gguf",
        },
    },
    "gemma4_e2b_it": {
        "model_id": "google/gemma-4-E2B-it",
        "model_dir": "/cache/models/gemma4-e2b-it",
        "hf_file": "/cache/models/gemma4-e2b-it/model.safetensors",
        "baseline_repo": "mradermacher/gemma-4-E2B-it-GGUF",
        "tensor_profile": "gemma4",
        "layers": ",".join(str(i) for i in range(35)),
        "baseline_ggufs": {
            "q2_k": "gemma-4-E2B-it.Q2_K.gguf",
            "q3_k_s": "gemma-4-E2B-it.Q3_K_S.gguf",
            "q3_k_m": "gemma-4-E2B-it.Q3_K_M.gguf",
            "q3_k_l": "gemma-4-E2B-it.Q3_K_L.gguf",
            "iq4_xs": "gemma-4-E2B-it.IQ4_XS.gguf",
            "q4_k_m": "gemma-4-E2B-it.Q4_K_M.gguf",
        },
    },
    "gemma4_e4b": {
        "model_id": "google/gemma-4-E4B",
        "model_dir": "/cache/models/gemma4-e4b",
        "hf_file": "/cache/models/gemma4-e4b/model.safetensors",
        "baseline_repo": "mradermacher/gemma-4-E4B-GGUF",
        "tensor_profile": "gemma4",
        "layers": ",".join(str(i) for i in range(42)),
        "baseline_ggufs": {
            "q2_k": "gemma-4-E4B.Q2_K.gguf",
            "q3_k_s": "gemma-4-E4B.Q3_K_S.gguf",
            "q3_k_m": "gemma-4-E4B.Q3_K_M.gguf",
            "q3_k_l": "gemma-4-E4B.Q3_K_L.gguf",
            "iq4_xs": "gemma-4-E4B.IQ4_XS.gguf",
            "q4_k_m": "gemma-4-E4B.Q4_K_M.gguf",
        },
    },
    "gemma4_e4b_it": {
        "model_id": "google/gemma-4-E4B-it",
        "model_dir": "/cache/models/gemma4-e4b-it",
        "hf_file": "/cache/models/gemma4-e4b-it/model.safetensors",
        "baseline_repo": "mradermacher/gemma-4-E4B-it-GGUF",
        "tensor_profile": "gemma4",
        "layers": ",".join(str(i) for i in range(42)),
        "baseline_ggufs": {
            "q2_k": "gemma-4-E4B-it.Q2_K.gguf",
            "q3_k_s": "gemma-4-E4B-it.Q3_K_S.gguf",
            "q3_k_m": "gemma-4-E4B-it.Q3_K_M.gguf",
            "q3_k_l": "gemma-4-E4B-it.Q3_K_L.gguf",
            "iq4_xs": "gemma-4-E4B-it.IQ4_XS.gguf",
            "q4_k_m": "gemma-4-E4B-it.Q4_K_M.gguf",
        },
    },
}

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "accelerate",
        "datasets",
        "gguf",
        "huggingface_hub",
        "numpy",
        "safetensors",
        "sentencepiece",
        "torch",
        "transformers==5.5.0",
    )
    .add_local_dir("scripts", remote_path="/workspace/scripts")
    .add_local_dir("modal", remote_path="/workspace/candidate")
)


def _run(cmd: list[str], cwd: str = "/workspace") -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/scripts:" + env.get("PYTHONPATH", "")
    print("[modal-sprint] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _ensure_assets(iq4_url: str | None = None) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    model_path = Path(MODEL_DIR)
    if not Path(HF_FILE).exists():
        model_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            MODEL_ID,
            local_dir=MODEL_DIR,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "tokenizer*",
                "*.model",
                "*.tiktoken",
                "merges.txt",
                "vocab.json",
            ],
        )
        cache_volume.commit()

    iq4_path = Path(IQ4_FILE)
    if not iq4_path.exists():
        if iq4_url:
            import urllib.request

            iq4_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(iq4_url, IQ4_FILE)
            cache_volume.commit()
        else:
            raise FileNotFoundError(
                "IQ4 GGUF is missing from the Modal volume. Upload it once with:\n"
                f"modal volume put {VOLUME_NAME} "
                r"<local-path-to-Qwen_Qwen3-1.7B-IQ4_XS.gguf> "
                "/models/Qwen_Qwen3-1.7B-IQ4_XS.gguf"
            )
    return HF_FILE, IQ4_FILE


def _ensure_baseline_gguf(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    baseline_dir = Path("/cache/models/baselines") / repo_id.replace("/", "__")
    baseline_path = baseline_dir / filename
    if not baseline_path.exists():
        baseline_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(baseline_dir),
        )
        cache_volume.commit()
    return str(baseline_path)


def _ensure_model_snapshot(config: dict) -> str:
    from huggingface_hub import snapshot_download

    def select_hf_path() -> str | None:
        preferred = Path(config["hf_file"])
        if preferred.exists():
            return str(preferred)
        model_dir = Path(config["model_dir"])
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            return str(index_path)
        shards = sorted(model_dir.glob("*.safetensors"))
        if len(shards) == 1:
            return str(shards[0])
        return None

    selected = select_hf_path()
    if selected:
        return selected

    hf_file = Path(config["hf_file"])
    Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        config["model_id"],
        local_dir=config["model_dir"],
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer*",
            "*.model",
            "*.tiktoken",
            "merges.txt",
            "vocab.json",
        ],
    )
    cache_volume.commit()
    selected = select_hf_path()
    if selected:
        return selected
    raise FileNotFoundError(f"no safetensors weights or shard index found under {hf_file.parent}")


def _ensure_configured_sources(config: dict, labels: set[str]) -> dict[str, str]:
    repo_id = config["baseline_repo"]
    specs = config["baseline_ggufs"]
    missing = sorted(label for label in labels if label not in specs)
    if missing:
        raise ValueError(f"model config is missing GGUF source specs for {missing}")
    return {label: _ensure_baseline_gguf(repo_id, specs[label]) for label in sorted(labels)}


def _ensure_job_gguf(job: dict, default_gguf_file: str) -> str:
    if job.get("gguf_file"):
        return job["gguf_file"]
    if job.get("gguf_filename"):
        return _ensure_baseline_gguf(
            job.get("gguf_repo_id", BASELINE_REPO),
            job["gguf_filename"],
        )
    return default_gguf_file


def _read_result(output_dir: str) -> dict:
    path = Path(output_dir) / "result.json"
    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    return {
        "output_dir": output_dir,
        "status": result.get("status") or result.get("verdict"),
        "decision_text": result.get("decision_text"),
        "overall": result.get("overall"),
        "variants": result.get("variants"),
    }


def _candidate_suffix(candidate_variant: str) -> str:
    if candidate_variant == "c2_calib_greedy_mixed":
        return ""
    label = candidate_variant.replace("c2_", "").replace("_mixed", "")
    return f"_candidate_{label}"


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 3,
)
def run_model_forward_job(job: dict) -> dict:
    hf_file, iq4_file = _ensure_assets(job.get("iq4_url"))
    output_dir = f"{RESULT_ROOT}/phase_a_model_forward/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/mlp_codebook_model_forward_gate.py",
        "--model-dir",
        MODEL_DIR,
        "--hf",
        hf_file,
        "--iq4",
        iq4_file,
        "--output-dir",
        output_dir,
        "--layers",
        job.get(
            "layers",
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        ),
        "--variants",
        job.get("variants", "fp16,static_int3_mlp,joint_codebook_mlp,iq4_mlp"),
        "--codebook-targets",
        job.get("codebook_targets", "gate,up,down"),
        "--calib-prompts",
        str(job.get("calib_prompts", 96)),
        "--heldout-prompts",
        str(job.get("heldout_prompts", 96)),
        "--eval-prompts",
        str(job.get("eval_prompts", 64)),
        "--max-activation-tokens",
        str(job.get("max_activation_tokens", 192)),
        "--max-tokens",
        str(job.get("max_tokens", 128)),
        "--eval-max-length",
        str(job.get("eval_max_length", 128)),
        "--block-size",
        str(job.get("block_size", 32)),
        "--codebook-size",
        str(job.get("codebook_size", 128)),
        "--train-blocks",
        str(job.get("train_blocks", 50000)),
        "--kmeans-iters",
        str(job.get("kmeans_iters", 12)),
        "--seed",
        str(job.get("seed", 6)),
        "--device",
        "cuda",
    ]
    if job.get("fit_payload_controls", False):
        cmd.append("--fit-payload-controls")
    _run(cmd)
    cache_volume.commit()
    return _read_result(output_dir)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 3,
)
def run_artifact_job(job: dict) -> dict:
    hf_file, _iq4_file = _ensure_assets(job.get("iq4_url"))
    output_dir = f"{RESULT_ROOT}/phase_e_artifact/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/mlp_codebook_artifact_gate.py",
        "--model-dir",
        MODEL_DIR,
        "--hf",
        hf_file,
        "--output-dir",
        output_dir,
        "--layers",
        job.get(
            "layers",
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        ),
        "--codebook-targets",
        job.get("codebook_targets", "gate,up,down"),
        "--calib-prompts",
        str(job.get("calib_prompts", 96)),
        "--heldout-prompts",
        str(job.get("heldout_prompts", 96)),
        "--max-activation-tokens",
        str(job.get("max_activation_tokens", 192)),
        "--max-tokens",
        str(job.get("max_tokens", 128)),
        "--block-size",
        str(job.get("block_size", 32)),
        "--codebook-size",
        str(job.get("codebook_size", 128)),
        "--train-blocks",
        str(job.get("train_blocks", 50000)),
        "--kmeans-iters",
        str(job.get("kmeans_iters", 12)),
        "--seed",
        str(job.get("seed", 6)),
        "--device",
        "cuda",
    ]
    _run(cmd)
    cache_volume.commit()
    return _read_result(output_dir)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 3,
)
def run_mixed_rate_job(job: dict) -> dict:
    hf_file, iq4_file = _ensure_assets(job.get("iq4_url"))
    iq4_file = _ensure_job_gguf(job, iq4_file)
    output_dir = f"{RESULT_ROOT}/phase_b_mixed_rate/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/mlp_codebook_model_forward_gate.py",
        "--model-dir",
        MODEL_DIR,
        "--hf",
        hf_file,
        "--iq4",
        iq4_file,
        "--output-dir",
        output_dir,
        "--layers",
        job.get(
            "layers",
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        ),
        "--variants",
        job.get(
            "variants",
            "fp16,iq4_all,static_int3_mlp_iq4_rest,joint_codebook_mlp_iq4_rest",
        ),
        "--codebook-targets",
        job.get("codebook_targets", "gate,up,down"),
        "--calib-prompts",
        str(job.get("calib_prompts", 96)),
        "--heldout-prompts",
        str(job.get("heldout_prompts", 96)),
        "--eval-prompts",
        str(job.get("eval_prompts", 64)),
        "--max-activation-tokens",
        str(job.get("max_activation_tokens", 192)),
        "--max-tokens",
        str(job.get("max_tokens", 128)),
        "--eval-max-length",
        str(job.get("eval_max_length", 128)),
        "--block-size",
        str(job.get("block_size", 32)),
        "--codebook-size",
        str(job.get("codebook_size", 128)),
        "--train-blocks",
        str(job.get("train_blocks", 50000)),
        "--kmeans-iters",
        str(job.get("kmeans_iters", 12)),
        "--seed",
        str(job.get("seed", 6)),
        "--device",
        "cuda",
    ]
    _run(cmd)
    cache_volume.commit()
    return _read_result(output_dir)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 3,
)
def run_baseline_bakeoff_job(job: dict) -> dict:
    hf_file, default_iq4_file = _ensure_assets(job.get("iq4_url"))
    gguf_file = _ensure_job_gguf(job, default_iq4_file)
    output_dir = f"{RESULT_ROOT}/phase_c_baseline_bakeoff/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/mlp_codebook_model_forward_gate.py",
        "--model-dir",
        MODEL_DIR,
        "--hf",
        hf_file,
        "--iq4",
        gguf_file,
        "--output-dir",
        output_dir,
        "--layers",
        job.get(
            "layers",
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        ),
        "--variants",
        job.get(
            "variants",
            "fp16,iq4_all,static_int3_mlp_iq4_rest,joint_codebook_mlp_iq4_rest",
        ),
        "--codebook-targets",
        job.get("codebook_targets", "gate,up,down"),
        "--calib-prompts",
        str(job.get("calib_prompts", 96)),
        "--heldout-prompts",
        str(job.get("heldout_prompts", 96)),
        "--eval-prompts",
        str(job.get("eval_prompts", 64)),
        "--max-activation-tokens",
        str(job.get("max_activation_tokens", 192)),
        "--max-tokens",
        str(job.get("max_tokens", 128)),
        "--eval-max-length",
        str(job.get("eval_max_length", 128)),
        "--block-size",
        str(job.get("block_size", 32)),
        "--codebook-size",
        str(job.get("codebook_size", 128)),
        "--train-blocks",
        str(job.get("train_blocks", 50000)),
        "--kmeans-iters",
        str(job.get("kmeans_iters", 12)),
        "--seed",
        str(job.get("seed", 6)),
        "--device",
        "cuda",
    ]
    if job.get("fit_payload_controls", False):
        cmd.append("--fit-payload-controls")
    _run(cmd)
    cache_volume.commit()
    result = _read_result(output_dir)
    result["baseline_label"] = job.get("baseline_label")
    result["baseline_repo_id"] = job.get("gguf_repo_id")
    result["baseline_filename"] = job.get("gguf_filename")
    return result


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 2,
)
def run_composition_job(job: dict) -> dict:
    hf_file, iq4_file = _ensure_assets(job.get("iq4_url"))
    output_dir = f"{RESULT_ROOT}/phase_d_composition_sweep/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/mlp_joint_codebook_composition_gate.py",
        "--model-dir",
        MODEL_DIR,
        "--hf",
        hf_file,
        "--iq4",
        iq4_file,
        "--output-dir",
        output_dir,
        "--layers",
        job.get(
            "layers",
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        ),
        "--codebook-targets",
        job.get("codebook_targets", "gate,up,down"),
        "--calib-prompts",
        str(job.get("calib_prompts", 96)),
        "--heldout-prompts",
        str(job.get("heldout_prompts", 96)),
        "--max-activation-tokens",
        str(job.get("max_activation_tokens", 192)),
        "--max-tokens",
        str(job.get("max_tokens", 128)),
        "--block-size",
        str(job.get("block_size", 32)),
        "--codebook-size",
        str(job.get("codebook_size", 128)),
        "--train-blocks",
        str(job.get("train_blocks", 50000)),
        "--kmeans-iters",
        str(job.get("kmeans_iters", 12)),
        "--seed",
        str(job.get("seed", 6)),
        "--device",
        "cuda",
    ]
    _run(cmd)
    cache_volume.commit()
    return _read_result(output_dir)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 4,
)
def run_production_mix_job(job: dict) -> dict:
    hf_file, iq4_file = _ensure_assets(job.get("iq4_url"))
    high_sources = job.get("high_sources", "q3_k_m,iq4_xs")
    high_labels = {part.strip() for part in high_sources.split(",") if part.strip()}
    demotion_sources = job.get("demotion_sources", "")
    demotion_labels = {part.strip() for part in demotion_sources.split(",") if part.strip()}
    required_labels = {
        job.get("low_source", "iq3_xs"),
        job.get("target_source", "q3_k_m"),
        job.get("demotion_base_source") or job.get("target_source", "q3_k_m"),
        *high_labels,
        *demotion_labels,
    }
    source_paths = {}
    for label in sorted(required_labels):
        if label == "iq4_xs":
            source_paths[label] = iq4_file
            continue
        if label not in BASELINE_GGUFS:
            raise ValueError(f"unknown source label {label}; choose from {sorted(BASELINE_GGUFS) + ['iq4_xs']}")
        spec = BASELINE_GGUFS[label]
        source_paths[label] = _ensure_baseline_gguf(spec["repo_id"], spec["filename"])
    output_dir = f"{RESULT_ROOT}/{job.get('result_bucket', 'run_007_c2_production_mixed_rate')}/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/production_mixed_rate_transcoder_gate.py",
        "--model-dir",
        MODEL_DIR,
        "--hf",
        hf_file,
        "--output-dir",
        output_dir,
        "--low-source",
        job.get("low_source", "iq3_xs"),
        "--target-source",
        job.get("target_source", "q3_k_m"),
        "--high-sources",
        high_sources,
        "--layers",
        job.get(
            "layers",
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        ),
        "--group-mode",
        job.get("group_mode", "tensor"),
        "--tensor-profile",
        job.get("tensor_profile", "qwen"),
        "--calib-prompts",
        str(job.get("calib_prompts", 12)),
        "--eval-prompts",
        str(job.get("eval_prompts", 64)),
        "--calib-max-length",
        str(job.get("calib_max_length", 96)),
        "--eval-max-length",
        str(job.get("eval_max_length", 128)),
        "--prompt-source",
        job.get("prompt_source", "synthetic"),
        "--dataset",
        job.get("dataset", "wikitext"),
        "--dataset-config",
        job.get("dataset_config", "wikitext-2-raw-v1"),
        "--text-column",
        job.get("text_column", "text"),
        "--calib-split",
        job.get("calib_split", "train"),
        "--eval-split",
        job.get("eval_split", "validation"),
        "--prompt-seed",
        str(job.get("prompt_seed", job.get("seed", 6) + 2000)),
        "--min-tokens",
        str(job.get("min_tokens", 64)),
        "--seed",
        str(job.get("seed", 6)),
        "--device",
        "cuda",
        "--candidate-variant",
        job.get("candidate_variant", "c2_calib_greedy_mixed"),
        "--knapsack-max-states",
        str(job.get("knapsack_max_states", 50000)),
    ]
    if job.get("sweep_payload_bpws", ""):
        cmd.extend(["--sweep-payload-bpws", job["sweep_payload_bpws"]])
    if job.get("sweep_selectors", ""):
        cmd.extend(["--sweep-selectors", job["sweep_selectors"]])
    if job.get("local_search_from", ""):
        cmd.extend(["--local-search-from", job["local_search_from"]])
    if int(job.get("local_search_steps", 0) or 0) > 0:
        cmd.extend(
            [
                "--local-search-steps",
                str(job["local_search_steps"]),
                "--local-search-candidates",
                str(job.get("local_search_candidates", 24)),
                "--local-search-min-improvement",
                str(job.get("local_search_min_improvement", 0.0001)),
            ]
        )
    if job.get("genetic_search_from", ""):
        cmd.extend(["--genetic-search-from", job["genetic_search_from"]])
    if int(job.get("genetic_search_generations", 0) or 0) > 0:
        cmd.extend(
            [
                "--genetic-search-generations",
                str(job["genetic_search_generations"]),
                "--genetic-search-population",
                str(job.get("genetic_search_population", 8)),
                "--genetic-search-elite",
                str(job.get("genetic_search_elite", 2)),
                "--genetic-search-mutation-rate",
                str(job.get("genetic_search_mutation_rate", 0.25)),
            ]
        )
    if bool(job.get("genetic_search_direct", False)):
        cmd.append("--genetic-search-direct")
    if demotion_sources:
        cmd.extend(["--demotion-sources", demotion_sources])
    if job.get("demotion_base_source"):
        cmd.extend(["--demotion-base-source", job["demotion_base_source"]])
    if job.get("demotion_selectors", ""):
        cmd.extend(["--demotion-selectors", job["demotion_selectors"]])
    if job.get("max_shrink_nll_loss") is not None:
        cmd.extend(["--max-shrink-nll-loss", str(job["max_shrink_nll_loss"])])
    for label, path in source_paths.items():
        cmd.extend(["--source", f"{label}={path}"])
    _run(cmd)
    cache_volume.commit()
    result = _read_result(output_dir)
    result["low_source"] = job.get("low_source", "iq3_xs")
    result["target_source"] = job.get("target_source", "q3_k_m")
    result["high_sources"] = high_sources
    return result


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 4,
)
def run_production_mix_configured_job(job: dict) -> dict:
    model_key = job.get("model_key", "qwen3_0p6b_base")
    config = MODEL_CONFIGS[model_key]
    hf_file = _ensure_model_snapshot(config)
    high_sources = job.get("high_sources", "q3_k_m,iq4_xs")
    high_labels = {part.strip() for part in high_sources.split(",") if part.strip()}
    demotion_sources = job.get("demotion_sources", "")
    demotion_labels = {part.strip() for part in demotion_sources.split(",") if part.strip()}
    required_labels = {
        job.get("low_source", "iq3_xs"),
        job.get("target_source", "iq3_m"),
        job.get("demotion_base_source") or job.get("target_source", "iq3_m"),
        *high_labels,
        *demotion_labels,
    }
    source_paths = _ensure_configured_sources(config, required_labels)
    output_dir = f"{RESULT_ROOT}/{job.get('result_bucket', 'run_008_c2_replication')}/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/production_mixed_rate_transcoder_gate.py",
        "--model-dir",
        config["model_dir"],
        "--hf",
        hf_file,
        "--output-dir",
        output_dir,
        "--low-source",
        job.get("low_source", "iq3_xs"),
        "--target-source",
        job.get("target_source", "iq3_m"),
        "--high-sources",
        high_sources,
        "--layers",
        job.get("layers", config["layers"]),
        "--group-mode",
        job.get("group_mode", "tensor"),
        "--tensor-profile",
        job.get("tensor_profile", config.get("tensor_profile", "qwen")),
        "--calib-prompts",
        str(job.get("calib_prompts", 12)),
        "--eval-prompts",
        str(job.get("eval_prompts", 256)),
        "--calib-max-length",
        str(job.get("calib_max_length", 96)),
        "--eval-max-length",
        str(job.get("eval_max_length", 128)),
        "--prompt-source",
        job.get("prompt_source", "synthetic"),
        "--dataset",
        job.get("dataset", "wikitext"),
        "--dataset-config",
        job.get("dataset_config", "wikitext-2-raw-v1"),
        "--text-column",
        job.get("text_column", "text"),
        "--calib-split",
        job.get("calib_split", "train"),
        "--eval-split",
        job.get("eval_split", "validation"),
        "--prompt-seed",
        str(job.get("prompt_seed", job.get("seed", 6) + 2000)),
        "--min-tokens",
        str(job.get("min_tokens", 64)),
        "--seed",
        str(job.get("seed", 6)),
        "--device",
        "cuda",
        "--candidate-variant",
        job.get("candidate_variant", "c2_calib_greedy_mixed"),
        "--knapsack-max-states",
        str(job.get("knapsack_max_states", 50000)),
    ]
    if job.get("sweep_payload_bpws", ""):
        cmd.extend(["--sweep-payload-bpws", job["sweep_payload_bpws"]])
    if job.get("sweep_selectors", ""):
        cmd.extend(["--sweep-selectors", job["sweep_selectors"]])
    if job.get("local_search_from", ""):
        cmd.extend(["--local-search-from", job["local_search_from"]])
    if int(job.get("local_search_steps", 0) or 0) > 0:
        cmd.extend(
            [
                "--local-search-steps",
                str(job["local_search_steps"]),
                "--local-search-candidates",
                str(job.get("local_search_candidates", 24)),
                "--local-search-min-improvement",
                str(job.get("local_search_min_improvement", 0.0001)),
            ]
        )
    if job.get("genetic_search_from", ""):
        cmd.extend(["--genetic-search-from", job["genetic_search_from"]])
    if int(job.get("genetic_search_generations", 0) or 0) > 0:
        cmd.extend(
            [
                "--genetic-search-generations",
                str(job["genetic_search_generations"]),
                "--genetic-search-population",
                str(job.get("genetic_search_population", 8)),
                "--genetic-search-elite",
                str(job.get("genetic_search_elite", 2)),
                "--genetic-search-mutation-rate",
                str(job.get("genetic_search_mutation_rate", 0.25)),
            ]
        )
    if bool(job.get("genetic_search_direct", False)):
        cmd.append("--genetic-search-direct")
    if demotion_sources:
        cmd.extend(["--demotion-sources", demotion_sources])
    if job.get("demotion_base_source"):
        cmd.extend(["--demotion-base-source", job["demotion_base_source"]])
    if job.get("demotion_selectors", ""):
        cmd.extend(["--demotion-selectors", job["demotion_selectors"]])
    if job.get("max_shrink_nll_loss") is not None:
        cmd.extend(["--max-shrink-nll-loss", str(job["max_shrink_nll_loss"])])
    for label, path in source_paths.items():
        cmd.extend(["--source", f"{label}={path}"])
    _run(cmd)
    cache_volume.commit()
    result = _read_result(output_dir)
    result["model_key"] = model_key
    result["low_source"] = job.get("low_source", "iq3_xs")
    result["target_source"] = job.get("target_source", "iq3_m")
    result["high_sources"] = high_sources
    return result


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    timeout=60 * 60,
)
def run_c2_artifact_job(job: dict) -> dict:
    result_json = job["result_json"]
    variant = job.get("variant", "c2_calib_greedy_mixed")
    selection_required_labels: set[str] = set()
    result_path = Path(result_json)
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            selector_result = json.load(f)
        result_args = selector_result.get("args", {})
        base_source = selector_result.get("selection_base_sources", {}).get(
            variant,
            result_args.get("low_source"),
        )
        for label in {base_source, result_args.get("low_source"), result_args.get("target_source")}:
            if label:
                selection_required_labels.add(label)
        for row in selector_result.get("selections", {}).get(variant, []):
            selection_required_labels.add(row["source"])
    high_sources = job.get("high_sources", "q3_k_m,iq4_xs")
    high_labels = {part.strip() for part in high_sources.split(",") if part.strip()}
    required_labels = {
        job.get("metadata_source", "iq3_xs"),
        job.get("target_source", "q3_k_m"),
        *high_labels,
        *selection_required_labels,
    }
    if "model_key" in job:
        config = MODEL_CONFIGS[job["model_key"]]
        source_paths = _ensure_configured_sources(config, required_labels)
        tensor_profile = job.get("tensor_profile", config.get("tensor_profile", "qwen"))
    else:
        iq4_file = None
        source_paths = {}
        for label in sorted(required_labels):
            if label == "iq4_xs" and label not in BASELINE_GGUFS:
                if iq4_file is None:
                    _hf_file, iq4_file = _ensure_assets(job.get("iq4_url"))
                source_paths[label] = iq4_file
                continue
            if label not in BASELINE_GGUFS:
                raise ValueError(f"unknown source label {label}; choose from {sorted(BASELINE_GGUFS) + ['iq4_xs']}")
            spec = BASELINE_GGUFS[label]
            source_paths[label] = _ensure_baseline_gguf(spec["repo_id"], spec["filename"])
        tensor_profile = job.get("tensor_profile", "qwen")
    output_dir = f"{RESULT_ROOT}/{job.get('artifact_bucket', 'run_007_c2_mixed_gguf_artifact')}/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/build_mixed_gguf_artifact.py",
        "--result-json",
        result_json,
        "--output-dir",
        output_dir,
        "--output-gguf",
        job.get("output_gguf", "mixed.gguf"),
        "--variant",
        variant,
        "--metadata-source",
        job.get("metadata_source", "iq3_xs"),
        "--tensor-profile",
        tensor_profile,
    ]
    for label, path in source_paths.items():
        cmd.extend(["--source", f"{label}={path}"])
    _run(cmd)
    cache_volume.commit()
    report_path = Path(output_dir) / "artifact_report.json"
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    return {
        "output_dir": output_dir,
        "status": report.get("status"),
        "decision_text": report.get("decision_text"),
        "output_gguf": report.get("output_gguf"),
        "file_size_bytes": report.get("file_size_bytes"),
        "payload_bytes": report.get("payload_bytes"),
        "file_bpw": report.get("file_bpw"),
        "payload_bpw": report.get("payload_bpw"),
        "load_check": report.get("load_check"),
        "c2_metrics": report.get("c2_metrics"),
    }


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 3,
)
def run_pmra_public_eval_job(job: dict) -> dict:
    model_key = job.get("model_key", "qwen3_1p7b")
    config = MODEL_CONFIGS[model_key]
    hf_file = _ensure_model_snapshot(config)
    high_sources = job.get("high_sources", "q2_k_l,q3_k_s,q3_k_m,iq4_xs")
    high_labels = {part.strip() for part in high_sources.split(",") if part.strip()}
    variant_labels = {
        part.strip()
        for part in job.get(
            "variants",
            "fp16,iq2_m,iq3_xs,q3_k_s,c2_calib_greedy_mixed,c2_random_same_budget",
        ).split(",")
        if part.strip() and not part.strip().startswith("c2_") and part.strip() != "fp16"
    }
    default_result_core = (
        f"c2_mix_low_{job.get('low_source', 'iq2_m')}_target_{job.get('target_source', 'iq3_xs')}_high_"
        f"{high_sources.replace(',', '_')}_seed_{job.get('seed', 7)}_eval_{job.get('eval_prompts', 1024)}_"
        f"calib_{job.get('calib_prompts', 48)}_{job.get('group_mode', 'tensor')}"
    )
    default_result_name = (
        default_result_core if model_key == "qwen3_1p7b" else f"{model_key}_{default_result_core}"
    )
    result_name = job.get("result_name") or default_result_name
    result_json = job.get(
        "result_json",
        f"{RESULT_ROOT}/{job.get('result_bucket', 'run_008_c2_subq3_iq2m_to_iq3xs_calib48_eval1024')}/{result_name}/result.json",
    )
    selection_required_labels: set[str] = set()
    result_path = Path(result_json)
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            selector_result = json.load(f)
        requested_variants = {
            part.strip()
            for part in job.get(
                "variants",
                "fp16,iq2_m,iq3_xs,q3_k_s,c2_calib_greedy_mixed,c2_random_same_budget",
            ).split(",")
            if part.strip()
        }
        for variant in requested_variants:
            base_source = selector_result.get("selection_base_sources", {}).get(variant)
            if base_source:
                selection_required_labels.add(base_source)
            for row in selector_result.get("selections", {}).get(variant, []):
                selection_required_labels.add(row["source"])
    required_labels = {
        job.get("low_source", "iq2_m"),
        job.get("target_source", "iq3_xs"),
        *high_labels,
        *variant_labels,
        *selection_required_labels,
    }
    source_paths = _ensure_configured_sources(config, required_labels)
    output_dir = f"{RESULT_ROOT}/{job.get('public_bucket', 'run_008_pmra_public_eval')}/{job['name']}"
    dataset_config = job.get("dataset_config", "wikitext-2-raw-v1")
    cmd = [
        sys.executable,
        "/workspace/scripts/evaluate_pmra_public_dataset.py",
        "--model-dir",
        config["model_dir"],
        "--hf",
        hf_file,
        "--result-json",
        result_json,
        "--output-dir",
        output_dir,
        "--variants",
        job.get("variants", "fp16,iq2_m,iq3_xs,q3_k_s,c2_calib_greedy_mixed,c2_random_same_budget"),
        "--candidate-variant",
        job.get("candidate_variant", "c2_calib_greedy_mixed"),
        "--random-variant",
        job.get("random_variant", "c2_random_same_budget"),
        "--dataset",
        job.get("dataset", "wikitext"),
        "--split",
        job.get("split", "test"),
        "--text-column",
        job.get("text_column", "text"),
        "--prompt-count",
        str(job.get("prompt_count", 512)),
        "--prompt-seed",
        str(job.get("prompt_seed", 1701)),
        "--eval-max-length",
        str(job.get("eval_max_length", 256)),
        "--min-tokens",
        str(job.get("min_tokens", 64)),
        "--device",
        "cuda",
    ]
    if dataset_config:
        cmd.extend(["--dataset-config", dataset_config])
    for label, path in source_paths.items():
        cmd.extend(["--source", f"{label}={path}"])
    _run(cmd)
    cache_volume.commit()
    result = _read_result(output_dir)
    result["model_key"] = model_key
    result["result_json"] = result_json
    return result


def _run_pmra_code_likelihood_inline(job: dict) -> dict:
    model_key = job.get("model_key", "gemma4_e2b_it")
    config = MODEL_CONFIGS[model_key]
    hf_file = _ensure_model_snapshot(config)
    high_sources = job.get("high_sources", "q3_k_m,q3_k_l,iq4_xs,q4_k_m")
    high_labels = {part.strip() for part in high_sources.split(",") if part.strip()}
    variant_labels = {
        part.strip()
        for part in job.get(
            "variants",
            "fp16,q2_k,q3_k_s,q4_k_m,c2_calib_greedy_mixed,c2_random_same_budget",
        ).split(",")
        if part.strip() and not part.strip().startswith("c2_") and part.strip() != "fp16"
    }
    required_labels = {
        job.get("low_source", "q2_k"),
        job.get("target_source", "q3_k_s"),
        *high_labels,
        *variant_labels,
    }
    source_paths = _ensure_configured_sources(config, required_labels)
    default_result_core = (
        f"{model_key}_c2_publiccal_{job.get('dataset', 'wikitext').replace('/', '_')}_"
        f"{job.get('dataset_config', 'wikitext-2-raw-v1').replace('/', '_')}_"
        f"{job.get('calib_split', 'train')}_to_{job.get('eval_split', 'validation')}_"
        f"low_{job.get('low_source', 'q2_k')}_target_{job.get('target_source', 'q3_k_s')}_"
        f"high_{high_sources.replace(',', '_')}_seed_{job.get('seed', 7)}_eval_{job.get('eval_prompts', 256)}_"
        f"calib_{job.get('calib_prompts', 24)}_{job.get('group_mode', 'tensor')}_len_{job.get('eval_max_length', 256)}"
    )
    result_name = job.get("result_name") or default_result_core
    result_json = job.get(
        "result_json",
        f"{RESULT_ROOT}/{job.get('result_bucket', 'run_009_gemma4_public_calibrated_wikitext_train_val')}/{result_name}/result.json",
    )
    output_dir = f"{RESULT_ROOT}/{job.get('coding_bucket', 'run_009_gemma4_code_likelihood')}/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/evaluate_pmra_code_likelihood.py",
        "--model-dir",
        config["model_dir"],
        "--hf",
        hf_file,
        "--result-json",
        result_json,
        "--output-dir",
        output_dir,
        "--variants",
        job.get("variants", "fp16,q2_k,q3_k_s,q4_k_m,c2_calib_greedy_mixed,c2_random_same_budget"),
        "--candidate-variant",
        job.get("candidate_variant", "c2_calib_greedy_mixed"),
        "--random-variant",
        job.get("random_variant", "c2_random_same_budget"),
        "--benchmark",
        job.get("benchmark", "mbpp_sanitized"),
        "--split",
        job.get("split", "test"),
        "--tasks",
        str(job.get("tasks", 0)),
        "--task-seed",
        str(job.get("task_seed", 0)),
        "--max-length",
        str(job.get("max_length", 2048)),
        "--tolerance",
        str(job.get("tolerance", 0.02)),
        "--device",
        "cuda",
    ]
    for label, path in source_paths.items():
        cmd.extend(["--source", f"{label}={path}"])
    _run(cmd)
    cache_volume.commit()
    result = _read_result(output_dir)
    result["model_key"] = model_key
    result["result_json"] = result_json
    return result


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 3,
)
def run_pmra_code_likelihood_job(job: dict) -> dict:
    return _run_pmra_code_likelihood_inline(job)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 3,
)
def run_pmra_code_likelihood_configured_direct(
    model_key: str = "gemma4_e2b_it",
    seed: int = 7,
    eval_prompts: int = 256,
    calib_prompts: int = 24,
    group_mode: str = "tensor",
    low_source: str = "q2_k",
    target_source: str = "q3_k_s",
    high_sources: str = "q3_k_m,q3_k_l,iq4_xs,q4_k_m",
    variants: str = "fp16,q2_k,q3_k_s,q4_k_m,c2_calib_greedy_mixed,c2_random_same_budget",
    candidate_variant: str = "c2_calib_greedy_mixed",
    random_variant: str = "c2_random_same_budget",
    benchmark: str = "mbpp_sanitized",
    split: str = "test",
    tasks: int = 0,
    task_seed: int = 0,
    max_length: int = 2048,
    tolerance: float = 0.02,
    result_bucket: str = "run_009_gemma4_public_calibrated_wikitext_train_val",
    result_name: str = "",
    coding_bucket: str = "run_009_gemma4_code_likelihood",
) -> dict:
    job = {
        "name": f"{model_key}_{benchmark}_{split}_likelihood_tasks_{tasks}_seed_{task_seed}",
        "model_key": model_key,
        "seed": seed,
        "eval_prompts": eval_prompts,
        "calib_prompts": calib_prompts,
        "group_mode": group_mode,
        "low_source": low_source,
        "target_source": target_source,
        "high_sources": high_sources,
        "variants": variants,
        "candidate_variant": candidate_variant,
        "random_variant": random_variant,
        "benchmark": benchmark,
        "split": split,
        "tasks": tasks,
        "task_seed": task_seed,
        "max_length": max_length,
        "tolerance": tolerance,
        "result_bucket": result_bucket,
        "result_name": result_name or None,
        "coding_bucket": coding_bucket,
    }
    return _run_pmra_code_likelihood_inline(job)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    timeout=60 * 60 * 4,
)
def run_iq4_erasure_job(job: dict) -> dict:
    hf_file, iq4_file = _ensure_assets(job.get("iq4_url"))
    source_specs = {
        "iq3_xs": BASELINE_GGUFS["iq3_xs"],
        "iq3_m": BASELINE_GGUFS["iq3_m"],
        "q3_k_m": BASELINE_GGUFS["q3_k_m"],
    }
    source_paths = {
        label: _ensure_baseline_gguf(spec["repo_id"], spec["filename"])
        for label, spec in source_specs.items()
    }
    source_paths["iq4_xs"] = iq4_file
    output_dir = f"{RESULT_ROOT}/run_007_c5_iq4_semantic_erasure/{job['name']}"
    cmd = [
        sys.executable,
        "/workspace/scripts/iq4_semantic_erasure_gate.py",
        "--model-dir",
        MODEL_DIR,
        "--hf",
        hf_file,
        "--output-dir",
        output_dir,
        "--iq4-source",
        job.get("iq4_source", "iq4_xs"),
        "--target-source",
        job.get("target_source", "q3_k_m"),
        "--layers",
        job.get(
            "layers",
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        ),
        "--group-mode",
        job.get("group_mode", "layer_family"),
        "--modes",
        job.get("modes", "q_bit0_mid,q_bit1_mid,scale_hi0"),
        "--target-saving-bpw",
        str(job.get("target_saving_bpw", 0.75)),
        "--calib-prompts",
        str(job.get("calib_prompts", 12)),
        "--eval-prompts",
        str(job.get("eval_prompts", 64)),
        "--calib-max-length",
        str(job.get("calib_max_length", 96)),
        "--eval-max-length",
        str(job.get("eval_max_length", 128)),
        "--seed",
        str(job.get("seed", 6)),
        "--device",
        "cuda",
    ]
    if job.get("include_global_groups", False):
        cmd.append("--include-global-groups")
    for label, path in source_paths.items():
        cmd.extend(["--source", f"{label}={path}"])
    _run(cmd)
    cache_volume.commit()
    result = _read_result(output_dir)
    result["iq4_source"] = job.get("iq4_source", "iq4_xs")
    result["target_source"] = job.get("target_source", "q3_k_m")
    result["target_saving_bpw"] = job.get("target_saving_bpw", 0.75)
    return result


@app.function(image=image, volumes={"/cache": cache_volume}, timeout=60 * 30)
def prepare_cache(iq4_url: str | None = None) -> dict:
    hf_file, iq4_file = _ensure_assets(iq4_url)
    return {"hf_file": hf_file, "iq4_file": iq4_file}


@app.function(image=image, volumes={"/cache": cache_volume}, timeout=60 * 15)
def stat_known_ggufs(extra_paths: list[str] | None = None) -> dict:
    _ensure_assets(None)
    paths = {"iq4_xs": IQ4_FILE}
    for label, spec in BASELINE_GGUFS.items():
        paths[label] = _ensure_baseline_gguf(spec["repo_id"], spec["filename"])
    for raw_path in extra_paths or []:
        label = Path(raw_path).stem
        paths[label] = raw_path
    return {
        label: {"path": path, "bytes": Path(path).stat().st_size}
        for label, path in sorted(paths.items())
        if Path(path).exists()
    }


@app.local_entrypoint()
def prep(iq4_url: str | None = None):
    """Download/cache HF weights and verify the IQ4 GGUF is present on the Modal volume."""
    print(prepare_cache.remote(iq4_url))


@app.local_entrypoint()
def phase_c2_sizes(extra_paths: str = ""):
    """Print exact byte sizes for production GGUF sources and optional C2 artifacts."""
    extras = [part.strip() for part in extra_paths.split(",") if part.strip()]
    print(json.dumps(stat_known_ggufs.remote(extras), indent=2))


@app.local_entrypoint()
def phase_a(seeds: str = "6,7,8", eval_prompts: int = 64):
    """Run parallel all-MLP model-forward propagation gates."""
    seed_values = [int(part.strip()) for part in seeds.split(",") if part.strip()]
    jobs = [
        {
            "name": f"seed_{seed}_eval_{eval_prompts}",
            "seed": seed,
            "eval_prompts": eval_prompts,
        }
        for seed in seed_values
    ]
    for result in run_model_forward_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_b(seeds: str = "6,7,8", eval_prompts: int = 64):
    """Run mixed-rate stack gates: candidate MLPs with IQ4 non-MLP weights."""
    seed_values = [int(part.strip()) for part in seeds.split(",") if part.strip()]
    jobs = [
        {
            "name": f"mixed_seed_{seed}_eval_{eval_prompts}",
            "seed": seed,
            "eval_prompts": eval_prompts,
        }
        for seed in seed_values
    ]
    for result in run_mixed_rate_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_c(baselines: str = "iq3_m,q3_k_m,iq3_xs", seeds: str = "6,7,8", eval_prompts: int = 64):
    """Run production 3-bit GGUF bakeoff jobs against the JMRC MLP replacement."""
    baseline_labels = [part.strip() for part in baselines.split(",") if part.strip()]
    seed_values = [int(part.strip()) for part in seeds.split(",") if part.strip()]
    jobs = []
    for label in baseline_labels:
        if label not in BASELINE_GGUFS:
            raise ValueError(f"unknown baseline {label}; choose from {sorted(BASELINE_GGUFS)}")
        spec = BASELINE_GGUFS[label]
        for seed in seed_values:
            jobs.append(
                {
                    "name": f"{label}_seed_{seed}_eval_{eval_prompts}",
                    "baseline_label": label,
                    "gguf_repo_id": spec["repo_id"],
                    "gguf_filename": spec["filename"],
                    "seed": seed,
                    "eval_prompts": eval_prompts,
                }
            )
    for result in run_baseline_bakeoff_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_c1_overlay(
    baselines: str = "iq3_xs,q3_k_m",
    seeds: str = "6,7,8",
    eval_prompts: int = 64,
    block_size: int = 64,
    codebook_size: int = 64,
):
    """Run Run 007 C1 production-base residual overlay gates."""
    baseline_labels = [part.strip() for part in baselines.split(",") if part.strip()]
    seed_values = [int(part.strip()) for part in seeds.split(",") if part.strip()]
    jobs = []
    for label in baseline_labels:
        if label not in BASELINE_GGUFS:
            raise ValueError(f"unknown baseline {label}; choose from {sorted(BASELINE_GGUFS)}")
        spec = BASELINE_GGUFS[label]
        for seed in seed_values:
            jobs.append(
                {
                    "name": f"c1_overlay_{label}_seed_{seed}_eval_{eval_prompts}_b{block_size}_k{codebook_size}",
                    "baseline_label": label,
                    "gguf_repo_id": spec["repo_id"],
                    "gguf_filename": spec["filename"],
                    "seed": seed,
                    "eval_prompts": eval_prompts,
                    "block_size": block_size,
                    "codebook_size": codebook_size,
                    "variants": "fp16,iq4_all,prod_residual_random_mlp_iq4_rest,prod_residual_mlp_iq4_rest",
                }
            )
    for result in run_baseline_bakeoff_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_c2_mix(
    seeds: str = "6",
    eval_prompts: int = 64,
    calib_prompts: int = 12,
    group_mode: str = "tensor",
    low_source: str = "iq3_xs",
    target_source: str = "q3_k_m",
    high_sources: str = "q3_k_m,iq4_xs",
    candidate_variant: str = "c2_calib_greedy_mixed",
    knapsack_max_states: int = 50000,
    local_search_from: str = "",
    local_search_steps: int = 0,
    local_search_candidates: int = 24,
    local_search_min_improvement: float = 0.0001,
    genetic_search_from: str = "",
    genetic_search_generations: int = 0,
    genetic_search_population: int = 8,
    genetic_search_elite: int = 2,
    genetic_search_mutation_rate: float = 0.25,
    genetic_search_direct: bool = False,
    sweep_payload_bpws: str = "",
    sweep_selectors: str = "calib_knapsack",
    demotion_sources: str = "",
    demotion_base_source: str = "",
    demotion_selectors: str = "reverse_knapsack",
    max_shrink_nll_loss: float = 0.05,
    result_bucket: str = "run_007_c2_production_mixed_rate",
):
    """Run Run 007 C2 production mixed-rate tensor allocation gates."""
    seed_values = [int(part.strip()) for part in seeds.split(",") if part.strip()]
    candidate_suffix = _candidate_suffix(candidate_variant)
    frontier_suffix = ""
    if sweep_payload_bpws:
        frontier_suffix += f"_sweep_{sweep_payload_bpws.replace(',', '_').replace('.', 'p')}"
    if local_search_steps:
        frontier_suffix += f"_local_{local_search_steps}x{local_search_candidates}"
    if genetic_search_generations:
        frontier_suffix += f"_genetic_{genetic_search_generations}x{genetic_search_population}"
        if genetic_search_direct:
            frontier_suffix += "_direct"
    if demotion_sources:
        frontier_suffix += f"_demote_{demotion_sources.replace(',', '_')}"
    jobs = [
        {
            "name": (
                f"c2_mix_low_{low_source}_target_{target_source}_high_"
                f"{high_sources.replace(',', '_')}_seed_{seed}_eval_{eval_prompts}_calib_{calib_prompts}_{group_mode}"
                f"{candidate_suffix}{frontier_suffix}"
            ),
            "seed": seed,
            "eval_prompts": eval_prompts,
            "calib_prompts": calib_prompts,
            "group_mode": group_mode,
            "low_source": low_source,
            "target_source": target_source,
            "high_sources": high_sources,
            "candidate_variant": candidate_variant,
            "knapsack_max_states": knapsack_max_states,
            "local_search_from": local_search_from,
            "local_search_steps": local_search_steps,
            "local_search_candidates": local_search_candidates,
            "local_search_min_improvement": local_search_min_improvement,
            "genetic_search_from": genetic_search_from,
            "genetic_search_generations": genetic_search_generations,
            "genetic_search_population": genetic_search_population,
            "genetic_search_elite": genetic_search_elite,
            "genetic_search_mutation_rate": genetic_search_mutation_rate,
            "genetic_search_direct": genetic_search_direct,
            "sweep_payload_bpws": sweep_payload_bpws,
            "sweep_selectors": sweep_selectors,
            "demotion_sources": demotion_sources,
            "demotion_base_source": demotion_base_source or None,
            "demotion_selectors": demotion_selectors,
            "max_shrink_nll_loss": max_shrink_nll_loss,
            "result_bucket": result_bucket,
        }
        for seed in seed_values
    ]
    for result in run_production_mix_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_c2_replicate(
    model_key: str = "qwen3_0p6b_base",
    seeds: str = "6,7,8",
    eval_prompts: int = 256,
    calib_prompts: int = 12,
    layers: str = "",
    group_mode: str = "tensor",
    low_source: str = "iq3_xs",
    target_source: str = "iq3_m",
    high_sources: str = "q3_k_m,iq4_xs",
    candidate_variant: str = "c2_calib_greedy_mixed",
    knapsack_max_states: int = 50000,
    local_search_from: str = "",
    local_search_steps: int = 0,
    local_search_candidates: int = 24,
    local_search_min_improvement: float = 0.0001,
    genetic_search_from: str = "",
    genetic_search_generations: int = 0,
    genetic_search_population: int = 8,
    genetic_search_elite: int = 2,
    genetic_search_mutation_rate: float = 0.25,
    genetic_search_direct: bool = False,
    sweep_payload_bpws: str = "",
    sweep_selectors: str = "calib_knapsack",
    demotion_sources: str = "",
    demotion_base_source: str = "",
    demotion_selectors: str = "reverse_knapsack",
    max_shrink_nll_loss: float = 0.05,
    result_bucket: str = "run_008_c2_replication",
):
    """Run a C2 mixed-rate replication gate on a configured second model."""
    seed_values = [int(part.strip()) for part in seeds.split(",") if part.strip()]
    candidate_suffix = _candidate_suffix(candidate_variant)
    frontier_suffix = ""
    if sweep_payload_bpws:
        frontier_suffix += f"_sweep_{sweep_payload_bpws.replace(',', '_').replace('.', 'p')}"
    if local_search_steps:
        frontier_suffix += f"_local_{local_search_steps}x{local_search_candidates}"
    if genetic_search_generations:
        frontier_suffix += f"_genetic_{genetic_search_generations}x{genetic_search_population}"
        if genetic_search_direct:
            frontier_suffix += "_direct"
    if demotion_sources:
        frontier_suffix += f"_demote_{demotion_sources.replace(',', '_')}"
    jobs = [
        {
            "name": (
                f"{model_key}_c2_mix_low_{low_source}_target_{target_source}_high_"
                f"{high_sources.replace(',', '_')}_seed_{seed}_eval_{eval_prompts}_calib_{calib_prompts}_{group_mode}"
                f"{candidate_suffix}{frontier_suffix}"
            ),
            "model_key": model_key,
            "seed": seed,
            "eval_prompts": eval_prompts,
            "calib_prompts": calib_prompts,
            "layers": layers or MODEL_CONFIGS[model_key]["layers"],
            "group_mode": group_mode,
            "low_source": low_source,
            "target_source": target_source,
            "high_sources": high_sources,
            "candidate_variant": candidate_variant,
            "knapsack_max_states": knapsack_max_states,
            "local_search_from": local_search_from,
            "local_search_steps": local_search_steps,
            "local_search_candidates": local_search_candidates,
            "local_search_min_improvement": local_search_min_improvement,
            "genetic_search_from": genetic_search_from,
            "genetic_search_generations": genetic_search_generations,
            "genetic_search_population": genetic_search_population,
            "genetic_search_elite": genetic_search_elite,
            "genetic_search_mutation_rate": genetic_search_mutation_rate,
            "genetic_search_direct": genetic_search_direct,
            "sweep_payload_bpws": sweep_payload_bpws,
            "sweep_selectors": sweep_selectors,
            "demotion_sources": demotion_sources,
            "demotion_base_source": demotion_base_source or None,
            "demotion_selectors": demotion_selectors,
            "max_shrink_nll_loss": max_shrink_nll_loss,
            "result_bucket": result_bucket,
        }
        for seed in seed_values
    ]
    for result in run_production_mix_configured_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_c2_artifact(
    model_key: str = "",
    seed: int = 7,
    eval_prompts: int = 64,
    calib_prompts: int = 12,
    group_mode: str = "tensor",
    low_source: str = "iq3_xs",
    target_source: str = "q3_k_m",
    high_sources: str = "q3_k_m,iq4_xs",
    variant: str = "c2_calib_greedy_mixed",
    result_bucket: str = "run_007_c2_production_mixed_rate",
    result_name: str = "",
    artifact_name: str = "",
    output_gguf: str = "",
    artifact_bucket: str = "run_007_c2_mixed_gguf_artifact",
):
    """Build a loadable mixed GGUF artifact from a completed C2 allocation result."""
    default_result_name = (
        f"c2_mix_low_{low_source}_target_{target_source}_high_"
        f"{high_sources.replace(',', '_')}_seed_{seed}_eval_{eval_prompts}_calib_{calib_prompts}_{group_mode}"
    )
    result_name = result_name or default_result_name
    result_json = f"{RESULT_ROOT}/{result_bucket}/{result_name}/result.json"
    variant_suffix = variant.replace("c2_", "").replace("_mixed", "")
    job = {
        "name": artifact_name or f"{result_name}_{variant_suffix}_artifact",
        "result_json": result_json,
        "variant": variant,
        "metadata_source": low_source,
        "target_source": target_source,
        "high_sources": high_sources,
        "artifact_bucket": artifact_bucket,
        "output_gguf": output_gguf or f"{result_name}_{variant_suffix}.gguf",
    }
    if model_key:
        job["model_key"] = model_key
    print(json.dumps(run_c2_artifact_job.remote(job), indent=2))


@app.local_entrypoint()
def phase_c2_public_calibrated(
    model_keys: str = "qwen3_1p7b,qwen3_0p6b_base",
    seed: int = 7,
    eval_prompts: int = 512,
    calib_prompts: int = 48,
    layers: str = "",
    group_mode: str = "tensor",
    low_source: str = "iq2_m",
    target_source: str = "iq3_xs",
    high_sources: str = "q2_k_l,q3_k_s,q3_k_m,iq4_xs",
    calib_max_length: int = 192,
    eval_max_length: int = 256,
    dataset: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    calib_split: str = "train",
    eval_split: str = "validation",
    prompt_seed: int = 2701,
    candidate_variant: str = "c2_calib_greedy_mixed",
    knapsack_max_states: int = 50000,
    local_search_from: str = "",
    local_search_steps: int = 0,
    local_search_candidates: int = 24,
    local_search_min_improvement: float = 0.0001,
    genetic_search_from: str = "",
    genetic_search_generations: int = 0,
    genetic_search_population: int = 8,
    genetic_search_elite: int = 2,
    genetic_search_mutation_rate: float = 0.25,
    genetic_search_direct: bool = False,
    sweep_payload_bpws: str = "",
    sweep_selectors: str = "calib_knapsack",
    demotion_sources: str = "",
    demotion_base_source: str = "",
    demotion_selectors: str = "reverse_knapsack",
    max_shrink_nll_loss: float = 0.05,
    result_bucket: str = "run_008_c2_public_calibrated_wikitext",
):
    """Calibrate PMRA on public data and evaluate on a held-out public split."""
    keys = [part.strip() for part in model_keys.split(",") if part.strip()]
    candidate_suffix = _candidate_suffix(candidate_variant)
    frontier_suffix = ""
    if sweep_payload_bpws:
        frontier_suffix += f"_sweep_{sweep_payload_bpws.replace(',', '_').replace('.', 'p')}"
    if local_search_steps:
        frontier_suffix += f"_local_{local_search_steps}x{local_search_candidates}"
    if genetic_search_generations:
        frontier_suffix += f"_genetic_{genetic_search_generations}x{genetic_search_population}"
        if genetic_search_direct:
            frontier_suffix += "_direct"
    if demotion_sources:
        frontier_suffix += f"_demote_{demotion_sources.replace(',', '_')}"
    jobs = []
    for model_key in keys:
        jobs.append(
            {
                "name": (
                    f"{model_key}_c2_publiccal_{dataset.replace('/', '_')}_{dataset_config.replace('/', '_')}_"
                    f"{calib_split}_to_{eval_split}_low_{low_source}_target_{target_source}_"
                    f"high_{high_sources.replace(',', '_')}_seed_{seed}_eval_{eval_prompts}_"
                    f"calib_{calib_prompts}_{group_mode}_len_{eval_max_length}"
                    f"{candidate_suffix}{frontier_suffix}"
                ),
                "model_key": model_key,
                "seed": seed,
                "eval_prompts": eval_prompts,
                "calib_prompts": calib_prompts,
                "layers": layers or MODEL_CONFIGS[model_key]["layers"],
                "group_mode": group_mode,
                "low_source": low_source,
                "target_source": target_source,
                "high_sources": high_sources,
                "calib_max_length": calib_max_length,
                "eval_max_length": eval_max_length,
                "prompt_source": "public",
                "dataset": dataset,
                "dataset_config": dataset_config,
                "calib_split": calib_split,
                "eval_split": eval_split,
                "prompt_seed": prompt_seed,
                "candidate_variant": candidate_variant,
                "knapsack_max_states": knapsack_max_states,
                "local_search_from": local_search_from,
                "local_search_steps": local_search_steps,
                "local_search_candidates": local_search_candidates,
                "local_search_min_improvement": local_search_min_improvement,
                "genetic_search_from": genetic_search_from,
                "genetic_search_generations": genetic_search_generations,
                "genetic_search_population": genetic_search_population,
                "genetic_search_elite": genetic_search_elite,
                "genetic_search_mutation_rate": genetic_search_mutation_rate,
                "genetic_search_direct": genetic_search_direct,
                "sweep_payload_bpws": sweep_payload_bpws,
                "sweep_selectors": sweep_selectors,
                "demotion_sources": demotion_sources,
                "demotion_base_source": demotion_base_source or None,
                "demotion_selectors": demotion_selectors,
                "max_shrink_nll_loss": max_shrink_nll_loss,
                "result_bucket": result_bucket,
            }
        )
    for result in run_production_mix_configured_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_c2_public_eval(
    model_key: str = "qwen3_1p7b",
    seed: int = 7,
    eval_prompts: int = 1024,
    calib_prompts: int = 48,
    group_mode: str = "tensor",
    low_source: str = "iq2_m",
    target_source: str = "iq3_xs",
    high_sources: str = "q2_k_l,q3_k_s,q3_k_m,iq4_xs",
    variants: str = "fp16,iq2_m,iq3_xs,q3_k_s,c2_calib_greedy_mixed,c2_random_same_budget",
    candidate_variant: str = "c2_calib_greedy_mixed",
    random_variant: str = "c2_random_same_budget",
    prompt_count: int = 512,
    eval_max_length: int = 256,
    dataset: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    result_bucket: str = "run_008_c2_subq3_iq2m_to_iq3xs_calib48_eval1024",
    result_name: str = "",
    public_bucket: str = "run_008_pmra_public_eval",
):
    """Evaluate an existing PMRA selection on a separate public HF dataset."""
    dataset_config_value = None if dataset_config.lower() in {"", "none", "null"} else dataset_config
    job = {
        "name": (
            f"{model_key}_pmra_public_{dataset.replace('/', '_')}_{dataset_config.replace('/', '_')}_"
            f"seed_{seed}_prompts_{prompt_count}_len_{eval_max_length}"
        ),
        "model_key": model_key,
        "seed": seed,
        "eval_prompts": eval_prompts,
        "calib_prompts": calib_prompts,
        "group_mode": group_mode,
        "low_source": low_source,
        "target_source": target_source,
        "high_sources": high_sources,
        "variants": variants,
        "candidate_variant": candidate_variant,
        "random_variant": random_variant,
        "prompt_count": prompt_count,
        "eval_max_length": eval_max_length,
        "dataset": dataset,
        "dataset_config": dataset_config_value,
        "split": split,
        "result_bucket": result_bucket,
        "result_name": result_name or None,
        "public_bucket": public_bucket,
    }
    print(json.dumps(run_pmra_public_eval_job.remote(job), indent=2))


@app.local_entrypoint()
def phase_c5_iq4_erasure(
    seeds: str = "6",
    eval_prompts: int = 64,
    calib_prompts: int = 12,
    group_mode: str = "layer_family",
    target_saving_bpw: float = 0.75,
    modes: str = "q_bit0_mid,q_bit1_mid,scale_hi0",
):
    """Run Run 007 C5 IQ4 semantic bitplane/subfield erasure gates."""
    seed_values = [int(part.strip()) for part in seeds.split(",") if part.strip()]
    jobs = [
        {
            "name": (
                f"c5_iq4_erasure_seed_{seed}_eval_{eval_prompts}_calib_{calib_prompts}_"
                f"{group_mode}_save_{str(target_saving_bpw).replace('.', 'p')}"
            ),
            "seed": seed,
            "eval_prompts": eval_prompts,
            "calib_prompts": calib_prompts,
            "group_mode": group_mode,
            "target_saving_bpw": target_saving_bpw,
            "modes": modes,
        }
        for seed in seed_values
    ]
    for result in run_iq4_erasure_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_d_small():
    """Run a small parallel composition sweep before spending model-forward budget."""
    jobs = []
    for codebook_size in [64, 96, 128, 160]:
        jobs.append(
            {
                "name": f"cb{codebook_size}_b32_seed6",
                "codebook_size": codebook_size,
                "block_size": 32,
                "seed": 6,
            }
        )
    for target_set in ["gate,up,down", "gate,up", "up,down", "up"]:
        jobs.append(
            {
                "name": f"targets_{target_set.replace(',', '_')}_cb128_seed6",
                "codebook_targets": target_set,
                "codebook_size": 128,
                "block_size": 32,
                "seed": 6,
            }
        )
    for result in run_composition_job.map(jobs):
        print(json.dumps(result, indent=2))


@app.local_entrypoint()
def phase_e(seed: int = 6):
    """Build and verify a logical encoded artifact for one seed."""
    job = {"name": f"artifact_seed_{seed}", "seed": seed}
    print(json.dumps(run_artifact_job.remote(job), indent=2))
