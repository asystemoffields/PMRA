"""Fisher-guided GGUF quantization pipeline on Modal.

End-to-end pipeline that produces Fisher-optimal mixed-precision GGUFs:
1. Convert HuggingFace model to GGUF F16
2. Generate importance matrix (imatrix) from calibration data
3. Generate Fisher-guided per-tensor recipes at multiple BPW targets
4. Quantize with each recipe + imatrix
5. Produce standard baselines (Q4_K_M, Q3_K_M) for comparison
6. Evaluate perplexity on all variants
7. Save everything to Modal volume

== General Usage (any model) ==

  modal run modal_fisher_gguf.py --model-id <hf_model_id> \\
    --profiling <path_to_per_tensor.json> \\
    --targets 3.0 3.5 4.0

== OLMo-3-7B-Think (default) ==

  modal run modal_fisher_gguf.py

== What You Need ==

  1. Per-tensor profiling JSON (from structured_search.py or similar profiler)
  2. HuggingFace model ID
  3. Modal account with GPU access
  4. HuggingFace token (for gated models) as Modal secret "huggingface-secret"
"""
from pathlib import Path

import modal

app = modal.App("fisher-gguf")

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "cmake", "build-essential", "curl", "wget")
    .pip_install(
        "torch", "numpy", "transformers", "safetensors",
        "huggingface-hub", "accelerate", "datasets", "zstandard",
        "sentencepiece", "protobuf", "gguf",
    )
    .env({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    .run_commands(
        f"git clone --depth 1 {LLAMA_CPP_REPO} /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release "
        "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 "
        '-DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" && '
        "cmake --build build --target llama-quantize llama-imatrix llama-perplexity llama-cli -j$(nproc)",
    )
    .add_local_file("fisher_gguf_recipe.py", "/root/fisher_gguf_recipe.py")
)

results_vol = modal.Volume.from_name("olmo-profiler-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Step 1: Convert HF model to GGUF F16
# ---------------------------------------------------------------------------

def convert_hf_to_gguf(model_id: str, output_dir: str) -> str:
    """Download HF model and convert to GGUF F16."""
    import subprocess
    from pathlib import Path

    out_path = Path(output_dir) / "model-f16.gguf"
    if out_path.exists():
        print(f"[convert] F16 GGUF already exists: {out_path}")
        return str(out_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[convert] Downloading {model_id} and converting to F16 GGUF...")
    cmd = [
        "python", "/opt/llama.cpp/convert_hf_to_gguf.py",
        model_id,
        "--outfile", str(out_path),
        "--outtype", "f16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"convert_hf_to_gguf.py failed (exit {result.returncode})")

    size_gb = out_path.stat().st_size / 1e9
    print(f"[convert] F16 GGUF: {out_path} ({size_gb:.2f} GB)")
    return str(out_path)


# ---------------------------------------------------------------------------
# Step 2: Generate imatrix
# ---------------------------------------------------------------------------

def generate_imatrix(
    f16_gguf: str, output_dir: str, calib_text: str, n_chunks: int = 200
) -> str:
    """Generate importance matrix from calibration data."""
    import subprocess
    from pathlib import Path

    imatrix_path = Path(output_dir) / "imatrix.gguf"
    if imatrix_path.exists():
        print(f"[imatrix] Already exists: {imatrix_path}")
        return str(imatrix_path)

    # Write calibration text to file
    calib_path = Path(output_dir) / "calibration.txt"
    calib_path.write_text(calib_text, encoding="utf-8")

    print(f"[imatrix] Generating from {n_chunks} chunks...")
    cmd = [
        "/opt/llama.cpp/build/bin/llama-imatrix",
        "-m", f16_gguf,
        "-f", str(calib_path),
        "-ngl", "99",
        "-o", str(imatrix_path),
        "--chunks", str(n_chunks),
        "--process-output",
        "--no-ppl",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-3000:]}")
        print(f"STDERR:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"llama-imatrix failed (exit {result.returncode})")

    print(f"[imatrix] Saved: {imatrix_path}")
    return str(imatrix_path)


# ---------------------------------------------------------------------------
# Step 3: Generate Fisher recipes
# ---------------------------------------------------------------------------

def generate_recipes(
    profiling_path: str, output_dir: str, targets: list[float]
) -> dict[float, str]:
    """Generate Fisher-guided recipes at multiple BPW targets."""
    import sys
    sys.path.insert(0, "/root")
    from pathlib import Path
    from fisher_gguf_recipe import generate_recipe

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    recipes = {}

    for target in targets:
        recipe_path = Path(output_dir) / f"recipe_{target:.1f}bpw.txt"
        summary_path = Path(output_dir) / f"recipe_{target:.1f}bpw.json"

        lines, summary = generate_recipe(
            Path(profiling_path), target_bpw=target,
        )

        recipe_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        import json
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        recipes[target] = str(recipe_path)
        print(f"[recipe] {target:.1f} BPW → actual {summary['actual_bpw']:.3f} BPW, "
              f"{summary['total_gb']:.2f} GB, {summary['n_tensors']} tensors")

    return recipes


# ---------------------------------------------------------------------------
# Step 4: Quantize
# ---------------------------------------------------------------------------

def quantize(
    f16_gguf: str,
    output_path: str,
    base_type: str = "Q4_K_M",
    imatrix: str | None = None,
    tensor_type_file: str | None = None,
) -> dict:
    """Run llama-quantize with optional recipe and imatrix.

    Tries --tensor-type-file first (merged Jan 2026). Falls back to
    individual --tensor-type arguments if the flag isn't recognized.
    """
    import subprocess
    from pathlib import Path

    out = Path(output_path)
    if out.exists():
        size_gb = out.stat().st_size / 1e9
        print(f"[quantize] Already exists: {out} ({size_gb:.2f} GB)")
        return {"path": str(out), "size_gb": size_gb}

    cmd = ["/opt/llama.cpp/build/bin/llama-quantize"]
    if imatrix:
        cmd += ["--imatrix", imatrix]

    if tensor_type_file:
        # Try --tensor-type-file first
        cmd_try = cmd + ["--tensor-type-file", tensor_type_file]
        cmd_try += [f16_gguf, str(out), base_type]

        print(f"[quantize] Trying --tensor-type-file...")
        result = subprocess.run(cmd_try, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0 and "tensor-type-file" in result.stderr.lower():
            # Fallback: individual --tensor-type arguments
            print(f"[quantize] --tensor-type-file not supported, using individual args")
            recipe_lines = Path(tensor_type_file).read_text().strip().split("\n")
            cmd_fallback = cmd[:]
            for line in recipe_lines:
                line = line.strip()
                if line and "=" in line:
                    cmd_fallback += ["--tensor-type", line]
            cmd_fallback += [f16_gguf, str(out), base_type]
            result = subprocess.run(
                cmd_fallback, capture_output=True, text=True, timeout=3600
            )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout[-3000:]}")
            print(f"STDERR:\n{result.stderr[-3000:]}")
            raise RuntimeError(f"llama-quantize failed (exit {result.returncode})")
    else:
        cmd += [f16_gguf, str(out), base_type]
        print(f"[quantize] {base_type}"
              f"{' +imatrix' if imatrix else ''}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout[-3000:]}")
            print(f"STDERR:\n{result.stderr[-3000:]}")
            raise RuntimeError(f"llama-quantize failed (exit {result.returncode})")

    size_gb = out.stat().st_size / 1e9
    print(f"[quantize] Output: {out} ({size_gb:.2f} GB)")
    return {"path": str(out), "size_gb": size_gb}


# ---------------------------------------------------------------------------
# Step 5: Evaluate perplexity
# ---------------------------------------------------------------------------

def eval_perplexity(gguf_path: str, eval_text: str, output_dir: str) -> dict:
    """Run llama-perplexity on the quantized model."""
    import subprocess
    from pathlib import Path

    eval_path = Path(output_dir) / "eval_data.txt"
    eval_path.write_text(eval_text, encoding="utf-8")

    print(f"[eval] Evaluating {Path(gguf_path).name}...")
    cmd = [
        "/opt/llama.cpp/build/bin/llama-perplexity",
        "-m", gguf_path,
        "-f", str(eval_path),
        "-ngl", "99",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    # Parse perplexity from output
    ppl = None
    for line in result.stdout.split("\n"):
        if "Final estimate:" in line or "perplexity" in line.lower():
            import re
            match = re.search(r"[\d.]+", line.split("=")[-1] if "=" in line else line)
            if match:
                try:
                    ppl = float(match.group())
                except ValueError:
                    pass

    # Also try the last line pattern: "Final estimate: PPL = X.XXX +/- Y.YYY"
    if ppl is None:
        for line in reversed(result.stdout.split("\n")):
            if "PPL" in line:
                import re
                match = re.search(r"PPL\s*=\s*([\d.]+)", line)
                if match:
                    ppl = float(match.group(1))
                    break

    size_gb = Path(gguf_path).stat().st_size / 1e9
    name = Path(gguf_path).stem

    if ppl is not None:
        print(f"[eval] {name}: PPL = {ppl:.4f} ({size_gb:.2f} GB)")
    else:
        print(f"[eval] {name}: PPL parse failed ({size_gb:.2f} GB)")
        print(f"  Last 500 chars of stdout: {result.stdout[-500:]}")

    return {
        "name": name,
        "path": gguf_path,
        "size_gb": size_gb,
        "ppl": ppl,
        "stdout_tail": result.stdout[-1000:],
    }


# ---------------------------------------------------------------------------
# Calibration data loader
# ---------------------------------------------------------------------------

def load_calibration_data(dataset_name: str = "wikitext", min_tokens: int = 100_000) -> str:
    """Load diverse calibration text for imatrix generation.

    Uses wikitext-2 train split by default. For production, consider
    mixing in instruction/chat data for instruct/Think models.
    """
    from datasets import load_dataset

    print(f"[data] Loading calibration data from {dataset_name}...")

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(row["text"] for row in ds if row["text"].strip())
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        chunks = []
        total_chars = 0
        for row in ds:
            chunks.append(row["text"])
            total_chars += len(row["text"])
            if total_chars > min_tokens * 5:
                break
        text = "\n".join(chunks)
    else:
        ds = load_dataset(dataset_name, split="train")
        text_field = "text" if "text" in ds.column_names else ds.column_names[0]
        text = "\n".join(row[text_field] for row in ds if row[text_field].strip())

    print(f"[data] Loaded {len(text):,} characters of calibration text")
    return text


def load_eval_data() -> str:
    """Load evaluation data (wikitext-2 test split)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    print(f"[data] Loaded {len(text):,} characters of eval text")
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="H100",
    memory=65536,
    timeout=14400,  # 4 hours
    volumes={"/results": results_vol, "/hf_cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run(
    model_id: str = "allenai/OLMo-3-7B-Think",
    profiling_json_content: str = "",
    output_dir: str = "/results/fisher_gguf",
    targets: list[float] = None,
    baselines: list[str] = None,
    skip_eval: bool = False,
    skip_imatrix: bool = False,
    imatrix_chunks: int = 200,
):
    import json
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = "/hf_cache"

    if targets is None:
        targets = [3.0, 3.5, 4.0]
    if baselines is None:
        baselines = ["Q4_K_M", "Q3_K_M", "Q3_K_S"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write profiling data to disk (passed from local entrypoint)
    profiling_path = str(out / "per_tensor.json")
    Path(profiling_path).write_text(profiling_json_content, encoding="utf-8")
    print(f"[setup] Wrote profiling data: {len(profiling_json_content):,} bytes")

    all_results = {
        "model_id": model_id,
        "profiling": profiling_path,
        "variants": {},
    }

    # Step 1: Convert to F16 GGUF
    print("\n" + "=" * 60)
    print("STEP 1: Convert to F16 GGUF")
    print("=" * 60)
    f16_gguf = convert_hf_to_gguf(model_id, str(out))

    # Step 2: Generate imatrix
    imatrix_path = None
    if not skip_imatrix:
        print("\n" + "=" * 60)
        print("STEP 2: Generate importance matrix")
        print("=" * 60)
        calib_text = load_calibration_data()
        imatrix_path = generate_imatrix(
            f16_gguf, str(out), calib_text, n_chunks=imatrix_chunks
        )

    # Step 3: Generate Fisher recipes
    print("\n" + "=" * 60)
    print("STEP 3: Generate Fisher-guided recipes")
    print("=" * 60)
    recipes = generate_recipes(profiling_path, str(out), targets)

    # Step 4: Quantize — Fisher variants + baselines
    print("\n" + "=" * 60)
    print("STEP 4: Quantize all variants")
    print("=" * 60)

    # Fisher-guided variants
    for target, recipe_path in recipes.items():
        name = f"fisher-{target:.1f}bpw"
        gguf_path = str(out / f"model-{name}.gguf")
        info = quantize(
            f16_gguf, gguf_path,
            base_type="Q4_K_M",  # base type (overridden by recipe for most tensors)
            imatrix=imatrix_path,
            tensor_type_file=recipe_path,
        )
        all_results["variants"][name] = info

    # Standard baselines (with imatrix for fair comparison)
    for base_type in baselines:
        name = f"baseline-{base_type}"
        gguf_path = str(out / f"model-{name}.gguf")
        info = quantize(
            f16_gguf, gguf_path,
            base_type=base_type,
            imatrix=imatrix_path,
        )
        all_results["variants"][name] = info

    # Also produce a baseline WITHOUT imatrix for reference
    name = "baseline-Q4_K_M-no-imatrix"
    gguf_path = str(out / f"model-{name}.gguf")
    info = quantize(f16_gguf, gguf_path, base_type="Q4_K_M")
    all_results["variants"][name] = info

    # Commit volume before eval (long step)
    results_vol.commit()

    # Step 5: Evaluate perplexity
    if not skip_eval:
        print("\n" + "=" * 60)
        print("STEP 5: Evaluate perplexity")
        print("=" * 60)
        eval_text = load_eval_data()

        for name, info in all_results["variants"].items():
            eval_result = eval_perplexity(info["path"], eval_text, str(out))
            info["ppl"] = eval_result["ppl"]

    # Save results
    results_path = out / "fisher_gguf_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    results_vol.commit()

    # Print final comparison table
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Variant':<35} {'Size (GB)':>10} {'PPL':>10}")
    print("-" * 60)
    for name, info in sorted(
        all_results["variants"].items(),
        key=lambda x: x[1].get("size_gb", 99),
    ):
        ppl_str = f"{info['ppl']:.4f}" if info.get("ppl") else "N/A"
        print(f"{name:<35} {info['size_gb']:>9.2f} {ppl_str:>10}")

    print(f"\nResults saved to: {results_path}")
    print(f"GGUFs saved to: {output_dir}/")
    return all_results


DEFAULT_OLMO_PROFILING = str(
    Path(__file__).resolve().parent / "results" / "think" / "per_tensor.json"
)


@app.local_entrypoint()
def main(
    model_id: str = "allenai/OLMo-3-7B-Think",
    profiling: str = DEFAULT_OLMO_PROFILING,
    output_dir: str = "/results/fisher_gguf",
    targets: str = "3.0,3.5,4.0",
    baselines: str = "Q4_K_M,Q3_K_M,Q3_K_S",
    skip_eval: bool = False,
    skip_imatrix: bool = False,
    imatrix_chunks: int = 200,
    detach: bool = False,
):
    from pathlib import Path as P

    target_list = [float(t.strip()) for t in targets.split(",")]
    baseline_list = [b.strip() for b in baselines.split(",")]

    # Read profiling data locally and pass as string to remote
    profiling_path = P(profiling)
    if not profiling_path.exists():
        raise FileNotFoundError(
            f"Profiling data not found: {profiling_path}\n"
            f"Generate it with structured_search.py or provide --profiling <path>"
        )
    profiling_content = profiling_path.read_text(encoding="utf-8")

    print(f"Fisher GGUF Pipeline")
    print(f"  Model: {model_id}")
    print(f"  Profiling: {profiling_path} ({len(profiling_content):,} bytes)")
    print(f"  Targets: {target_list}")
    print(f"  Baselines: {baseline_list}")
    print()

    kwargs = dict(
        model_id=model_id,
        profiling_json_content=profiling_content,
        output_dir=output_dir,
        targets=target_list,
        baselines=baseline_list,
        skip_eval=skip_eval,
        skip_imatrix=skip_imatrix,
        imatrix_chunks=imatrix_chunks,
    )

    if detach:
        fc = run.spawn(**kwargs)
        print(f"Spawned as function call: {fc.object_id}")
        print("Check Modal dashboard for progress.")
    else:
        results = run.remote(**kwargs)
        print("\nDone! Results:")
        import json
        print(json.dumps(results, indent=2, default=str))
