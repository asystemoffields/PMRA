import modal

app = modal.App("dsmollm-refine")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy", "scipy", "transformers", "datasets",
        "huggingface-hub", "accelerate", "zstandard",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file("structured_search.py", "/root/structured_search.py")
    .add_local_file("compress_model.py", "/root/compress_model.py")
    .add_local_file("refine_model.py", "/root/refine_model.py")
)

results_vol = modal.Volume.from_name("olmo-profiler-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    volumes={"/results": results_vol, "/hf_cache": hf_cache},
)
def refine(allocation_name: str, n_steps: int = 20000,
           lr: float = 1e-4, alpha: float = 1.0, n_train: int = 100000,
           resume_path: str = None, teacher: str = None):
    import os, sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/hf_cache"
    argv = [
        "refine_model.py",
        "--allocation", f"/results/{allocation_name}",
        "--output", "/results/dsmollm_refined",
        "--device", "cuda",
        "--n-steps", str(n_steps),
        "--batch-size", "8",
        "--lr", str(lr),
        "--temperature", "2.0",
        "--alpha", str(alpha),
        "--n-eval", "128",
        "--n-calib", "64",
        "--n-train", str(n_train),
    ]
    if resume_path:
        argv.extend(["--resume", resume_path])
    if teacher:
        argv.extend(["--teacher", teacher])
    sys.argv = argv
    from refine_model import main
    main(commit_fn=results_vol.commit)


@app.local_entrypoint()
def entry(ratio: str = "2.0", steps: int = 20000,
          lr: float = 1e-4, alpha: float = 1.0, n_train: int = 100000,
          resume: str = None, teacher: str = None):
    alloc_name = f"allocation_{ratio}x.json"
    print(f">>> DSmolLM-{ratio}x: {steps} steps, lr={lr}, alpha={alpha}, "
          f"{n_train} train samples")
    if teacher:
        print(f">>> Cross-distillation teacher: {teacher}")
    if resume:
        print(f">>> Resuming from: {resume}")
    print(f">>> Checkpoints saved every 500 steps. Use `modal run --detach`!")
    refine.remote(alloc_name, steps, lr, alpha, n_train, resume, teacher)
    print("\nDone.")
