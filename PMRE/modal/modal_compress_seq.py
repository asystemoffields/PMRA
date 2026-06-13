import modal

app = modal.App("dsmollm-sequential")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy", "scipy", "transformers", "datasets",
        "huggingface-hub", "accelerate", "zstandard",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file("structured_search.py", "/root/structured_search.py")
    .add_local_file("compress_sequential.py", "/root/compress_sequential.py")
)

results_vol = modal.Volume.from_name("olmo-profiler-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": results_vol, "/hf_cache": hf_cache},
)
def compress(allocation_name: str):
    import os, sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/hf_cache"
    sys.argv = [
        "compress_sequential.py",
        "--allocation", f"/results/{allocation_name}",
        "--output", "/results/dsmollm_seq",
        "--device", "cuda",
        "--n-eval", "128",
        "--n-calib", "64",
    ]
    from compress_sequential import main
    main(commit_fn=results_vol.commit)


@app.local_entrypoint()
def entry(ratio: str = "5.0"):
    alloc_name = f"allocation_{ratio}x.json"
    print(f">>> Sequential SAES-SVD compression at {ratio}x...")
    print(f">>> Use `modal run --detach` to survive disconnects!")
    compress.remote(alloc_name)
    print("\nDone.")
