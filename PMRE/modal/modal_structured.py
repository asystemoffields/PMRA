import modal

app = modal.App("smol-structured-search")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "scipy",
        "transformers",
        "datasets",
        "huggingface-hub",
        "accelerate",
        "zstandard",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file("structured_search.py", "/root/structured_search.py")
)

results_vol = modal.Volume.from_name("olmo-profiler-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": results_vol, "/hf_cache": hf_cache},
)
def run():
    import os
    import sys

    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/hf_cache"
    sys.argv = [
        "structured_search.py",
        "--output", "/results/structured_search",
        "--device", "cuda",
        "--n-calib", "128",
        "--n-eval", "64",
        "--layers", "0", "14", "29",
    ]

    from structured_search import main
    main(commit_fn=results_vol.commit)


@app.local_entrypoint()
def entry():
    print(">>> Running structured family search (with Fisher-weighted families)...")
    print(">>> TIP: use `modal run --detach modal_structured.py` to survive disconnects!")
    run.remote()
    print("\nDone. Download results with:")
    print("  modal volume get olmo-profiler-results structured_search results/structured_search")
