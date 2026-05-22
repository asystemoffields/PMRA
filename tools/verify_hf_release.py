from __future__ import annotations

import hashlib
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "Asystemoffields/Qwen3-1.7B-PMRA-IQ3XS-budget-GGUF"
FILENAME = "qwen17_publiccal_pmra_calib_greedy.gguf"
EXPECTED_SHA256 = "cc405feb01fe8f79e44fc27f48fe15e5f591f9860dc304be6477886bf7548420"
OUT_DIR = Path("tmp/hf_download_smoke")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--filename", default=FILENAME)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="model",
            local_dir=str(args.out_dir),
        )
    )
    digest = sha256(downloaded)
    print(f"path={downloaded}")
    print(f"sha256={digest}")
    if digest != args.expected_sha256:
        raise SystemExit(f"hash mismatch: expected {args.expected_sha256}")
    print("hash_ok=true")


if __name__ == "__main__":
    main()
