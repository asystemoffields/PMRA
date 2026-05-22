from __future__ import annotations

import json
import shutil
import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = ROOT / "tmp/run008_c2_publiccal_qwen17_artifact/qwen17_publiccal_pmra_calib_greedy.gguf"
REPORT_JSON = ROOT / "artifacts/artifact_report.json"
REPORT_MD = ROOT / "artifacts/artifact_report.md"
MODEL_CARD = ROOT / "docs/HF_MODEL_CARD.md"
RELEASE_CARD = ROOT / "docs/RELEASE_CANDIDATE.md"
STAGING = ROOT / "tmp/hf_upload"
REPO_ID = "Asystemoffields/Qwen3-1.7B-PMRA-IQ3XS-budget-GGUF"


def copy_if_present(source: Path | None, destination: Path) -> None:
    if source is None:
        return
    if not source.exists():
        raise FileNotFoundError(f"missing upload file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def parse_extra_file(spec: str) -> tuple[Path, Path]:
    if "=" in spec:
        source, destination = spec.split("=", 1)
        return Path(source), Path(destination)
    source = Path(spec)
    return source, Path(source.name)


def assert_gguf_magic(path: Path) -> None:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        preview = magic.decode("utf-8", errors="replace")
        raise ValueError(f"{path} is not a GGUF file; first four bytes are {preview!r}")


def stage_files(
    gguf: Path,
    *,
    readme: Path,
    report_json: Path | None,
    report_md: Path | None,
    release_card: Path | None,
    staging_dir: Path,
    hf_filename: str,
    extra_files: list[tuple[Path, Path]],
) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    shutil.copy2(readme, staging_dir / "README.md")
    shutil.copy2(gguf, staging_dir / hf_filename)
    copy_if_present(report_json, staging_dir / "artifact_report.json")
    copy_if_present(report_md, staging_dir / "artifact_report.md")
    copy_if_present(release_card, staging_dir / release_card.name)
    for source, destination in extra_files:
        copy_if_present(source, staging_dir / destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--filename", default=None, help="Filename to use for the GGUF on Hugging Face.")
    parser.add_argument("--readme", type=Path, default=MODEL_CARD)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--release-card", type=Path, default=RELEASE_CARD)
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Additional file to include. Use PATH or PATH=DESTINATION.",
    )
    parser.add_argument("--staging-dir", type=Path, default=STAGING)
    parser.add_argument("--public", action="store_true", help="Create/update the HF repo as public instead of private.")
    args = parser.parse_args()

    if not args.gguf.exists():
        raise FileNotFoundError(f"missing GGUF: {args.gguf}")
    assert_gguf_magic(args.gguf)
    hf_filename = args.filename or args.gguf.name

    api = HfApi()
    who = api.whoami()
    print(json.dumps({"whoami": who.get("name"), "repo_id": args.repo_id}, indent=2), flush=True)
    stage_files(
        args.gguf,
        readme=args.readme,
        report_json=args.report_json,
        report_md=args.report_md,
        release_card=args.release_card,
        staging_dir=args.staging_dir,
        hf_filename=hf_filename,
        extra_files=[parse_extra_file(spec) for spec in args.extra_file],
    )
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    print("[hf] repo ready, uploading folder", flush=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.staging_dir),
        commit_message="Upload PMRA experimental GGUF release candidate",
    )
    print("[hf] upload complete", flush=True)


if __name__ == "__main__":
    main()
