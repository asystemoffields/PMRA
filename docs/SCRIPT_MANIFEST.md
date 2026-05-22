# Script Manifest

The PMRA release package needs the PMRA entrypoint scripts and two local helper
modules they import.

## Required Entrypoints

- `scripts/production_mixed_rate_transcoder_gate.py`
- `scripts/build_mixed_gguf_artifact.py`
- `scripts/evaluate_pmra_public_dataset.py`
- `scripts/evaluate_pmra_code_likelihood.py`
- `scripts/summarize_pmra_results.py`

## Required Helper Modules

- `scripts/activation_conditioned_scale_mirage.py`
- `scripts/mlp_codebook_model_forward_gate.py`

## Modal Harness

The Modal harness is useful for reproducing the exact A100 runs. It is included
in this repository, but it is not required for local script use if the user
already has model files and GGUF controls.

- `modal/modal_sprint.py`

## Release Helpers

These are optional helpers for maintaining the Hugging Face release package.

- `tools/upload_hf_release.py`
- `tools/verify_hf_release.py`

## Python Dependencies

See `requirements.txt`.

## Release Packaging Note

If this candidate is split into a standalone GitHub repository, preserve the
relative import layout by placing all required scripts in one `scripts/`
directory, or convert the scripts into a package with explicit module imports.
