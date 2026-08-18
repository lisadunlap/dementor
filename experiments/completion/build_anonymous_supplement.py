#!/usr/bin/env python3
"""Build a deterministic, identity-scrubbed AAAI review code/data package.

The review manuscript must not point to mutable web material.  This builder
selects the analysis code and compact aggregate evidence needed for review,
omits Git history, credentials, logs, model weights, and raw generations, and
fails closed if author-identifying strings or credential-like tokens remain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
ROOT_FILES = ("config.yaml", "pyproject.toml", "METHODS.md")
SOURCE_GLOBS = (
    "dementor/**/*.py",
    "experiments/_paths.py",
    "experiments/imitation_safety/*.py",
    "experiments/figures/*.py",
    "experiments/steering/port/*.py",
    "tests/test_results_scripts_golden.py",
    "tests/test_fidelity_completion.py",
    "tests/test_cone_eval_verdict.py",
)
# Obsolete one-off utilities can contain provider-side run identifiers that encode the
# submitting institution.  They are not needed to audit any manuscript result.
EXCLUDED_FILES = {
    "dementor/training/sync_openai_metrics.py",
}
RESULT_FILES = (
    "data/results/safety/erosion_coverage.json",
    "data/results/safety/erosion_seed42_long.csv",
    "data/results/safety/erosion_seed42_summary.csv",
    "data/results/safety/erosion_campaign_headlines.json",
    "data/results/safety/target_safety_transfer.json",
    "data/results/safety/self_sft_headlines.json",
    "data/results/safety/base_steering_coverage.json",
    "data/results/fidelity/fidelity_all_embed_long.csv",
    "data/results/fidelity/fidelity_all_judge_long.csv",
    "data/results/fidelity/fidelity_campaign_headlines.json",
    "paper/naz_aaai2027/img/fpall_comparison_stats.json",
    "paper/naz_aaai2027/img/steering_figure_stats.json",
    "paper/naz_aaai2027/target_safety_transfer_stats.json",
    "paper/naz_aaai2027/local_dpo_training_length_audit.json",
)
REPLACEMENTS = {
    "/home/eecs/ethantsliu": "/anonymous/home",
    "/data/ethantsliu": "/anonymous/data",
    "/work/ethantsliu": "/anonymous/work",
    "dementor-research": "anonymous-artifacts",
}
FORBIDDEN = (
    re.compile(r"ethantsliu", re.I),
    re.compile(r"lisadunlap", re.I),
    re.compile(r"dementor-research", re.I),
    re.compile(r"uc[-_ ]?berkeley", re.I),
    re.compile(r"trevor[-_ ]?darrell", re.I),
    re.compile(r"(?:sk-|hf_)[A-Za-z0-9_-]{20,}"),
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
)
README = """# Anonymous AAAI review supplement

This package contains the configuration, analysis implementation, compact
machine-readable aggregates, and regression tests needed to audit the paper's
reported results. It intentionally excludes Git history, credentials, model
weights, raw generations, runtime logs, and author-identifying metadata.

Coverage represented here:

- 528 SFT and 528 matched SFT-to-DPO safety cells;
- 48 self-SFT controls;
- two 200-prompt behavioral-fidelity scorers for all 1,104 adapters;
- 29 models by seven benchmarks by two base-steering operators;
- direct target-relative safety projection with crossed-role and joint-identity bootstraps;
- a tokenizer-level audit of the backend-specific local DPO sequence caps.

The target-relative analysis requires no new model inference. Recompute its
stage statistics with:

    python experiments/imitation_safety/variance_decomp.py --stage sft
    python experiments/imitation_safety/variance_decomp.py --stage dpo

Run the CPU regression tests with:

    pytest -q tests/test_results_scripts_golden.py \
      tests/test_fidelity_completion.py tests/test_cone_eval_verdict.py

Training and most local evaluation used 80 GB NVIDIA H100 GPUs; the largest
models used two- or three-card model parallelism, and one independently checked
steering run used a separate RTX worker. The Python analysis environment is
specified by pyproject.toml. Exact training-environment lockfiles were not
captured, so the corresponding reproducibility-checklist item is marked partial.

Permanent public locations are deliberately omitted during double-blind review.
"""


def selected_files() -> list[Path]:
    paths = {REPO / name for name in ROOT_FILES + RESULT_FILES}
    for pattern in SOURCE_GLOBS:
        paths.update(path for path in REPO.glob(pattern) if path.is_file())
    paths = {
        path for path in paths
        if str(path.relative_to(REPO)) not in EXCLUDED_FILES
    }
    missing = sorted(str(path.relative_to(REPO)) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"required supplement inputs missing: {missing}")
    return sorted(paths, key=lambda path: str(path.relative_to(REPO)))


def scrub(data: bytes, suffix: str) -> bytes:
    if suffix not in TEXT_SUFFIXES:
        return data
    text = data.decode("utf-8")
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("utf-8")


def validate(rel: str, data: bytes) -> None:
    if Path(rel).suffix not in TEXT_SUFFIXES:
        return
    text = data.decode("utf-8")
    for pattern in FORBIDDEN:
        if pattern.search(text):
            raise ValueError(f"forbidden identity/credential pattern in {rel}: {pattern.pattern}")


def write_zip(output: Path) -> dict:
    entries: dict[str, bytes] = {"README.md": README.encode("utf-8")}
    for path in selected_files():
        rel = str(path.relative_to(REPO))
        entries[rel] = scrub(path.read_bytes(), path.suffix)
    manifest = {
        rel: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for rel, data in sorted(entries.items())
    }
    entries["MANIFEST.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    for rel, data in entries.items():
        validate(rel, data)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for rel, data in sorted(entries.items()):
                info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                zf.writestr(info, data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "files": len(entries),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_zip(args.output.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
