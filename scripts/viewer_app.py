#!/usr/bin/env python3
"""
Interactive Gradio viewer for disguising runs.

Features
- Discover disguise result CSVs under data/results/.
- Inspect prompt, disguised response, target response, and optional source baseline side by side.
- Filter prompts by substring search and jump between matches quickly.
- Auto-suggest matching source/target baseline CSV paths based on run metadata (editable in the UI).
"""
from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import types

import gradio as gr
import pandas as pd

RESULTS_ROOT = Path("data/results")
BASE_RESPONSES_ROOT = Path("data/model-responses")
BASELINE_SEARCH_ROOTS: List[Tuple[str, Path]] = [
    ("results", RESULTS_ROOT),
    ("model-responses", BASE_RESPONSES_ROOT),
]
_EXCLUDED_DIR_NAMES = {"scores", "metrics", "judgments"}
_EXCLUDED_SEGMENT_SUFFIXES = ("_scores", "_metrics", "_judgments")
_EXCLUDED_FILE_STEMS = {"scored", "scored_metrics"}
RUN_METADATA: Dict[str, Dict[str, Optional[str]]] = {}
ALL_DATASETS_LABEL = "All datasets"
ALL_SUBSETS_LABEL = "All subsets"
ALL_METHODS_LABEL = "All methods"


def _has_excluded_segment(path: Path) -> bool:
    for part in path.parts:
        normalized = part.lower()
        if normalized in _EXCLUDED_DIR_NAMES:
            return True
        if any(normalized.endswith(suffix) for suffix in _EXCLUDED_SEGMENT_SUFFIXES):
            return True
    return path.stem.lower() in _EXCLUDED_FILE_STEMS


class RunInfo(Tuple[str, str]):
    """Typed alias for (label, path) pairs."""


def _normalize_filter_value(value: Optional[str]) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "unknown"


def _sort_filter_values(values: set[str]) -> List[str]:
    return sorted(values, key=lambda v: (v == "unknown", v))


def _dataset_matches(meta_value: Optional[str], selection: str) -> bool:
    if selection == ALL_DATASETS_LABEL:
        return True
    return _normalize_filter_value(meta_value) == selection


def _subset_matches(meta_value: Optional[str], selection: Optional[str]) -> bool:
    if not selection or selection == ALL_SUBSETS_LABEL:
        return True
    return _normalize_filter_value(meta_value) == selection


def _dataset_choices() -> List[str]:
    if not RUN_METADATA:
        return [ALL_DATASETS_LABEL]
    values = {_normalize_filter_value(meta.get("dataset")) for meta in RUN_METADATA.values()}
    ordered = _sort_filter_values(values) if values else ["unknown"]
    return [ALL_DATASETS_LABEL] + ordered


def _subset_choice_info(dataset_choice: str) -> Tuple[List[str], str, bool]:
    if not RUN_METADATA or dataset_choice == ALL_DATASETS_LABEL:
        return [ALL_SUBSETS_LABEL], ALL_SUBSETS_LABEL, False
    values = {
        _normalize_filter_value(meta.get("subset"))
        for meta in RUN_METADATA.values()
        if _dataset_matches(meta.get("dataset"), dataset_choice)
    }
    ordered = _sort_filter_values(values) if values else ["unknown"]
    choices = [ALL_SUBSETS_LABEL] + ordered
    return choices, ALL_SUBSETS_LABEL, True


def _method_matches(meta_value: Optional[str], selection: Optional[str]) -> bool:
    if not selection or selection == ALL_METHODS_LABEL:
        return True
    return _normalize_filter_value(meta_value) == selection


def _method_choice_info(dataset_choice: str, subset_choice: Optional[str]) -> Tuple[List[str], str, bool]:
    if not RUN_METADATA or dataset_choice == ALL_DATASETS_LABEL:
        return [ALL_METHODS_LABEL], ALL_METHODS_LABEL, False
    values = {
        _normalize_filter_value(meta.get("method"))
        for meta in RUN_METADATA.values()
        if _dataset_matches(meta.get("dataset"), dataset_choice)
        and _subset_matches(meta.get("subset"), subset_choice or ALL_SUBSETS_LABEL)
    }
    ordered = _sort_filter_values(values) if values else ["unknown"]
    choices = [ALL_METHODS_LABEL] + ordered
    interactive = len(choices) > 1
    return choices, ALL_METHODS_LABEL, interactive


def _filter_runs_for_selection(
    dataset_choice: str,
    subset_choice: Optional[str],
    method_choice: Optional[str],
    runs: List[RunInfo],
) -> List[RunInfo]:
    filtered: List[RunInfo] = []
    subset_value = subset_choice or ALL_SUBSETS_LABEL
    method_value = method_choice or ALL_METHODS_LABEL
    for label, path in runs:
        meta = RUN_METADATA.get(path, {})
        if not _dataset_matches(meta.get("dataset"), dataset_choice or ALL_DATASETS_LABEL):
            continue
        if not _subset_matches(meta.get("subset"), subset_value):
            continue
        if not _method_matches(meta.get("method"), method_value):
            continue
        filtered.append((label, path))
    return filtered


def _canonical_model_label(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return "unknown"
    if text.startswith("ft:"):
        text = text.split(":", 1)[1]
    for prefix in (
        "openai/",
        "meta-llama/",
        "mistralai/",
        "google/",
        "microsoft/",
        "Qwen/",
        "OpenGVLab/",
        "anthropic/",
        "xai/",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    replacements = {
        "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B",
        "Meta-Llama-3-8B-Instruct": "Llama-3-8B",
        "Meta-Llama-3-70B-Instruct": "Llama-3-70B",
        "Meta-Llama-Guard-3-8B": "Llama-Guard-3-8B",
        "gpt-4.1-mini-2025-04-14": "GPT-4.1-mini",
        "gpt-4.1-mini": "GPT-4.1-mini",
        "gpt-4o-mini": "GPT-4o-mini",
        "gpt-4o": "GPT-4o",
        "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    }
    for original, short in replacements.items():
        text = text.replace(original, short)
    text = text.replace("_", " ")
    return text


def _format_run_label(path: Path, source_model: str, target_model: str) -> str:
    dataset, subset, method = _infer_dataset_subset(path)
    hierarchy = " / ".join(part for part in (dataset, subset, method) if part)
    label_core = hierarchy or str(path.relative_to(RESULTS_ROOT))
    pair = path.stem
    source_short = _canonical_model_label(source_model)
    target_short = _canonical_model_label(target_model)
    label = f"{label_core} | {pair} | {source_short} → {target_short}"
    return label if len(label) <= 160 else f"{label[:157]}..."


def _is_disguise_csv(path: Path) -> bool:
    if not path.name.endswith(".csv"):
        return False
    if _has_excluded_segment(path):
        return False
    try:
        # Look for expected columns without loading the full file
        head = pd.read_csv(path, nrows=1)
        return "prompt" in head.columns and "model_response" in head.columns
    except Exception:
        return False


def _infer_model_pair_from_stem(stem: str, source_model: str, target_model: str) -> Tuple[str, str]:
    """Backfill missing source/target metadata using filename conventions."""
    inferred_source = source_model
    inferred_target = target_model
    try:
        if (not inferred_source or inferred_source in {"?", "unknown"}) and "_as_" in stem:
            parts = stem.split("_as_", 1)
            if parts and parts[0]:
                inferred_source = parts[0]
        if (not inferred_target or inferred_target in {"?", "unknown"}) and "_as_" in stem:
            parts = stem.split("_as_", 1)
            if len(parts) > 1 and parts[1]:
                inferred_target = parts[1]
        if (not inferred_source or inferred_source in {"?", "unknown"}) and "_vs_" in stem:
            parts = stem.split("_vs_", 1)
            if parts and parts[0]:
                inferred_source = parts[0]
            if len(parts) > 1 and (not inferred_target or inferred_target in {"?", "unknown"}):
                inferred_target = parts[1]
        elif (not inferred_target or inferred_target in {"?", "unknown"}) and "_vs_" in stem:
            parts = stem.split("_vs_", 1)
            if len(parts) > 1 and parts[1]:
                inferred_target = parts[1]
    except Exception:
        pass
    return inferred_source, inferred_target


@functools.lru_cache(maxsize=1)
def discover_runs() -> List[RunInfo]:
    """Return sorted (label, path) pairs for disguise runs."""
    if not RESULTS_ROOT.exists():
        return []

    runs: List[RunInfo] = []
    RUN_METADATA.clear()
    for path in RESULTS_ROOT.rglob("*.csv"):
        if not _is_disguise_csv(path):
            continue
        rel = path.relative_to(RESULTS_ROOT)
        try:
            head = pd.read_csv(path, nrows=1)
        except Exception:
            head = pd.DataFrame()

        if head.empty:
            source_model = "unknown"
            target_model = "unknown"
        else:
            source_series = head.get("source_model")
            target_series = head.get("target_model")
            source_model = str(source_series.iloc[0]) if source_series is not None and not source_series.empty else "?"
            target_model = str(target_series.iloc[0]) if target_series is not None and not target_series.empty else "?"

        source_model, target_model = _infer_model_pair_from_stem(path.stem, source_model, target_model)
        dataset, subset, method = _infer_dataset_subset(path)
        label = _format_run_label(path, source_model, target_model)
        RUN_METADATA[str(path)] = {
            "dataset": dataset,
            "subset": subset,
            "method": method,
            "source_model": source_model,
            "target_model": target_model,
        }
        runs.append((label, str(path)))

    runs.sort(key=lambda item: item[0])
    return runs


_RESPONSE_COLUMNS = [
    "model_response",
    "target_response",
    "response",
]
_PROVIDER_PREFIXES = {"openai", "azure", "anthropic", "vertex", "bedrock", "google", "meta", "hf", "huggingface"}


@functools.lru_cache(maxsize=16)
def load_csv(path: str, *, require_model_response: bool = False) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        raise ValueError(f"{path} is missing required column 'prompt'.")
    has_response_col = any(col in df.columns for col in _RESPONSE_COLUMNS)
    if not has_response_col:
        raise ValueError(f"{path} must include at least one response column ({', '.join(_RESPONSE_COLUMNS)}).")
    if require_model_response and "model_response" not in df.columns:
        raise ValueError(f"{path} is missing required column 'model_response'.")
    return df


@functools.lru_cache(maxsize=16)
def load_prompt_map(path: str, *, column: str) -> Dict[str, str]:
    df = load_csv(path)
    # comparison CSV from source_vs_target has columns model_response (A) and target_response (B)
    # single-file baselines have either model_response or target_response
    if column in df.columns:
        actual_column = column
    else:
        actual_column = "model_response" if "model_response" in df.columns else "target_response"
    grouped = df.groupby("prompt")[actual_column].first()
    return grouped.to_dict()


def _candidate_model_paths(model_name: str, dataset: Optional[str], subset: Optional[str]) -> List[Path]:
    sanitized = (model_name or "").replace("/", "_").replace(":", "_")
    if not sanitized or any(ch in sanitized for ch in "*?[]"):
        return []
    roots: List[Path] = []
    if dataset:
        subset_dirs = _subset_dir_candidates(dataset, subset)
        for rel_dir in subset_dirs:
            roots.append(BASE_RESPONSES_ROOT / dataset / rel_dir)
        default_root = BASE_RESPONSES_ROOT / dataset
        if default_root not in roots:
            roots.append(default_root)
    roots.append(BASE_RESPONSES_ROOT / "generic")
    roots.append(BASE_RESPONSES_ROOT)

    candidates: List[Path] = []
    seen: set[Path] = set()

    def _add(path: Path):
        resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    for root in roots:
        for suffix in [
            f"{sanitized}.csv",
            f"{sanitized}_responses.csv",
            f"{sanitized}_responses-1000.csv",
        ]:
            _add(root / suffix)
        if root.exists():
            for path in sorted(root.glob(f"{sanitized}*.csv")):
                _add(path)
    return candidates


def auto_locate_model_response(model_name: str, dataset: Optional[str], subset: Optional[str]) -> str:
    tried: set[str] = set()
    for candidate_name in (model_name, _normalize_model_name(model_name)):
        if not candidate_name or candidate_name in tried:
            continue
        tried.add(candidate_name)
        for candidate in _candidate_model_paths(candidate_name, dataset, subset):
            if candidate.exists():
                return str(candidate)
    return ""


def _subset_dir_candidates(dataset: Optional[str], subset: Optional[str]) -> List[Path]:
    if not dataset or not subset:
        return []
    normalized = subset.lower()
    if dataset.lower() == "gsm8k" and normalized in {"full", "500"}:
        return []
    token = re.sub(r"(?<=\D)(\d+)$", r"_\1", normalized)
    candidates: List[Path] = []
    if dataset.lower() == "gsm8k":
        candidates.append(Path("splits") / "seed42" / token)
    candidates.append(Path(token))
    unique: List[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _baseline_path_allowed(path: Path) -> bool:
    return True


def _sanitize_model_id(name: str) -> str:
    return (name or "").replace("/", "_").replace(":", "_")


def _normalize_model_name(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""
    if "/" in text:
        prefix, remainder = text.split("/", 1)
        if prefix.lower() in _PROVIDER_PREFIXES and remainder:
            return remainder
    return text


def auto_locate_score_csv(run_path: str) -> str:
    if not run_path:
        return ""
    path = Path(run_path)
    candidates = []
    method_dir = path.parent
    scores_dir = method_dir / "scores"
    candidates.append(scores_dir / path.stem / "scored.csv")
    candidates.append(scores_dir / f"{path.stem}_scored.csv")
    candidates.append(method_dir / f"{path.stem}_scored.csv")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def _alias_tokens(name: str) -> List[str]:
    normalized = _normalize_model_name(name).lower()
    tokens = re.split(r"[^a-z0-9]+", normalized)
    return [token for token in tokens if token and len(token) > 1]


def _locate_baseline_summary_path(
    dataset: Optional[str],
    subset: Optional[str],
    source_model: str,
    target_model: str,
) -> Optional[Path]:
    if not dataset or not subset:
        return None
    base_dir = RESULTS_ROOT / dataset / subset / "base"
    if not base_dir.exists():
        return None
    source_tokens = _alias_tokens(source_model)
    target_tokens = _alias_tokens(target_model)
    if not source_tokens or not target_tokens:
        return None

    candidate_summaries = list(base_dir.rglob("summary.json"))
    best_path: Optional[Path] = None
    best_score = 0
    for summary_path in candidate_summaries:
        baseline_id = summary_path.parent.name.lower()
        baseline_id = re.sub(r"_scores?$", "", baseline_id)
        src_hits = sum(1 for token in source_tokens if token in baseline_id)
        tgt_hits = sum(1 for token in target_tokens if token in baseline_id)
        if src_hits and tgt_hits:
            score = src_hits + tgt_hits
            if score > best_score:
                best_score = score
                best_path = summary_path
    return best_path


def _format_metric_line(mean: float, std: Optional[float], count: int, label: str) -> str:
    if count <= 0:
        return ""
    if std is not None and count > 1:
        stderr = std / (count ** 0.5)
        ci95 = 1.96 * stderr
        return f"- **{label}**: {mean:.4f} ± {ci95:.4f} (95% CI, n={count})"
    return f"- **{label}**: {mean:.4f} (n={count})"


SUMMARY_COLUMNS = [
    ("semantic_score", "Semantic"),
    ("stylistic_score", "Stylistic"),
    ("heuristic_match_score", "Heuristic match"),
]


def _summary_from_dataframe(df: pd.DataFrame, title: str) -> str:
    lines = [f"### {title}"]
    found = False
    for column, label in SUMMARY_COLUMNS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else None
        lines.append(_format_metric_line(mean, std, len(series), label))
        found = True
    if not found:
        lines.append("(no metrics found)")
    return "\n".join(lines)


def _summary_from_metric_dict(metrics: Dict[str, float], title: str) -> str:
    lines = [f"### {title}"]
    found = False
    for column, label in SUMMARY_COLUMNS:
        mean = metrics.get(f"{column}_mean")
        count = metrics.get(f"{column}_count")
        if mean is None or count is None or count <= 0:
            continue
        std = metrics.get(f"{column}_std")
        line = _format_metric_line(float(mean), float(std) if std is not None else None, int(count), label)
        lines.append(line)
        found = True
    if not found:
        lines.append("(no metrics found)")
    return "\n".join(lines)


def _load_baseline_summary(
    dataset: Optional[str],
    subset: Optional[str],
    source_model: str,
    target_model: str,
) -> str:
    summary_path = _locate_baseline_summary_path(dataset, subset, source_model, target_model)
    if not summary_path:
        return ""
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        label = summary_path.parent.name
        label = re.sub(r"_scores?$", "", label).replace("_", " ")
        return _summary_from_metric_dict(metrics, f"Baseline ({label})")
    except Exception as exc:
        return f"Baseline summary unavailable: {exc}"


def _sanitize_file_api_info(component: gr.File) -> None:
    """Gradio's JSON schema builder chokes when additionalProperties is bool; coerce it."""
    def _patched(schema: dict) -> dict:
        if not isinstance(schema, dict):
            return schema
        meta = schema.get("properties", {}).get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("additionalProperties"), bool):
            meta["additionalProperties"] = {"type": "string"}
        return schema

    for attr in ("api_info", "api_info_as_input", "api_info_as_output"):
        fn = getattr(component, attr, None)
        if callable(fn):
            original = fn

            def wrapper(self, *args, _original=original, **kwargs):
                return _patched(_original(*args, **kwargs))

            setattr(component, attr, types.MethodType(wrapper, component))


def _make_prompt_choices(df: pd.DataFrame, search_term: str, max_results: int = 500) -> Tuple[List[str], str]:
    series = df["prompt"].fillna("").astype(str)
    if search_term:
        mask = series.str.contains(search_term, case=False, regex=False)
    else:
        mask = pd.Series(True, index=series.index)

    matching_indices = series[mask].index.tolist()
    preview = series.loc[matching_indices].head(max_results)
    choices = [
        f"{idx} :: {text[:120].replace(os.linesep, ' ')}"
        for idx, text in preview.items()
    ]

    summary = (
        f"Found {len(matching_indices)} matching prompts."
        if search_term
        else f"Displaying {min(len(df), max_results)} prompts (use filter to narrow down)."
    )
    return choices, summary


def _parse_choice(choice_label: str) -> int:
    idx_str = choice_label.split("::", 1)[0].strip()
    return int(idx_str)


def _infer_dataset_subset(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Expected: data/results/<dataset>/<subset>/<method>/<file>.csv
    try:
        rel = path.relative_to(RESULTS_ROOT)
    except ValueError:
        return None, None, None
    parts = rel.parts
    dataset = parts[0] if len(parts) > 0 else None
    subset = parts[1] if len(parts) > 1 else None
    method = parts[2] if len(parts) > 2 else None
    return dataset, subset, method


def _is_prompt_response_csv(path: Path) -> bool:
    if not path.name.endswith(".csv"):
        return False
    if _has_excluded_segment(path):
        return False
    try:
        head = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    if "prompt" not in head.columns:
        return False
    candidate_columns = [
        "model_response",
        "target_response",
        "response",
    ]
    return any(column in head.columns for column in candidate_columns)


@functools.lru_cache(maxsize=1)
def discover_baseline_csvs() -> List[RunInfo]:
    """Return (label, path) pairs for CSVs that can provide baseline responses."""
    choices: List[RunInfo] = []
    for prefix, root in BASELINE_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if not _is_prompt_response_csv(path):
                continue
            if not _baseline_path_allowed(path):
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = Path(path.name)
            rel_parts = rel.parts[-3:] if len(rel.parts) > 3 else rel.parts
            short_rel = Path(*rel_parts)
            label = f"{prefix}: {short_rel}"
            choices.append((label, str(path)))

    seen_paths: set[str] = set()
    unique_choices: List[RunInfo] = []
    for label, path in choices:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_choices.append((label, path))
    return unique_choices


def load_run(selected_path: str, custom_path: str) -> Tuple[
    Dict[str, object],
    object,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]:
    path = (custom_path or "").strip() or selected_path
    if not path:
        raise gr.Error("Select a run or enter a custom CSV path.")

    df = load_csv(path, require_model_response=True)
    dataset, subset, method = _infer_dataset_subset(Path(path))
    stem = Path(path).stem
    source_model = df.get("source_model", pd.Series(["?"])).iloc[0]
    target_model = df.get("target_model", pd.Series(["?"])).iloc[0]
    source_model, target_model = _infer_model_pair_from_stem(stem, source_model, target_model)

    choices, summary = _make_prompt_choices(df, search_term="")
    if not choices:
        raise gr.Error("No prompts found in this CSV.")

    # Attempt to infer a system prompt/template from common columns
    sys_prompt = ""
    for col in ("system_prompt", "prompt_template", "template", "completion_template"):
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val and val != "nan":
                sys_prompt = val
                break

    metadata_lines = [
        f"**File:** `{path}`",
        f"**Dataset:** {dataset or 'unknown'}",
        f"**Subset:** {subset or 'unknown'}",
        f"**Method:** {method or df.get('method', pd.Series(['unknown'])).iloc[0]}",
        f"**Source model:** {_canonical_model_label(source_model)}",
        f"**Target model:** {_canonical_model_label(target_model)}",
        f"**Total rows:** {len(df)}",
    ]

    is_baseline = method == "base" or "_vs_" in stem

    # Suggest baselines from matching dataset/subset directories
    source_suggestion = auto_locate_model_response(source_model, dataset, subset)
    target_suggestion = auto_locate_model_response(target_model, dataset, subset)
    if is_baseline:
        source_suggestion = ""

    state_payload = {
        "path": path,
        "records": df.to_dict("records"),
        "dataset": dataset,
        "subset": subset,
        "method": method,
        "source_model": source_model,
        "target_model": target_model,
        "system_prompt": sys_prompt,
        "is_baseline": is_baseline,
    }

    metadata = "\n".join(metadata_lines)
    first_choice = choices[0]
    prompt_text, disguised_text, target_text, info_text = display_example(
        state_payload,
        first_choice,
        target_suggestion,
    )
    source_text = ""

    score_suggestion = auto_locate_score_csv(path)

    return (
        state_payload,
        gr.update(choices=choices, value=choices[0]),
        summary,
        metadata,
        sys_prompt,
        source_suggestion,
        target_suggestion,
        prompt_text,
        disguised_text,
        target_text,
        info_text,
        source_text,
        first_choice,
        score_suggestion,
    )


def filter_prompts(
    state: Dict[str, object],
    term: str,
    target_path: str,
    show_source: bool,
    source_path: str,
) -> Tuple[gr.Dropdown, str, str, str, str, str, str]:
    if not state:
        raise gr.Error("Load a run first.")
    df = pd.DataFrame(state["records"])
    choices, summary = _make_prompt_choices(df, term)
    if not choices:
        raise gr.Error("No prompts match this filter.")
    first_choice = choices[0]
    prompt_text, disguised_text, target_text, info_text = display_example(state, first_choice, target_path)
    effective_show = _should_show_source_column(state, show_source)
    source_text = display_source(state, first_choice, effective_show, source_path) if effective_show else ""
    source_title_update, source_box_update = _source_outputs(state, effective_show, source_text)
    return (
        gr.update(choices=choices, value=first_choice),
        summary,
        prompt_text,
        disguised_text,
        target_text,
        info_text,
        source_title_update,
        source_box_update,
    )


def _fetch_baseline(prompt: str, path: str, column: str) -> str:
    if not path:
        return ""
    try:
        mapping = load_prompt_map(path, column=column)
        return mapping.get(prompt, "")
    except Exception:
        return ""


def display_example(
    state: Dict[str, object],
    selected_choice: str,
    target_path: str,
) -> Tuple[str, str, str, str]:
    if not state:
        raise gr.Error("Load a run first.")
    idx = _parse_choice(selected_choice)
    df = pd.DataFrame(state["records"])
    if idx < 0 or idx >= len(df):
        raise gr.Error(f"Index {idx} is out of bounds.")

    row = df.iloc[idx]
    prompt_text = row.get("prompt", "")
    disguised_text = row.get("model_response", "")
    target_text = row.get("target_response", "")

    if not target_text:
        target_text = _fetch_baseline(prompt_text, target_path, column="model_response")

    info_lines = [
        f"**Row:** {idx}",
        f"**Method:** {row.get('method', state.get('method', ''))}",
        f"**Source model:** {_canonical_model_label(row.get('source_model', state.get('source_model', '')))}",
        f"**Target model:** {_canonical_model_label(row.get('target_model', state.get('target_model', '')))}",
    ]

    return prompt_text, disguised_text, target_text, "\n".join(info_lines)


def display_source(
    state: Dict[str, object],
    selected_choice: str,
    show_source: bool,
    source_path: str,
) -> str:
    if not show_source:
        return ""
    if not state:
        return ""
    idx = _parse_choice(selected_choice)
    df = pd.DataFrame(state["records"])
    if idx < 0 or idx >= len(df):
        raise gr.Error(f"Index {idx} is out of bounds.")
    prompt_text = df.iloc[idx].get("prompt", "")
    return _fetch_baseline(prompt_text, source_path, column="model_response")


def _should_show_source_column(state: Optional[Dict[str, object]], user_toggle: bool) -> bool:
    if not state:
        return False
    if state.get("is_baseline"):
        return False
    return bool(user_toggle)


def _source_outputs(state: Optional[Dict[str, object]], show_source: bool, source_text: str) -> Tuple[gr.Update, gr.Update]:
    if not state:
        return (
            gr.update(value="**Source baseline**", visible=False),
            gr.update(value="", visible=False),
        )
    if state.get("is_baseline"):
        return (
            gr.update(value="**Source baseline (n/a for comparison runs)**", visible=False),
            gr.update(value="", visible=False),
        )
    effective = bool(show_source)
    label = _canonical_model_label(str(state.get("source_model", "?")))
    return (
        gr.update(value=f"**Source ({label})**", visible=effective),
        gr.update(value=source_text if effective else "", visible=effective),
    )


def build_interface() -> gr.Blocks:
    runs = discover_runs()
    runs_map = {label: path for label, path in runs}
    baseline_choices = discover_baseline_csvs()
    baseline_map = {label: path for label, path in baseline_choices}
    baseline_inverse_map = {str(Path(path).resolve()): label for label, path in baseline_choices}
    dataset_choices = _dataset_choices()
    dataset_default = dataset_choices[0] if dataset_choices else ALL_DATASETS_LABEL
    subset_options, subset_default, subset_enabled = _subset_choice_info(dataset_default)
    method_options, method_default, method_enabled = _method_choice_info(dataset_default, subset_default)
    initial_filtered_runs = _filter_runs_for_selection(dataset_default, subset_default, method_default, runs)
    initial_run_choices = [label for label, _ in initial_filtered_runs]
    initial_run_value = initial_run_choices[0] if initial_run_choices else None
    initial_run_path = initial_filtered_runs[0][1] if initial_filtered_runs else ""

    def _path_to_label(path: str) -> Optional[str]:
        if not path:
            return None
        resolved = str(Path(path).resolve())
        return baseline_inverse_map.get(resolved)

    # Minimal, clean theme + light CSS polish
    try:
        theme = gr.themes.Soft(
            primary_hue="slate",
            neutral_hue="slate",
            radius_scale=1.0,
            spacing_scale=1.0,
        ).set(
            body_text_size="13px",
            block_title_text_size="16px",
        )
    except Exception:
        theme = None  # Fallback if themes API changes

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Baskervville:ital,wght@0,400;0,700;1,400&display=swap');
    
    :root { --text: #0f172a; --muted: #64748b; --border: #e5e7eb; }
    
    .gradio-container { max-width: 1080px; margin: 0 auto; font-family: 'Baskervville', Baskerville, Georgia, serif; }
    .section-subtle { color: var(--muted); margin-top: -6px; font-size: 15px; }
    .mono textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; line-height: 1.55; }
    .tight > * { margin-top: 6px !important; margin-bottom: 6px !important; }
    .right-note { text-align: right; color: #9ca3af; font-size: 12px; }
    
    /* Card-like textboxes */
    .gr-textbox { border-radius: 10px !important; }
    .gr-textbox textarea { padding: 14px 14px; }
    
    /* Compact labels */
    label.svelte-1ipelgc, label { font-size: 12px !important; color: #475569 !important; }
    
    /* Reduce vertical whitespace */
    .gradio-container .block.padded { padding-top: 10px !important; padding-bottom: 10px !important; }
    """

    with gr.Blocks(title="Dementor Disguise Viewer", theme=theme, css=css) as demo:
        gr.Markdown("# Dementor Disguise Inspector")
        gr.Markdown(
            "A minimal viewer for disguise runs. Pick a run, filter prompts, and compare outputs.",
            elem_classes=["section-subtle"],
        )

        state = gr.State({})

        with gr.Row(elem_classes=["tight"]):
            dataset_dropdown = gr.Dropdown(
                label="Dataset",
                choices=dataset_choices,
                value=dataset_default,
                interactive=len(dataset_choices) > 1,
            )
            subset_dropdown = gr.Dropdown(
                label="Subset",
                choices=subset_options,
                value=subset_default,
                interactive=subset_enabled,
            )
            method_dropdown = gr.Dropdown(
                label="Method",
                choices=method_options,
                value=method_default,
                interactive=method_enabled,
            )

        with gr.Row(elem_classes=["tight"]):
            run_dropdown = gr.Dropdown(
                label="Run",
                choices=initial_run_choices,
                value=initial_run_value,
            )
            status = gr.Markdown(elem_classes=["right-note"])

        metadata = gr.Markdown(visible=False)
        system_prompt_box = gr.Textbox(label="System prompt / template", lines=3, interactive=False, elem_classes=["mono"])
        with gr.Row():
            score_summary = gr.Markdown()
            baseline_score_summary = gr.Markdown()
        row_score = gr.Markdown()

        search_term = gr.Textbox(
            label="Filter prompts",
            placeholder="Type to filter by substring and press Enter",
        )

        prompt_dropdown = gr.Dropdown(label="Prompts", choices=[], value=None)

        prompt_box = gr.Textbox(label="Prompt", lines=5, interactive=False, elem_classes=["mono"])

        with gr.Row():
            with gr.Column(scale=1):
                disguised_title = gr.Markdown("**Disguised (model)**")
                disguised_box = gr.Textbox(
                    label=None,
                    lines=14,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["mono"],
                )
            with gr.Column(scale=1):
                target_title = gr.Markdown("**Target (reference)**")
                target_box = gr.Textbox(
                    label=None,
                    lines=14,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["mono"],
                )
            with gr.Column(scale=1):
                source_title = gr.Markdown("**Source baseline**", visible=False)
                source_box = gr.Textbox(
                    label=None,
                    lines=14,
                    interactive=False,
                    visible=False,
                    show_copy_button=True,
                    elem_classes=["mono"],
                )
        info_box = gr.Markdown()

        with gr.Accordion("Advanced options", open=False):
            custom_path = gr.Textbox(
                label="Load custom CSV",
                placeholder="Absolute or relative path to a disguise CSV",
                show_copy_button=True,
            )
            custom_file = gr.File(label="Browse CSV file", file_types=[".csv"], type="filepath")
            _sanitize_file_api_info(custom_file)
            load_custom_btn = gr.Button("Load custom CSV", variant="secondary")
            target_csv_dropdown = gr.Dropdown(
                label="Select target CSV",
                choices=[label for label, _ in baseline_choices],
                value=None,
                allow_custom_value=True,
            )
            target_file = gr.File(label="Browse target baseline", file_types=[".csv"], type="filepath")
            _sanitize_file_api_info(target_file)
            target_path = gr.Textbox(label="Target baseline CSV (B)", show_copy_button=True)
            source_csv_dropdown = gr.Dropdown(
                label="Select source CSV",
                choices=[label for label, _ in baseline_choices],
                value=None,
                allow_custom_value=True,
            )
            source_file = gr.File(label="Browse source baseline", file_types=[".csv"], type="filepath")
            _sanitize_file_api_info(source_file)
            source_path = gr.Textbox(label="Source baseline CSV (A)", show_copy_button=True)
            show_source = gr.Checkbox(label="Show source baseline column", value=True)
            score_path = gr.Textbox(label="Override scored.csv path", show_copy_button=True)

        def _load_scores(path: str):
            if not path:
                return {}, ""
            try:
                sdf = pd.read_csv(path)
                # Build maps by prompt
                by_prompt = {}
                for _, r in sdf.iterrows():
                    p = str(r.get("prompt", ""))
                    if p:
                        by_prompt[p] = r.to_dict()
                summary = _summary_from_dataframe(sdf, "Disguised scores")
                # Add supplemental metrics not in the main list
                extra_cols = [
                    c
                    for c in sdf.columns
                    if sdf[c].dtype.kind in ("i", "u", "f")
                    and all(base not in c for base, _ in SUMMARY_COLUMNS)
                    and any(x in c.lower() for x in ("win", "score", "similar", "pref", "acc"))
                ]
                if extra_cols:
                    extras = sdf[extra_cols].mean().to_dict()
                    extra_lines = [f"- **{k}**: {v:.4f}" for k, v in extras.items()]
                    summary = f"{summary}\n\n#### Additional metrics\n" + "\n".join(extra_lines)
                return by_prompt, summary
            except Exception as e:
                return {}, f"Failed to load scores: {e}"

        def _render_row_score(score_map, records, idx: Optional[int]):
            if not score_map or idx is None or not records:
                return ""
            df = pd.DataFrame(records)
            if idx < 0 or idx >= len(df):
                return ""
            prompt = df.iloc[idx].get("prompt", "")
            row = score_map.get(prompt)
            if not row:
                return ""
            keys = [k for k in row.keys() if any(x in k.lower() for x in ("win", "score", "similar", "pref", "acc"))]
            view = "\n".join([f"- **{k}**: {row[k]}" for k in sorted(set(keys))])
            return f"### Row scores\n{view}" if view else ""

        with gr.Accordion("Run metadata", open=False):
            meta_area = gr.Markdown()

        def _load_from_path(path: str, show_src: bool):
            if not path:
                raise gr.Error("Select a run or provide a CSV path.")
            result = load_run(path, "")
            (
                new_state,
                prompt_update,
                summary,
                meta_text,
                sys_prompt,
                source_suggestion,
                target_suggestion,
                prompt_text,
                disguised_text,
                target_text,
                info_text,
                source_text,
                first_choice,
                score_suggestion,
            ) = result
            # Friendly hint when source baseline is missing
            hint = "" if source_suggestion else "(No source baseline found; you can provide one under Advanced options.)"
            status_text = f"{summary} {hint}".strip()
            effective_show = _should_show_source_column(new_state, show_src)
            source_text = (
                display_source(new_state, first_choice, True, source_suggestion) if effective_show else ""
            )
            source_title_update, source_box_update = _source_outputs(new_state, effective_show, source_text)
            source_dropdown_update = gr.update(value=_path_to_label(source_suggestion))
            target_dropdown_update = gr.update(value=_path_to_label(target_suggestion))
            # Update dynamic titles with model names
            src = new_state.get("source_model", "?")
            tgt = new_state.get("target_model", "?")
            src_label = _canonical_model_label(str(src))
            tgt_label = _canonical_model_label(str(tgt))
            is_baseline = bool(new_state.get("is_baseline"))
            if is_baseline:
                disguised_label = f"Source ({src_label})"
            else:
                disguised_label = f"Disguised ({src_label} → {tgt_label})"
            target_label = f"Target ({tgt_label})"
            score_map, score_summary_md = _load_scores(score_suggestion)
            first_idx = _parse_choice(first_choice) if first_choice else None
            row_score_text = _render_row_score(score_map, new_state.get("records"), first_idx)
            baseline_summary_md = _load_baseline_summary(
                new_state.get("dataset"),
                new_state.get("subset"),
                new_state.get("source_model", ""),
                new_state.get("target_model", ""),
            )

            return (
                new_state,
                prompt_update,
                status_text,
                meta_text,
                gr.update(value=sys_prompt or "", visible=bool(sys_prompt)),
                source_suggestion,
                target_suggestion,
                prompt_text,
                gr.update(value=f"**{disguised_label}**"),
                gr.update(value=disguised_text),
                gr.update(value=f"**{target_label}**"),
                gr.update(value=target_text),
                info_text,
                source_title_update,
                source_box_update,
                source_dropdown_update,
                target_dropdown_update,
                score_suggestion,
                score_summary_md,
                baseline_summary_md,
                row_score_text,
                score_map,
            )

        def _load_uploaded_run(file_path: Optional[str], show_src: bool):
            if not file_path:
                raise gr.Error("Upload a CSV file first.")
            outputs = _load_from_path(file_path, show_src)
            return (*outputs, gr.update(value=file_path))

        def _update_example(state_value, choice, target_csv, show_src, source_csv):
            if not state_value or not choice:
                return (
                    "",
                    "",
                    "",
                    "",
                    gr.update(value="**Source baseline**", visible=False),
                    gr.update(value="" if show_src else "", visible=False),
                )
            prompt_text, disguised_text, target_text, info_text = display_example(state_value, choice, target_csv)
            effective_show = _should_show_source_column(state_value, show_src)
            source_text = display_source(state_value, choice, effective_show, source_csv)
            source_title_update, source_box_update = _source_outputs(state_value, effective_show, source_text)
            return prompt_text, disguised_text, target_text, info_text, source_title_update, source_box_update

        def _update_meta(meta_text: str):
            return gr.update(value=meta_text, visible=True)

        def _apply_target_choice(state_value, choice, selected_label, show_src, source_csv_path):
            path = baseline_map.get(selected_label, "") if selected_label else ""
            prompt_text, disguised_text, target_text, info_text, source_title_update, source_box_update = _update_example(
                state_value,
                choice,
                path,
                show_src,
                source_csv_path,
            )
            return path, prompt_text, disguised_text, target_text, info_text, source_title_update, source_box_update

        def _apply_source_choice(state_value, choice, selected_label, show_src):
            path = baseline_map.get(selected_label, "") if selected_label else ""
            effective_show = _should_show_source_column(state_value, show_src)
            source_text = display_source(state_value, choice, effective_show, path)
            title_update, box_update = _source_outputs(state_value, effective_show, source_text)
            return path, title_update, box_update

        def _apply_target_file(state_value, choice, file_path, show_src, source_csv_path):
            if not file_path:
                title_update, box_update = _source_outputs(state_value, False, "")
                return "", "", "", "", "", title_update, box_update
            prompt_text, disguised_text, target_text, info_text, source_title_update, source_box_update = _update_example(
                state_value,
                choice,
                file_path,
                show_src,
                source_csv_path,
            )
            return file_path, prompt_text, disguised_text, target_text, info_text, source_title_update, source_box_update

        def _apply_source_file(state_value, choice, file_path, show_src):
            if not file_path:
                title_update, box_update = _source_outputs(state_value, False, "")
                return "", title_update, box_update
            effective_show = _should_show_source_column(state_value, show_src)
            source_text = display_source(state_value, choice, effective_show, file_path)
            title_update, box_update = _source_outputs(state_value, effective_show, source_text)
            return file_path, title_update, box_update

        # Scores wiring: load on path, update per-row
        def _on_score_path(path):
            score_map, summary_md = _load_scores(path)
            return score_map, summary_md

        score_state = gr.State({})
        score_path.change(fn=_on_score_path, inputs=[score_path], outputs=[score_state, score_summary])

        def _update_row_score(score_map, s, choice):
            if not score_map or not s or not choice:
                return ""
            idx = _parse_choice(choice)
            return _render_row_score(score_map, s.get("records"), idx)

        prompt_dropdown.change(
            fn=_update_row_score,
            inputs=[score_state, state, prompt_dropdown],
            outputs=[row_score],
        )

        load_custom_btn.click(
            fn=lambda path, show_src: _load_from_path(path, show_src),
            inputs=[custom_path, show_source],
            outputs=[
                state,
                prompt_dropdown,
                status,
                metadata,
                system_prompt_box,
                source_path,
                target_path,
                prompt_box,
                disguised_title,
                disguised_box,
                target_title,
                target_box,
                info_box,
                source_title,
                source_box,
                source_csv_dropdown,
                target_csv_dropdown,
                score_path,
                score_summary,
                baseline_score_summary,
                row_score,
                score_state,
            ],
        )

        custom_file.change(
            fn=lambda file_obj, show_src: _load_uploaded_run(file_obj, show_src),
            inputs=[custom_file, show_source],
            outputs=[
                state,
                prompt_dropdown,
                status,
                metadata,
                system_prompt_box,
                source_path,
                target_path,
                prompt_box,
                disguised_title,
                disguised_box,
                target_title,
                target_box,
                info_box,
                source_title,
                source_box,
                source_csv_dropdown,
                target_csv_dropdown,
                score_path,
                score_summary,
                baseline_score_summary,
                row_score,
                score_state,
                custom_path,
            ],
        )

        search_term.submit(
            fn=lambda s, term, tgt_path, show_src, src_path: filter_prompts(s, term, tgt_path, show_src, src_path),
            inputs=[state, search_term, target_path, show_source, source_path],
            outputs=[prompt_dropdown, status, prompt_box, disguised_box, target_box, info_box, source_title, source_box],
        )

        prompt_dropdown.change(
            fn=lambda s, choice, tgt_path, show_src, src_path: _update_example(s, choice, tgt_path, show_src, src_path),
            inputs=[state, prompt_dropdown, target_path, show_source, source_path],
            outputs=[prompt_box, disguised_box, target_box, info_box, source_title, source_box],
        )

        target_csv_dropdown.change(
            fn=lambda s, choice, label, show_src, src_path: _apply_target_choice(s, choice, label, show_src, src_path),
            inputs=[state, prompt_dropdown, target_csv_dropdown, show_source, source_path],
            outputs=[target_path, prompt_box, disguised_box, target_box, info_box, source_title, source_box],
        )

        target_file.change(
            fn=lambda s, choice, file_obj, show_src, src_path: _apply_target_file(s, choice, file_obj, show_src, src_path),
            inputs=[state, prompt_dropdown, target_file, show_source, source_path],
            outputs=[target_path, prompt_box, disguised_box, target_box, info_box, source_title, source_box],
        )

        source_csv_dropdown.change(
            fn=lambda s, choice, label, show_src: _apply_source_choice(s, choice, label, show_src),
            inputs=[state, prompt_dropdown, source_csv_dropdown, show_source],
            outputs=[source_path, source_title, source_box],
        )

        source_file.change(
            fn=lambda s, choice, file_obj, show_src: _apply_source_file(s, choice, file_obj, show_src),
            inputs=[state, prompt_dropdown, source_file, show_source],
            outputs=[source_path, source_title, source_box],
        )

        def _toggle_source(state_value, choice, show_src, src_path):
            effective = _should_show_source_column(state_value, show_src)
            source_text = display_source(state_value, choice, effective, src_path) if effective else ""
            return _source_outputs(state_value, effective, source_text)

        show_source.change(
            fn=_toggle_source,
            inputs=[state, prompt_dropdown, show_source, source_path],
            outputs=[source_title, source_box],
        )

        def _update_run_dropdown(filtered_runs: List[RunInfo], current_label: Optional[str]) -> Tuple[List[str], str]:
            if not filtered_runs:
                raise gr.Error("No runs match these filters.")
            labels = [label for label, _ in filtered_runs]
            value = current_label if current_label in labels else labels[0]
            return labels, value

        def _run_outputs_list():
            return [
                state,
                prompt_dropdown,
                status,
                metadata,
                system_prompt_box,
                source_path,
                target_path,
                prompt_box,
                disguised_title,
                disguised_box,
                target_title,
                target_box,
                info_box,
                source_title,
                source_box,
                source_csv_dropdown,
                target_csv_dropdown,
                score_path,
                score_summary,
                baseline_score_summary,
                row_score,
                score_state,
            ]

        run_outputs = _run_outputs_list()

        def _load_filtered_run(path: str, show_src: bool):
            if not path:
                raise gr.Error("Select a run or provide a CSV path.")
            return _load_from_path(path, show_src)

        def _apply_dataset_filter(selected_dataset: str, current_subset: Optional[str], current_method: Optional[str], show_src: bool):
            dataset_value = selected_dataset or ALL_DATASETS_LABEL
            subset_choices_list, default_subset, subset_interactive = _subset_choice_info(dataset_value)
            if subset_interactive and current_subset in subset_choices_list:
                subset_value = current_subset or default_subset
            else:
                subset_value = default_subset
            method_choices_list, default_method, method_interactive = _method_choice_info(dataset_value, subset_value)
            if method_interactive and current_method in method_choices_list:
                method_value = current_method or default_method
            else:
                method_value = default_method
            filtered = _filter_runs_for_selection(dataset_value, subset_value, method_value, runs)
            run_choices, run_value = _update_run_dropdown(filtered, None)
            path = runs_map.get(run_value, "") or filtered[0][1]
            outputs = _load_filtered_run(path, show_src)
            return (
                gr.update(choices=subset_choices_list, value=subset_value, interactive=subset_interactive),
                gr.update(choices=method_choices_list, value=method_value, interactive=method_interactive),
                gr.update(choices=run_choices, value=run_value),
                *outputs,
            )

        def _apply_subset_filter(selected_dataset: str, subset_value: Optional[str], current_method: Optional[str], show_src: bool):
            dataset_value = selected_dataset or ALL_DATASETS_LABEL
            method_choices_list, default_method, method_interactive = _method_choice_info(dataset_value, subset_value)
            if method_interactive and current_method in method_choices_list:
                method_value = current_method or default_method
            else:
                method_value = default_method
            filtered = _filter_runs_for_selection(dataset_value, subset_value, method_value, runs)
            run_choices, run_value = _update_run_dropdown(filtered, None)
            path = runs_map.get(run_value, "") or filtered[0][1]
            outputs = _load_filtered_run(path, show_src)
            return (
                gr.update(choices=method_choices_list, value=method_value, interactive=method_interactive),
                gr.update(choices=run_choices, value=run_value),
                *outputs,
            )

        def _apply_method_filter(selected_dataset: str, subset_value: Optional[str], method_value: Optional[str], show_src: bool):
            dataset_value = selected_dataset or ALL_DATASETS_LABEL
            filtered = _filter_runs_for_selection(dataset_value, subset_value, method_value, runs)
            run_choices, run_value = _update_run_dropdown(filtered, None)
            path = runs_map.get(run_value, "") or filtered[0][1]
            outputs = _load_filtered_run(path, show_src)
            return (
                gr.update(choices=run_choices, value=run_value),
                *outputs,
            )

        dataset_dropdown.change(
            fn=_apply_dataset_filter,
            inputs=[dataset_dropdown, subset_dropdown, method_dropdown, show_source],
            outputs=[subset_dropdown, method_dropdown, run_dropdown, *run_outputs],
        ).then(
            fn=_update_meta,
            inputs=[metadata],
            outputs=[meta_area],
        )

        subset_dropdown.change(
            fn=_apply_subset_filter,
            inputs=[dataset_dropdown, subset_dropdown, method_dropdown, show_source],
            outputs=[method_dropdown, run_dropdown, *run_outputs],
        ).then(
            fn=_update_meta,
            inputs=[metadata],
            outputs=[meta_area],
        )

        method_dropdown.change(
            fn=_apply_method_filter,
            inputs=[dataset_dropdown, subset_dropdown, method_dropdown, show_source],
            outputs=[run_dropdown, *run_outputs],
        ).then(
            fn=_update_meta,
            inputs=[metadata],
            outputs=[meta_area],
        )

        run_dropdown.change(
            fn=lambda label, show_src: _load_from_path(runs_map.get(label, ""), show_src),
            inputs=[run_dropdown, show_source],
            outputs=run_outputs,
        ).then(
            fn=_update_meta,
            inputs=[metadata],
            outputs=[meta_area],
        )

        if initial_run_path:
            demo.load(
                fn=lambda show_src: _load_from_path(initial_run_path, show_src),
                inputs=[show_source],
                outputs=run_outputs,
            ).then(
                fn=_update_meta,
                inputs=[metadata],
                outputs=[meta_area],
            )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    port_env = os.environ.get("GRADIO_SERVER_PORT")
    if port_env:
        server_port = int(port_env)
    else:
        import socket

        def _find_port(start: int = 7860, end: int = 7900) -> int:
            for port in range(start, end + 1):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        s.bind(("", port))
                        return port
                    except OSError:
                        continue
            raise OSError(f"Cannot find empty port in range: {start}-{end}.")

        server_port = _find_port()

    interface.launch(server_name=server_name, server_port=server_port, share=share)
