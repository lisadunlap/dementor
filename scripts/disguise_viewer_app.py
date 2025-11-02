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
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

RESULTS_ROOT = Path("data/results")
BASE_RESPONSES_ROOT = Path("data/model-responses")
BASELINE_SEARCH_ROOTS: List[Tuple[str, Path]] = [
    ("results", RESULTS_ROOT),
    ("model-responses", BASE_RESPONSES_ROOT),
]
_EXCLUDED_DIR_NAMES = {"scores", "metrics", "judgments"}


class RunInfo(Tuple[str, str]):
    """Typed alias for (label, path) pairs."""


def _is_disguise_csv(path: Path) -> bool:
    if not path.name.endswith(".csv"):
        return False
    parts = set(path.parts)
    if _EXCLUDED_DIR_NAMES.intersection(parts):
        return False
    try:
        # Look for expected columns without loading the full file
        head = pd.read_csv(path, nrows=1)
        return "prompt" in head.columns and "model_response" in head.columns
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def discover_runs() -> List[RunInfo]:
    """Return sorted (label, path) pairs for disguise runs."""
    if not RESULTS_ROOT.exists():
        return []

    runs: List[RunInfo] = []
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
        label = f"{rel} | {source_model} → {target_model}"
        runs.append((label, str(path)))

    runs.sort(key=lambda item: item[0])
    return runs


@functools.lru_cache(maxsize=16)
def load_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "prompt" not in df.columns or "model_response" not in df.columns:
        raise ValueError(f"{path} is missing required columns ('prompt', 'model_response').")
    return df


@functools.lru_cache(maxsize=16)
def load_prompt_map(path: str, *, column: str) -> Dict[str, str]:
    df = load_csv(path)
    # comparison CSV from source_vs_target has columns model_response (A) and target_response (B)
    # single-file baselines have either model_response or target_response
    if column in df.columns:
        actual_column = column
    elif column == "target_response" and "model_response_b" in df.columns:
        actual_column = "model_response_b"
    elif column == "model_response" and "model_response_a" in df.columns:
        actual_column = "model_response_a"
    else:
        actual_column = "model_response" if "model_response" in df.columns else "target_response"
    grouped = df.groupby("prompt")[actual_column].first()
    return grouped.to_dict()


def _candidate_model_paths(model_name: str, dataset: Optional[str], subset: Optional[str]) -> List[Path]:
    sanitized = (model_name or "").replace("/", "_").replace(":", "_")
    roots: List[Path] = []
    if dataset:
        if subset:
            roots.append(BASE_RESPONSES_ROOT / dataset / subset)
        roots.append(BASE_RESPONSES_ROOT / dataset / "full")
        roots.append(BASE_RESPONSES_ROOT / dataset / "500")
    roots.append(BASE_RESPONSES_ROOT / "generic")
    roots.append(BASE_RESPONSES_ROOT)

    candidates: List[Path] = []
    for root in roots:
        for suffix in [
            f"{sanitized}.csv",
            f"{sanitized}_responses.csv",
            f"{sanitized}_responses-1000.csv",
        ]:
            candidate = root / suffix
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def auto_locate_model_response(model_name: str, dataset: Optional[str], subset: Optional[str]) -> str:
    for candidate in _candidate_model_paths(model_name, dataset, subset):
        if candidate.exists():
            return str(candidate)
    return ""


def _sanitize_model_id(name: str) -> str:
    return (name or "").replace("/", "_").replace(":", "_")


def auto_locate_compare_csv(source_model: str, target_model: str, dataset: Optional[str], subset: Optional[str]) -> str:
    """Suggest path for source_vs_target comparison CSV based on run metadata."""
    if not dataset or not subset:
        return ""
    src = _sanitize_model_id(source_model)
    tgt = _sanitize_model_id(target_model)
    path = Path("data") / "results" / dataset / subset / "comparisons" / "source_vs_target" / f"{src}_vs_{tgt}.csv"
    return str(path) if path.exists() else ""


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
    if _EXCLUDED_DIR_NAMES.intersection(path.parts):
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
        "model_response_a",
        "model_response_b",
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
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = Path(path.name)
            label = f"{prefix}: {rel}"
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
]:
    path = (custom_path or "").strip() or selected_path
    if not path:
        raise gr.Error("Select a run or enter a custom CSV path.")

    df = load_csv(path)
    dataset, subset, method = _infer_dataset_subset(Path(path))
    source_model = df.get("source_model", pd.Series(["?"])).iloc[0]
    target_model = df.get("target_model", pd.Series(["?"])).iloc[0]

    choices, summary = _make_prompt_choices(df, search_term="")
    if not choices:
        raise gr.Error("No prompts found in this CSV.")

    metadata_lines = [
        f"**File:** `{path}`",
        f"**Dataset:** {dataset or 'unknown'}",
        f"**Subset:** {subset or 'unknown'}",
        f"**Method:** {method or df.get('method', pd.Series(['unknown'])).iloc[0]}",
        f"**Source model:** {source_model}",
        f"**Target model:** {target_model}",
        f"**Total rows:** {len(df)}",
    ]

    # Prefer pairwise compare CSV when available (single file contains both source and target baselines)
    compare_suggestion = auto_locate_compare_csv(source_model, target_model, dataset, subset)
    source_suggestion = compare_suggestion or auto_locate_model_response(source_model, dataset, subset)
    target_suggestion = compare_suggestion or auto_locate_model_response(target_model, dataset, subset)

    state_payload = {
        "path": path,
        "records": df.to_dict("records"),
        "dataset": dataset,
        "subset": subset,
        "method": method,
        "source_model": source_model,
        "target_model": target_model,
    }

    metadata = "\n".join(metadata_lines)
    first_choice = choices[0]
    prompt_text, disguised_text, target_text, info_text = display_example(
        state_payload,
        first_choice,
        target_suggestion,
    )
    source_text = ""

    return (
        state_payload,
        gr.update(choices=choices, value=choices[0]),
        summary,
        metadata,
        source_suggestion,
        target_suggestion,
        prompt_text,
        disguised_text,
        target_text,
        info_text,
        source_text,
        first_choice,
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
    source_text = display_source(state, first_choice, show_source, source_path) if show_source else ""
    source_update = gr.update(value=source_text, visible=show_source)
    return (
        gr.update(choices=choices, value=first_choice),
        summary,
        prompt_text,
        disguised_text,
        target_text,
        info_text,
        source_update,
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
        f"**Source model:** {row.get('source_model', state.get('source_model', ''))}",
        f"**Target model:** {row.get('target_model', state.get('target_model', ''))}",
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


def build_interface() -> gr.Blocks:
    runs = discover_runs()
    runs_map = {label: path for label, path in runs}
    baseline_choices = discover_baseline_csvs()
    baseline_map = {label: path for label, path in baseline_choices}
    baseline_inverse_map = {str(Path(path).resolve()): label for label, path in baseline_choices}

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
            run_dropdown = gr.Dropdown(
                label="Run",
                choices=[label for label, _ in runs],
                value=runs[0][0] if runs else None,
            )
            status = gr.Markdown(elem_classes=["right-note"])

        metadata = gr.Markdown(visible=False)

        search_term = gr.Textbox(
            label="Filter prompts",
            placeholder="Type to filter by substring and press Enter",
        )

        prompt_dropdown = gr.Dropdown(label="Prompts", choices=[], value=None)

        prompt_box = gr.Textbox(label="Prompt", lines=5, interactive=False, elem_classes=["mono"])

        with gr.Row():
            disguised_box = gr.Textbox(
                label="Disguised (model)",
                lines=14,
                interactive=False,
                show_copy_button=True,
                elem_classes=["mono"],
                scale=1,
            )
            target_box = gr.Textbox(
                label="Target (reference)",
                lines=14,
                interactive=False,
                show_copy_button=True,
                elem_classes=["mono"],
                scale=1,
            )
            source_box = gr.Textbox(
                label="Source baseline",
                lines=14,
                interactive=False,
                visible=False,
                show_copy_button=True,
                elem_classes=["mono"],
                scale=1,
            )
        info_box = gr.Markdown()

        with gr.Accordion("Advanced options", open=False):
            custom_path = gr.Textbox(
                label="Load custom CSV",
                placeholder="Absolute or relative path to a disguise CSV",
            )
            load_custom_btn = gr.Button("Load custom CSV", variant="secondary")
            target_csv_dropdown = gr.Dropdown(
                label="Select target CSV",
                choices=[label for label, _ in baseline_choices],
                value=None,
            )
            target_path = gr.Textbox(label="Target baseline CSV or compare CSV (B)")
            source_csv_dropdown = gr.Dropdown(
                label="Select source CSV",
                choices=[label for label, _ in baseline_choices],
                value=None,
            )
            source_path = gr.Textbox(label="Source baseline CSV or compare CSV (A)")
            show_source = gr.Checkbox(label="Show source baseline column", value=True)

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
                source_suggestion,
                target_suggestion,
                prompt_text,
                disguised_text,
                target_text,
                info_text,
                source_text,
                first_choice,
            ) = result
            # Friendly hint when source baseline is missing
            hint = "" if source_suggestion else "(No source baseline found; you can provide one under Advanced options.)"
            status_text = f"{summary} {hint}".strip()
            if show_src:
                source_text = display_source(new_state, first_choice, True, source_suggestion)
            source_update = gr.update(value=source_text if show_src else "", visible=show_src)
            source_dropdown_update = gr.update(value=_path_to_label(source_suggestion))
            target_dropdown_update = gr.update(value=_path_to_label(target_suggestion))
            return (
                new_state,
                prompt_update,
                status_text,
                meta_text,
                source_suggestion,
                target_suggestion,
                prompt_text,
                disguised_text,
                target_text,
                info_text,
                source_update,
                source_dropdown_update,
                target_dropdown_update,
            )

        def _update_example(state_value, choice, target_csv, show_src, source_csv):
            if not state_value or not choice:
                return "", "", "", "", gr.update(value="" if show_src else "", visible=show_src)
            prompt_text, disguised_text, target_text, info_text = display_example(state_value, choice, target_csv)
            source_text = display_source(state_value, choice, show_src, source_csv)
            source_update = gr.update(value=source_text if show_src else "", visible=show_src)
            return prompt_text, disguised_text, target_text, info_text, source_update

        def _update_meta(meta_text: str):
            return gr.update(value=meta_text, visible=True)

        def _apply_target_choice(state_value, choice, selected_label, show_src, source_csv_path):
            path = baseline_map.get(selected_label, "") if selected_label else ""
            prompt_text, disguised_text, target_text, info_text, source_update = _update_example(
                state_value,
                choice,
                path,
                show_src,
                source_csv_path,
            )
            return path, prompt_text, disguised_text, target_text, info_text, source_update

        def _apply_source_choice(state_value, choice, selected_label, show_src):
            path = baseline_map.get(selected_label, "") if selected_label else ""
            source_text = display_source(state_value, choice, show_src, path)
            return path, gr.update(value=source_text if show_src else "", visible=show_src)

        run_dropdown.change(
            fn=lambda label, show_src: _load_from_path(runs_map.get(label, ""), show_src),
            inputs=[run_dropdown, show_source],
            outputs=[
                state,
                prompt_dropdown,
                status,
                metadata,
                source_path,
                target_path,
                prompt_box,
                disguised_box,
                target_box,
                info_box,
                source_box,
                source_csv_dropdown,
                target_csv_dropdown,
            ],
        ).then(
            fn=_update_meta,
            inputs=[metadata],
            outputs=[meta_area],
        )

        load_custom_btn.click(
            fn=lambda path, show_src: _load_from_path(path, show_src),
            inputs=[custom_path, show_source],
            outputs=[
                state,
                prompt_dropdown,
                status,
                metadata,
                source_path,
                target_path,
                prompt_box,
                disguised_box,
                target_box,
                info_box,
                source_box,
                source_csv_dropdown,
                target_csv_dropdown,
            ],
        )

        search_term.submit(
            fn=lambda s, term, tgt_path, show_src, src_path: filter_prompts(s, term, tgt_path, show_src, src_path),
            inputs=[state, search_term, target_path, show_source, source_path],
            outputs=[prompt_dropdown, status, prompt_box, disguised_box, target_box, info_box, source_box],
        )

        prompt_dropdown.change(
            fn=lambda s, choice, tgt_path, show_src, src_path: _update_example(s, choice, tgt_path, show_src, src_path),
            inputs=[state, prompt_dropdown, target_path, show_source, source_path],
            outputs=[prompt_box, disguised_box, target_box, info_box, source_box],
        )

        target_csv_dropdown.change(
            fn=lambda s, choice, label, show_src, src_path: _apply_target_choice(s, choice, label, show_src, src_path),
            inputs=[state, prompt_dropdown, target_csv_dropdown, show_source, source_path],
            outputs=[target_path, prompt_box, disguised_box, target_box, info_box, source_box],
        )

        source_csv_dropdown.change(
            fn=lambda s, choice, label, show_src: _apply_source_choice(s, choice, label, show_src),
            inputs=[state, prompt_dropdown, source_csv_dropdown, show_source],
            outputs=[source_path, source_box],
        )

        show_source.change(
            fn=lambda s, choice, show_src, src_path: gr.update(
                value=display_source(s, choice, show_src, src_path) if show_src else "",
                visible=show_src,
            ),
            inputs=[state, prompt_dropdown, show_source, source_path],
            outputs=[source_box],
        )

        if runs:
            demo.load(
                fn=lambda show_src: _load_from_path(runs[0][1], show_src),
                inputs=[show_source],
                outputs=[
                    state,
                    prompt_dropdown,
                    status,
                    metadata,
                    source_path,
                    target_path,
                    prompt_box,
                    disguised_box,
                    target_box,
                    info_box,
                    source_box,
                    source_csv_dropdown,
                    target_csv_dropdown,
                ],
            ).then(
                fn=_update_meta,
                inputs=[metadata],
                outputs=[meta_area],
            )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
