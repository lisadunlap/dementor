"""Build human-eval CSVs for GSM8K stylometric comparisons."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


def _extract_method(base: str) -> str:
    stem = base.split("_as_", 1)[0]
    for marker in ("_openai", "_meta-llama"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem.split("_")[0]


def _direction_from_source(source: str) -> str:
    if "gpt-4.1-mini" in source:
        return "GPT -> Llama"
    if "Meta-Llama-3.1-8B-Instruct" in source:
        return "Llama -> GPT"
    return "unknown"


def _shuffle_pair(row: Dict[str, object], rng: random.Random) -> Dict[str, object]:
    if rng.random() < 0.5:
        return row
    return {
        **row,
        "candidate_a": row["candidate_b"],
        "candidate_b": row["candidate_a"],
        "method_a": row["method_b"],
        "method_b": row["method_a"],
    }


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(cleaned) <= max_chars:
        return cleaned
    head = max_chars // 2
    tail = max_chars - head - 1
    return f"{cleaned[:head].rstrip()}…{cleaned[-tail:].lstrip()}"


def _over_limit(value: object, max_chars: int) -> bool:
    if max_chars is None:
        return False
    text = "" if value is None else str(value)
    return len(text) > max_chars


def _bucket_key(row: pd.Series, baseline_method: str) -> str:
    if row["pair_type"] == "self_control":
        return str(row["direction"])
    method = row["method_a"] if row["method_a"] != baseline_method else row["method_b"]
    return f"{method}|{row['direction']}"


def _balanced_sample(
    df: pd.DataFrame,
    baseline_method: str,
    total: int,
    *,
    rng: random.Random,
) -> pd.DataFrame:
    buckets: Dict[str, pd.DataFrame] = {}
    for key, group in df.groupby(df.apply(_bucket_key, axis=1, baseline_method=baseline_method)):
        buckets[key] = group

    bucket_keys = sorted(buckets.keys())
    bucket_count = len(bucket_keys)
    if bucket_count == 0:
        raise ValueError("No buckets available for balanced sampling.")

    base = total // bucket_count
    remainder = total % bucket_count

    rng.shuffle(bucket_keys)
    target_sizes: Dict[str, int] = {key: base for key in bucket_keys}
    for key in bucket_keys[:remainder]:
        target_sizes[key] += 1

    selected_frames: List[pd.DataFrame] = []
    selected_idx: set[int] = set()
    deficit = 0
    capacity_keys = []
    for key in bucket_keys:
        group = buckets[key]
        target = target_sizes[key]
        if len(group) <= target:
            selected_frames.append(group)
            selected_idx.update(group.index.tolist())
            deficit += target - len(group)
        else:
            sampled = group.sample(n=target, random_state=rng.randint(0, 1_000_000))
            selected_frames.append(sampled)
            selected_idx.update(sampled.index.tolist())
            capacity_keys.append(key)

    if deficit > 0 and capacity_keys:
        rng.shuffle(capacity_keys)
        for key in capacity_keys:
            if deficit <= 0:
                break
            group = buckets[key]
            extra = group.drop(index=group.index.intersection(selected_idx))
            extra_capacity = len(extra)
            if extra_capacity <= 0:
                continue
            take = min(extra_capacity, deficit)
            extra_sample = extra.sample(n=take, random_state=rng.randint(0, 1_000_000))
            selected_frames.append(extra_sample)
            selected_idx.update(extra_sample.index.tolist())
            deficit -= take

    result = pd.concat(selected_frames).reset_index(drop=True)
    if len(result) > total:
        result = result.sample(n=total, random_state=rng.randint(0, 1_000_000)).reset_index(drop=True)
    return result


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    return df


def _build_baseline_pairs(
    input_dir: Path,
    baseline_method: str,
    *,
    rng: random.Random,
    max_per_direction: int | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    baseline_paths = sorted(input_dir.glob(f"{baseline_method}_*.csv"))
    baseline_by_direction: Dict[str, pd.DataFrame] = {}
    for path in baseline_paths:
        df = _load_csv(path)
        direction = _direction_from_source(str(df["source_model"].iloc[0]))
        baseline_by_direction[direction] = df

    for path in sorted(input_dir.glob("*.csv")):
        if path.name.startswith("self_"):
            continue
        method = _extract_method(path.name)
        if method == baseline_method:
            continue
        df = _load_csv(path)
        direction = _direction_from_source(str(df["source_model"].iloc[0]))
        baseline_df = baseline_by_direction.get(direction)
        if baseline_df is None:
            continue

        merged = df.merge(
            baseline_df[["prompt", "model_response"]].rename(columns={"model_response": "baseline_response"}),
            on="prompt",
            how="inner",
        )
        rows_iter = merged.itertuples(index=False)
        if max_per_direction is not None and len(merged) > max_per_direction:
            sample_idx = rng.sample(range(len(merged)), max_per_direction)
            rows_iter = (merged.iloc[i] for i in sample_idx)
        for row in rows_iter:
            entry = {
                "prompt": row.prompt,
                "target_response": getattr(row, "target_response", ""),
                "candidate_a": row.model_response,
                "candidate_b": row.baseline_response,
                "method_a": method,
                "method_b": baseline_method,
                "direction": direction,
                "pair_type": "baseline_vs_method",
            }
            rows.append(_shuffle_pair(entry, rng))

    return rows


def _build_self_controls(
    input_dir: Path,
    *,
    rng: random.Random,
    max_per_direction: int | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, label in [("gpt", "GPT self"), ("llama", "Llama self")]:
        run1 = input_dir / f"self_{key}_run1.csv"
        run2 = input_dir / f"self_{key}_run2.csv"
        if not run1.exists() or not run2.exists():
            continue
        df1 = _load_csv(run1)
        df2 = _load_csv(run2)
        merged = df1.merge(
            df2[["prompt", "model_response"]].rename(columns={"model_response": "model_response_2"}),
            on="prompt",
            how="inner",
        )
        rows_iter = merged.itertuples(index=False)
        if max_per_direction is not None and len(merged) > max_per_direction:
            sample_idx = rng.sample(range(len(merged)), max_per_direction)
            rows_iter = (merged.iloc[i] for i in sample_idx)
        for row in rows_iter:
            entry = {
                "prompt": row.prompt,
                "target_response": row.model_response,
                "candidate_a": row.model_response,
                "candidate_b": row.model_response_2,
                "method_a": f"self_{key}_run1",
                "method_b": f"self_{key}_run2",
                "direction": label,
                "pair_type": "self_control",
            }
            rows.append(_shuffle_pair(entry, rng))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build human-eval CSVs against a baseline method.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/results/gsm8k/eval200/stylometric_probs_ensemble"),
        help="Directory containing stylometric_probs_ensemble CSVs.",
    )
    parser.add_argument(
        "--baseline",
        default="random_sampling",
        help="Baseline method name to compare against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/gsm8k/eval200/human_eval/gsm8k_eval200_baseline_pairs_public.csv"),
        help="Output CSV path for annotators (no method labels).",
    )
    parser.add_argument(
        "--output-key",
        type=Path,
        default=Path("data/results/gsm8k/eval200/human_eval/gsm8k_eval200_baseline_pairs_key.csv"),
        help="Output CSV path with method labels for analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for candidate order.",
    )
    parser.add_argument(
        "--include-self-controls",
        action="store_true",
        help="Include self-control pairs (run1 vs run2).",
    )
    parser.add_argument(
        "--max-per-direction",
        type=int,
        default=None,
        help="Max rows per method-direction pair (and per self-control) to balance counts.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle row order for annotators.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Max characters for prompt/target/candidates in the public outputs.",
    )
    parser.add_argument(
        "--max-chars-mode",
        choices=["truncate", "filter"],
        default="truncate",
        help="Whether to truncate long fields or drop rows that exceed --max-chars.",
    )
    parser.add_argument(
        "--total-comparisons",
        type=int,
        default=None,
        help="Target total comparisons with balanced sampling across method/direction/self buckets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_dir = args.input_dir.resolve()

    rows = _build_baseline_pairs(
        input_dir,
        args.baseline,
        rng=rng,
        max_per_direction=args.max_per_direction,
    )
    if args.include_self_controls:
        rows.extend(_build_self_controls(input_dir, rng=rng, max_per_direction=args.max_per_direction))

    if not rows:
        raise ValueError("No rows generated. Check input directory and baseline name.")

    df = pd.DataFrame(rows)
    df.insert(0, "comparison_id", [f"cmp_{i:06d}" for i in range(len(df))])
    df["winner"] = ""

    if args.max_chars is not None and args.max_chars_mode == "filter":
        mask = ~(
            df["prompt"].apply(lambda x: _over_limit(x, args.max_chars))
            | df["target_response"].apply(lambda x: _over_limit(x, args.max_chars))
            | df["candidate_a"].apply(lambda x: _over_limit(x, args.max_chars))
            | df["candidate_b"].apply(lambda x: _over_limit(x, args.max_chars))
        )
        df = df[mask].reset_index(drop=True)
        df["comparison_id"] = [f"cmp_{i:06d}" for i in range(len(df))]

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.total_comparisons is not None:
        df = _balanced_sample(df, args.baseline, args.total_comparisons, rng=rng)
        df["comparison_id"] = [f"cmp_{i:06d}" for i in range(len(df))]

    public_cols = [
        "comparison_id",
        "prompt",
        "target_response",
        "candidate_a",
        "candidate_b",
        "winner",
    ]
    public_df = df[public_cols].copy()
    key_cols = [
        "comparison_id",
        "method_a",
        "method_b",
        "direction",
        "pair_type",
    ]
    key_df = df[key_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_key.parent.mkdir(parents=True, exist_ok=True)
    if args.max_chars is not None and args.max_chars_mode == "truncate":
        for col in ["prompt", "target_response", "candidate_a", "candidate_b"]:
            public_df[col] = public_df[col].astype(str).apply(lambda x: _truncate(x, args.max_chars))
    public_df.to_csv(args.output, index=False)
    key_df.to_csv(args.output_key, index=False)
    public_xlsx = args.output.with_suffix(".xlsx")
    cleaned = public_df.copy()
    for col in ["prompt", "target_response", "candidate_a", "candidate_b"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.replace(ILLEGAL_CHARACTERS_RE, "", regex=True)
    with pd.ExcelWriter(public_xlsx, engine="openpyxl") as writer:
        cleaned.to_excel(writer, index=False, sheet_name="annotations")
    print(f"Wrote {len(cleaned)} rows to {public_xlsx}")
    print(f"Wrote {len(public_df)} rows to {args.output}")
    print(f"Wrote {len(key_df)} rows to {args.output_key}")


if __name__ == "__main__":
    main()
