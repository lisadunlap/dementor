#!/usr/bin/env python3
"""
Migrate legacy results layout to the new structure.

From (old):
  data/results/<dataset>/comparisons/disguised_vs_target/<method>/[<run>/]{pair}.csv
  and scored variants in metrics/, metrics_<pair>/, or <run>/metrics/

To (new):
  Raw disguised:
    data/results/<dataset>/<method>/{pair}.csv
  Disguised scores:
    data/results/<dataset>/<method>/scores/{pair}/scored.csv
    data/results/<dataset>/<method>/scores/{pair}/scored_metrics.csv

Base scores (single-file) remain under:
  data/results/<dataset>/scores/<model>/scored.csv

Usage:
  python scripts/migrate_results_layout.py --dataset gsm8k [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
import re


def is_raw_disguised_csv(p: Path) -> bool:
    name = p.name
    if not name.endswith('.csv'):
        return False
    if name.endswith('_scored.csv') or name.endswith('_scores.csv'):
        return False
    if name.endswith('_metrics.csv') or 'metrics' in name:
        return False
    # Heuristic: consider files containing "_as_" as disguised pairs
    return '_as_' in name


def find_legacy_disguised_dirs(dataset: str) -> list[Path]:
    root = Path('data') / 'results' / dataset / 'comparisons' / 'disguised_vs_target'
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def migrate_dataset(dataset: str, dry_run: bool = False) -> list[str]:
    logs: list[str] = []
    legacy_methods = find_legacy_disguised_dirs(dataset)
    if not legacy_methods:
        logs.append(f"No legacy disguised_vs_target dirs found under dataset '{dataset}'.")

    # Phase 1: migrate legacy comparisons/disguised_vs_target
    for method_dir in legacy_methods:
        method = method_dir.name
        new_method_dir = Path('data') / 'results' / dataset / method
        new_scores_root = new_method_dir / 'scores'
        if not dry_run:
            new_scores_root.mkdir(parents=True, exist_ok=True)
            new_method_dir.mkdir(parents=True, exist_ok=True)

        # Walk old structure (may include runs, metrics folders)
        for path in method_dir.rglob('*.csv'):
            name = path.name
            # Raw disguised CSV
            if is_raw_disguised_csv(path):
                dest = new_method_dir / name
                if dest.exists() and dest.read_bytes() == path.read_bytes():
                    logs.append(f"SKIP raw (identical): {path} -> {dest}")
                else:
                    # Avoid overwrite by suffixing if different
                    final = dest
                    if dest.exists() and dest.read_bytes() != path.read_bytes():
                        base = dest.stem
                        suffix = dest.suffix
                        k = 2
                        while True:
                            cand = new_method_dir / f"{base}_dup{k}{suffix}"
                            if not cand.exists():
                                final = cand
                                break
                            k += 1
                    logs.append(f"MOVE raw: {path} -> {final}")
                    if not dry_run:
                        shutil.move(str(path), str(final))
                continue

            # Scored CSV files
            if name.endswith('_scores.csv') or name.endswith('_scored.csv'):
                # Normalize to per-pair directory
                # Extract pair id from filename
                pair_match = re.search(r"(.+?_as_.+?)(?:_scores|_scored)\.csv$", name)
                if not pair_match:
                    logs.append(f"WARN: could not parse pair id from {path}")
                    continue
                pair_id = pair_match.group(1)
                dest_dir = new_scores_root / pair_id
                dest = dest_dir / 'scored.csv'
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.read_bytes() == path.read_bytes():
                    logs.append(f"SKIP scored (identical): {path} -> {dest}")
                else:
                    logs.append(f"MOVE scored: {path} -> {dest}")
                    if not dry_run:
                        shutil.move(str(path), str(dest))
                continue

            # Metrics CSV
            if name.endswith('_metrics.csv'):
                pair_match = re.search(r"(.+?_as_.+?)(?:_scores|_scored)?_metrics\.csv$", name)
                if not pair_match:
                    logs.append(f"WARN: could not parse pair id from metrics {path}")
                    continue
                pair_id = pair_match.group(1)
                dest_dir = new_scores_root / pair_id
                dest = dest_dir / 'scored_metrics.csv'
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.read_bytes() == path.read_bytes():
                    logs.append(f"SKIP metrics (identical): {path} -> {dest}")
                else:
                    logs.append(f"MOVE metrics: {path} -> {dest}")
                    if not dry_run:
                        shutil.move(str(path), str(dest))
                continue

        # Attempt to remove empty legacy directories
        if not dry_run:
            try:
                # Remove empty dirs bottom-up
                for p in sorted(method_dir.rglob('*'), key=lambda x: len(str(x)), reverse=True):
                    if p.is_dir():
                        try:
                            p.rmdir()
                        except OSError:
                            pass
                method_dir.rmdir()
                logs.append(f"REMOVED empty legacy dir: {method_dir}")
            except OSError:
                logs.append(f"LEFT legacy dir (not empty): {method_dir}")

    # Phase 2: normalize existing per-method dirs that still have scores at root or metrics/ subdir
    per_method_root = Path('data') / 'results' / dataset
    if per_method_root.exists():
        for method_dir in per_method_root.iterdir():
            if not method_dir.is_dir() or method_dir.name in ("scores", "comparisons", "visualization"):
                continue
            method = method_dir.name
            new_scores_root = method_dir / 'scores'
            if not dry_run:
                new_scores_root.mkdir(parents=True, exist_ok=True)

            # Root-level scored artifacts
            for p in list(method_dir.glob("*_scored.csv")) + list(method_dir.glob("*_scores.csv")):
                name = p.name
                m = re.search(r"(.+?_as_.+?)_(?:scored|scores)\.csv$", name)
                if not m:
                    continue
                pair_id = m.group(1)
                dest_dir = new_scores_root / pair_id
                dest = dest_dir / 'scored.csv'
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.read_bytes() == p.read_bytes():
                    logs.append(f"SKIP scored(root) identical: {p} -> {dest}")
                else:
                    logs.append(f"MOVE scored(root): {p} -> {dest}")
                    if not dry_run:
                        shutil.move(str(p), str(dest))
            for p in list(method_dir.glob("*_scored_metrics.csv")) + list(method_dir.glob("*_scores_metrics.csv")):
                name = p.name
                m = re.search(r"(.+?_as_.+?)_(?:scored|scores)_metrics\.csv$", name)
                if not m:
                    continue
                pair_id = m.group(1)
                dest_dir = new_scores_root / pair_id
                dest = dest_dir / 'scored_metrics.csv'
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.read_bytes() == p.read_bytes():
                    logs.append(f"SKIP metrics(root) identical: {p} -> {dest}")
                else:
                    logs.append(f"MOVE metrics(root): {p} -> {dest}")
                    if not dry_run:
                        shutil.move(str(p), str(dest))

            # metrics/ subdir normalization
            metrics_dir = method_dir / 'metrics'
            if metrics_dir.exists():
                for p in metrics_dir.glob("*_scores.csv"):
                    m = re.search(r"(.+?_as_.+?)_scores\.csv$", p.name)
                    if not m:
                        continue
                    pair_id = m.group(1)
                    dest_dir = new_scores_root / pair_id
                    dest = dest_dir / 'scored.csv'
                    if not dry_run:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and dest.read_bytes() == p.read_bytes():
                        logs.append(f"SKIP scored(metrics) identical: {p} -> {dest}")
                    else:
                        logs.append(f"MOVE scored(metrics): {p} -> {dest}")
                        if not dry_run:
                            shutil.move(str(p), str(dest))
                for p in metrics_dir.glob("*_scores_metrics.csv"):
                    m = re.search(r"(.+?_as_.+?)_scores_metrics\.csv$", p.name)
                    if not m:
                        continue
                    pair_id = m.group(1)
                    dest_dir = new_scores_root / pair_id
                    dest = dest_dir / 'scored_metrics.csv'
                    if not dry_run:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and dest.read_bytes() == p.read_bytes():
                        logs.append(f"SKIP metrics(metrics) identical: {p} -> {dest}")
                    else:
                        logs.append(f"MOVE metrics(metrics): {p} -> {dest}")
                        if not dry_run:
                            shutil.move(str(p), str(dest))
                # Move summaries (json) if present
                for p in metrics_dir.glob("*_summary.json"):
                    # Try to place as summary.json under per-pair dir
                    m = re.search(r"(.+?_as_.+?)_summary\.json$", p.name)
                    if not m:
                        continue
                    pair_id = m.group(1)
                    dest_dir = new_scores_root / pair_id
                    dest = dest_dir / 'summary.json'
                    if not dry_run:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                    logs.append(f"MOVE summary: {p} -> {dest}")
                    if not dry_run:
                        shutil.move(str(p), str(dest))
                # Attempt to remove empty metrics dir
                try:
                    if not dry_run:
                        metrics_dir.rmdir()
                        logs.append(f"REMOVED empty metrics dir: {metrics_dir}")
                except OSError:
                    pass

            # run* subdirs (e.g., run2) – move raw and metrics similarly
            for run_dir in method_dir.glob('run*'):
                if not run_dir.is_dir():
                    continue
                # Raw pairs inside run dir
                for p in run_dir.glob("*_as_*.csv"):
                    if p.name.endswith('_scores.csv') or p.name.endswith('_scored.csv') or p.name.endswith('_metrics.csv'):
                        continue
                    dest = method_dir / p.name
                    if dest.exists() and dest.read_bytes() == p.read_bytes():
                        logs.append(f"SKIP raw(run) identical: {p} -> {dest}")
                    else:
                        logs.append(f"MOVE raw(run): {p} -> {dest}")
                        if not dry_run:
                            shutil.move(str(p), str(dest))
                # runX/metrics
                rmetrics = run_dir / 'metrics'
                if rmetrics.exists():
                    for p in rmetrics.glob("*_scores.csv"):
                        m = re.search(r"(.+?_as_.+?)_scores\.csv$", p.name)
                        if not m:
                            continue
                        pair_id = m.group(1)
                        dest_dir = new_scores_root / pair_id
                        dest = dest_dir / 'scored.csv'
                        if not dry_run:
                            dest_dir.mkdir(parents=True, exist_ok=True)
                        logs.append(f"MOVE scored(run): {p} -> {dest}")
                        if not dry_run:
                            shutil.move(str(p), str(dest))
                    for p in rmetrics.glob("*_scores_metrics.csv"):
                        m = re.search(r"(.+?_as_.+?)_scores_metrics\.csv$", p.name)
                        if not m:
                            continue
                        pair_id = m.group(1)
                        dest_dir = new_scores_root / pair_id
                        dest = dest_dir / 'scored_metrics.csv'
                        if not dry_run:
                            dest_dir.mkdir(parents=True, exist_ok=True)
                        logs.append(f"MOVE metrics(run): {p} -> {dest}")
                        if not dry_run:
                            shutil.move(str(p), str(dest))
                    for p in rmetrics.glob("*_summary.json"):
                        m = re.search(r"(.+?_as_.+?)_summary\.json$", p.name)
                        if not m:
                            continue
                        pair_id = m.group(1)
                        dest_dir = new_scores_root / pair_id
                        dest = dest_dir / 'summary.json'
                        if not dry_run:
                            dest_dir.mkdir(parents=True, exist_ok=True)
                        logs.append(f"MOVE summary(run): {p} -> {dest}")
                        if not dry_run:
                            shutil.move(str(p), str(dest))
                # Try to remove run dir if empty
                try:
                    if not dry_run:
                        for q in sorted(run_dir.rglob('*'), key=lambda x: len(str(x)), reverse=True):
                            if q.is_dir():
                                try:
                                    q.rmdir()
                                except OSError:
                                    pass
                        run_dir.rmdir()
                        logs.append(f"REMOVED empty run dir: {run_dir}")
                except OSError:
                    pass

    return logs


def main():
    ap = argparse.ArgumentParser(description='Migrate legacy results layout to new per-method directories')
    ap.add_argument('--dataset', default='gsm8k')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    logs = migrate_dataset(args.dataset, dry_run=args.dry_run)
    for line in logs:
        print(line)


if __name__ == '__main__':
    main()
