from pathlib import Path
import subprocess

files = sorted(Path('data/results/gsm8k/eval200').glob('*/*.csv'))
for csv_path in files:
    pair_id = csv_path.stem
    out_dir = csv_path.parent / 'scores' / pair_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scoring {csv_path} -> {out_dir}")
    subprocess.run(
        [
            'python', 'scripts/scorer.py', 'pairwise',
            '--input', str(csv_path),
            '--output', str(out_dir / 'scored.csv'),
            '--judge-model', 'openai/gpt-4.1-mini',
        ],
        check=True,
    )
