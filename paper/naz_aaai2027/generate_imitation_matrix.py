#!/usr/bin/env python3
"""Compatibility entry point for strict, stage-aware campaign regeneration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGENERATE = REPO / "experiments" / "imitation_safety" / "regenerate_campaign.py"

raise SystemExit(subprocess.call([sys.executable, str(REGENERATE), *sys.argv[1:]], cwd=REPO))
