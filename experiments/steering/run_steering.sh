#!/usr/bin/env bash
# =============================================================================
# run_steering.sh — one-command launcher for a SINGLE steering RDO cone on any
# H100 box (our shared box, a partner's dedicated 4-card box, a future cluster).
#
# After `git clone` + `git checkout ethan` + venv setup, and with this box's
# weights reachable (see .env.example / experiments/steering/README.md), this
# script runs the full per-model steering cone for one roster slug:
#   1. run_rdo_model.py <slug>     — benign+fingerprint reuse, DIM, RDO cone
#                                    train, dim select, AdvBench cone verdict.
#   2. run_benchmark_eval.py <slug> — the standard-5 dissociation benchmarks.
# Both stages are RESUMABLE (each skips any artifact already on disk), so the
# script is idempotent — safe to re-run.
#
# This is what the partner uses for nemotron-super-120b on 4 cards:
#   DEMENTOR_GPUS=0,1,2,3 DEMENTOR_MODELS_DIR=/weights \
#     bash experiments/steering/run_steering.sh nemotron-super-120b
#
# Usage:
#   bash experiments/steering/run_steering.sh <slug>            # run the cone
#   bash experiments/steering/run_steering.sh <slug> --dry-run  # plan + resolved config
#   bash experiments/steering/run_steering.sh --help
#
# Key env overrides (all optional; defaults reproduce our shared box):
#   DEMENTOR_GPUS        GPU ids this cone may use (default 5,6,7). >1 id -> model-parallel.
#   DEMENTOR_MODELS_DIR  local weights root; per-slug subdir <dir>/<slug> (default /data/ethantsliu/models_dl)
#   DEMENTOR_HF_HOME     HF cache root (default /data/ethantsliu/huggingface)
#   DEMENTOR_DATA        data/outputs root (default <repo>/data)
#   DEMENTOR_STEER_WORK  where per-model <slug>/ outputs land (default our-box repl80_rdo)
#   DEMENTOR_PY          python interpreter (default: `python` on PATH)
#   RDO_MAX_DIM/RDO_BETAS/RDO_OUT_SUFFIX  cone-strength / sibling-dir overrides (see run_rdo_model.py)
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# 0. Parse args (one positional <slug> + flags)
# --------------------------------------------------------------------------- #
DRY_RUN=0
SLUG=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'
      exit 0 ;;
    -*)
      echo "ERROR: unknown flag '$arg' (try --help)" >&2; exit 2 ;;
    *)
      if [ -n "$SLUG" ]; then echo "ERROR: only one <slug> allowed (got '$SLUG' and '$arg')" >&2; exit 2; fi
      SLUG="$arg" ;;
  esac
done

log()  { echo "[run_steering] $*"; }
warn() { echo "[run_steering][WARN] $*" >&2; }
die()  { echo "[run_steering][FATAL] $*" >&2; exit 1; }

[ -n "$SLUG" ] || die "no model slug given (try --help; run_steering.sh <slug> [--dry-run])"

# --------------------------------------------------------------------------- #
# 1. Locate repo + package, pick interpreter, set GPU pool
# --------------------------------------------------------------------------- #
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PKG="$(dirname "$SCRIPT_PATH")"                          # experiments/steering
DEMENTOR_REPO="${DEMENTOR_REPO:-$(cd "$PKG/../.." && pwd)}"   # auto-detect repo root
export DEMENTOR_REPO
[ -f "$PKG/run_rdo_model.py" ] || die "package layout wrong: $PKG/run_rdo_model.py missing"

PY="${DEMENTOR_PY:-python}"
command -v "$PY" >/dev/null 2>&1 || die "python interpreter '$PY' not found (set DEMENTOR_PY)"
export DEMENTOR_PY="$PY"

export DEMENTOR_GPUS="${DEMENTOR_GPUS:-5,6,7}"
# Model-parallel (device_map=auto) whenever more than one GPU is offered to this cone.
NGPU="$(echo "$DEMENTOR_GPUS" | tr ',' '\n' | grep -c '[0-9]')"
if [ "$NGPU" -gt 1 ]; then export DEMENTOR_MP=1; else unset DEMENTOR_MP || true; fi
export CUDA_VISIBLE_DEVICES="$DEMENTOR_GPUS"

# Load repo-root .env credentials so the child judges/graders inherit them (env already-set wins).
if [ -f "$DEMENTOR_REPO/.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|\#*) continue ;; esac
    [ "${line#*=}" = "$line" ] && continue
    k="${line%%=*}"; v="${line#*=}"
    k="$(echo "$k" | tr -d '[:space:]')"
    v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
    [ -n "$k" ] || continue
    if [ -z "$(printenv "$k" 2>/dev/null || true)" ]; then export "$k=$v"; fi
  done < "$DEMENTOR_REPO/.env"
fi

# --------------------------------------------------------------------------- #
# 2. Resolve + print config (imports steer_config: same resolution the scripts use)
# --------------------------------------------------------------------------- #
log "repo   = $DEMENTOR_REPO"
log "python = $PY  ($("$PY" --version 2>&1))"
log "GPUs   = $DEMENTOR_GPUS  (MP=${DEMENTOR_MP:-0}, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
log "slug   = $SLUG"
[ "$DRY_RUN" = 1 ] && log "MODE   = DRY-RUN (resolve config + print plan; launch nothing)"

# steer_config resolves paths/worklist/seed exactly as the python stages do. A hard failure here
# (unknown slug, unfindable worklist) aborts before we burn a GPU.
if ! "$PY" - "$SLUG" <<'PYEOF'
import os, sys
sys.path.insert(0, os.path.join(os.environ["DEMENTOR_REPO"], "experiments", "steering"))
import steer_config as C
slug = sys.argv[1]
wl = C.load_worklist()
print("  steer_config: REPO=%s" % C.REPO)
print("  steer_config: WORK_ROOT=%s" % C.WORK_ROOT)
print("  steer_config: MODELS_DIR=%s" % C.MODELS_DIR)
print("  steer_config: HF_HOME=%s" % C.HF_HOME)
print("  steer_config: GPUS=%s  SEED=%s" % (C.GPUS, C.SEED))
print("  steer_config: worklist=%s (%d models)" % (C.worklist_path(), len(wl["models"])))
spec = next((m for m in wl["models"] if m["slug"] == slug), None)
if spec is None:
    sys.stderr.write("unknown slug %r -- not in the rdo_worklist.json roster\n" % slug)
    sys.exit(3)
print("  resolved model path: %s  (needs_mp=%s, params_b=%s)"
      % (spec["path"], bool(spec.get("needs_mp")), spec.get("params_b")))
PYEOF
then
  die "config preflight failed for slug '$SLUG' (see error above)"
fi

# --------------------------------------------------------------------------- #
# 3. Run the two stages (resumable; each skips artifacts already present)
# --------------------------------------------------------------------------- #
run_stage() {
  local name="$1"; shift
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] would run $name:  $*"
    return 0
  fi
  log "=== $name ==="
  "$@"
}

run_stage "run_rdo_model.py $SLUG"     "$PY" "$PKG/run_rdo_model.py" "$SLUG"
run_stage "run_benchmark_eval.py $SLUG" "$PY" "$PKG/run_benchmark_eval.py" "$SLUG"

if [ "$DRY_RUN" = 1 ]; then
  log "dry-run complete — config resolved, no GPU used."
else
  log "DONE $SLUG — cone verdict in \$DEMENTOR_STEER_WORK/$SLUG/eval/metrics.json,"
  log "     dissociation table in \$DEMENTOR_STEER_WORK/$SLUG/benchmarks_summary.json"
fi
