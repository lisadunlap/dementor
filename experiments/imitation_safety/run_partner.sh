#!/usr/bin/env bash
# =============================================================================
# run_partner.sh — one-command launcher for the Dementor IMITATION eval on a
# fresh, DEDICATED 4xH100 box (the partner's machine — NOT our shared box).
#
# After `git clone` + `git checkout ethan` + `git lfs pull` + venv setup
# (see experiments/imitation_safety/PARTNER_SETUP.md §2-§3), this single script:
#   1. Preflights credentials + tooling.
#   2. Prefetches the HF judge/grader models + Tinker source tokenizers.
#   3. Pulls the current local chatbot_arena seed42 PEFT adapters from HF and
#      points the daemons' registry at them.
#   4. Launches the whole imitation-eval daemon set (setsid nohup, logs/).
#   5. Is idempotent + resumable — safe to re-run; daemons skip done work.
#
# Usage:
#   bash experiments/imitation_safety/run_partner.sh            # full run
#   bash experiments/imitation_safety/run_partner.sh --dry-run  # show plan only
#   bash experiments/imitation_safety/run_partner.sh --skip-prefetch
#   bash experiments/imitation_safety/run_partner.sh --skip-local-adapters
#
# Key env overrides (all optional except the credentials in §1):
#   DEMENTOR_GPUS           GPU pool (default 0,1,2,3)
#   DEMENTOR_PY             python interpreter (default: `python` on PATH)
#   DEMENTOR_HF_HOME        HF cache root (recommend a big-disk path; exported as HF_HOME)
#   DEMENTOR_DATA           big-disk data/outputs root (default <repo>/data)
#   DEMENTOR_LOCAL_ADAPTERS where to materialize pulled PEFT adapters
#                           (default <DEMENTOR_DATA>/local_adapters)
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# 0. Parse args
# --------------------------------------------------------------------------- #
DRY_RUN=0
SKIP_PREFETCH=0
SKIP_LOCAL_ADAPTERS=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)             DRY_RUN=1 ;;
    --skip-prefetch)       SKIP_PREFETCH=1 ;;
    --skip-local-adapters) SKIP_LOCAL_ADAPTERS=1 ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'
      exit 0 ;;
    *)
      echo "ERROR: unknown argument '$arg' (try --help)" >&2
      exit 2 ;;
  esac
done

log()  { echo "[run_partner] $*"; }
warn() { echo "[run_partner][WARN] $*" >&2; }
die()  { echo "[run_partner][FATAL] $*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1. Locate repo + package, pick interpreter, set GPU pool
# --------------------------------------------------------------------------- #
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PKG="$(dirname "$SCRIPT_PATH")"                     # experiments/imitation_safety
DEMENTOR_REPO="${DEMENTOR_REPO:-$(cd "$PKG/../.." && pwd)}"   # auto-detect repo root
export DEMENTOR_REPO
[ -f "$PKG/tinker_erosion.py" ] || die "package layout wrong: $PKG/tinker_erosion.py missing"

PY="${DEMENTOR_PY:-python}"
command -v "$PY" >/dev/null 2>&1 || die "python interpreter '$PY' not found (set DEMENTOR_PY)"
export DEMENTOR_PY="$PY"
# `hf` CLI: prefer the one next to the interpreter, else PATH (huggingface-cli is deprecated).
HF="$(dirname "$(command -v "$PY")")/hf"
[ -x "$HF" ] || HF="$(command -v hf 2>/dev/null || true)"
[ -n "$HF" ] || warn "'hf' CLI not found next to \$PY or on PATH — prefetch/pull steps will be skipped"

export DEMENTOR_GPUS="${DEMENTOR_GPUS:-0,1,2,3}"
LOG_DIR="$PKG/logs"
mkdir -p "$LOG_DIR"

log "repo         = $DEMENTOR_REPO"
log "package      = $PKG"
log "python       = $PY  ($("$PY" --version 2>&1))"
log "hf cli       = ${HF:-<none>}"
log "GPUs         = $DEMENTOR_GPUS"
log "logs         = $LOG_DIR"
[ "$DRY_RUN" = 1 ] && log "MODE         = DRY-RUN (no downloads, no daemon launches)"

# --------------------------------------------------------------------------- #
# 2. Credentials preflight (env OR repo-root .env — mirrors the modules' _load_env)
# --------------------------------------------------------------------------- #
if [ -f "$DEMENTOR_REPO/.env" ]; then
  # export any KEY=value lines so the launched daemons inherit them.
  while IFS= read -r line || [ -n "$line" ]; do   # `|| [ -n ]` -> also read a final line w/o newline
    line="${line%$'\r'}"                          # tolerate CRLF
    case "$line" in ''|\#*) continue ;; esac
    [ "${line#*=}" = "$line" ] && continue          # skip lines without '='
    k="${line%%=*}"; v="${line#*=}"
    k="$(echo "$k" | tr -d '[:space:]')"
    v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
    [ -n "$k" ] || continue
    if [ -z "$(printenv "$k" 2>/dev/null || true)" ]; then export "$k=$v"; fi
  done < "$DEMENTOR_REPO/.env"
fi

# Bridge the canonical DEMENTOR_HF_HOME to the standard HF_HOME that the `hf` CLI + HF libraries read,
# so the prefetch step and the daemons agree on one cache root. (The daemons also read DEMENTOR_HF_HOME
# directly.)  Runs AFTER .env load so a DEMENTOR_HF_HOME set there is honored.
if [ -n "${DEMENTOR_HF_HOME:-}" ] && [ -z "${HF_HOME:-}" ]; then export HF_HOME="$DEMENTOR_HF_HOME"; fi

missing=0
for k in TINKER_API_KEY HF_TOKEN OPENAI_API_KEY; do
  if [ -z "$(printenv "$k" 2>/dev/null || true)" ]; then
    warn "$k is not set (export it, or add it to $DEMENTOR_REPO/.env)"
    missing=1
  else
    log "$k         = set"
  fi
done
if [ "$missing" = 1 ]; then
  echo "" >&2
  echo "  TINKER_API_KEY  -> sample the 572 Tinker adapters (our account-scoped key)" >&2
  echo "  HF_TOKEN        -> gated judge/base models (meta-llama/Llama-Guard-3-8B)" >&2
  echo "  OPENAI_API_KEY  -> prompt-rung behavioral/contrastive analyzer (gpt-4.1-mini)" >&2
  die "missing required credentials (see PARTNER_SETUP.md §4)"
fi

# git-lfs presence + LFS-materialization check (the registries/benchmarks are LFS-tracked).
if command -v git-lfs >/dev/null 2>&1; then
  log "git-lfs      = $(git-lfs version 2>/dev/null | head -1)"
else
  die "git-lfs not found — install it and re-run 'git lfs install && git lfs pull' (PARTNER_SETUP.md §1-§2)"
fi
REG_DEFAULT="$PKG/registry/tinker_adapters.json"
if [ -f "$REG_DEFAULT" ] && head -c 64 "$REG_DEFAULT" | grep -q 'version https://git-lfs'; then
  die "registry is still an LFS pointer stub — run 'git lfs pull' before launching (PARTNER_SETUP.md §2)"
fi

# --------------------------------------------------------------------------- #
# 3. Prefetch HF judge/grader models + Tinker source tokenizers (idempotent)
# --------------------------------------------------------------------------- #
# Full-weight graders (download to the HF cache; the daemons then read them there):
JUDGE_MODELS=(
  "Qwen/Qwen3-8B"                        # RTL harm judge + OR-Bench/XSTest classifiers + fidelity judge
  "cais/HarmBench-Llama-2-13b-cls"       # HarmBench grader (13B)
  "meta-llama/Llama-Guard-3-8B"          # SG-Bench grader (GATED -> needs HF_TOKEN)
  "FlorianJK/strongreject-gemma-2b-merged"  # StrongREJECT ft grader (ungated mirror)
  "sentence-transformers/all-MiniLM-L6-v2"  # fidelity embed scorer (CPU)
)
# Tinker source/target bases — TOKENIZER FILES ONLY (weights are sampled remotely):
BASE_TOKENIZERS=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.6-27B"
  "Qwen/Qwen3.6-35B-A3B"
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
  "openai/gpt-oss-20b"
  "openai/gpt-oss-120b"
)

if [ "$SKIP_PREFETCH" = 1 ]; then
  log "prefetch     = SKIPPED (--skip-prefetch)"
elif [ -z "${HF:-}" ]; then
  warn "prefetch skipped: no 'hf' CLI available"
else
  log "prefetch: judge/grader models + base tokenizers -> HF cache (${HF_HOME:-~/.cache/huggingface})"
  # hf_transfer accelerates downloads if installed; enable only when importable.
  if "$PY" -c "import hf_transfer" >/dev/null 2>&1; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi
  (
    export HF_HUB_OFFLINE=0                          # allow fetching during prefetch only
    for repo in "${JUDGE_MODELS[@]}"; do
      if [ "$DRY_RUN" = 1 ]; then log "  [dry-run] would fetch (full)  $repo"; continue; fi
      log "  fetch (full)  $repo"
      "$HF" download "$repo" >>"$LOG_DIR/prefetch.log" 2>&1 \
        || warn "prefetch failed for $repo (see logs/prefetch.log) — continuing"
    done
    for repo in "${BASE_TOKENIZERS[@]}"; do
      if [ "$DRY_RUN" = 1 ]; then log "  [dry-run] would fetch (tokenizer) $repo"; continue; fi
      log "  fetch (tok)   $repo"
      "$HF" download "$repo" \
          --include "tokenizer*" --include "*.model" --include "*.json" \
          >>"$LOG_DIR/prefetch.log" 2>&1 \
        || warn "tokenizer prefetch failed for $repo (see logs/prefetch.log) — continuing"
    done
  )
fi

# --------------------------------------------------------------------------- #
# 4. Pull current local chatbot_arena seed42 PEFT adapters + rewrite registry
# --------------------------------------------------------------------------- #
# Resolve manifest: the committed in-repo copy (override with DEMENTOR_ADAPTER_MANIFEST).
MANIFEST="${DEMENTOR_ADAPTER_MANIFEST:-$PKG/current_adapters_hf_manifest.json}"
LOCAL_ADAPTERS_ROOT="${DEMENTOR_LOCAL_ADAPTERS:-${DEMENTOR_DATA:-$DEMENTOR_REPO/data}/local_adapters}"

if [ "$SKIP_LOCAL_ADAPTERS" = 1 ]; then
  log "local adapters = SKIPPED (--skip-local-adapters); erosion_daemon will report 0 local items"
elif [ ! -f "$MANIFEST" ]; then
  warn "manifest not found in repo or /data — skipping local-adapter pull (Tinker track unaffected)"
elif [ -z "${HF:-}" ]; then
  warn "local-adapter pull skipped: no 'hf' CLI available"
else
  log "local adapters: manifest = $MANIFEST"
  log "               materialize into $LOCAL_ADAPTERS_ROOT/<key>"
  mkdir -p "$LOCAL_ADAPTERS_ROOT"
  TSV="$(mktemp "${TMPDIR:-/tmp}/seed42_repos.XXXXXX.tsv")"
  # Emit ONLY the manifest's chatbot_arena seed42 repos (never the ~398 stale org repos).
  "$PY" - "$MANIFEST" "$LOCAL_ADAPTERS_ROOT" >"$TSV" <<'PYEOF'
import sys, json, os
manifest_path, root = sys.argv[1], sys.argv[2]
m = json.load(open(manifest_path))
n = 0
for key, e in (m.get("adapters") or {}).items():
    if str(e.get("dataset")) != "chatbot_arena":
        continue
    try:
        if int(e.get("seed", -1)) != 42:
            continue
    except (TypeError, ValueError):
        continue
    repo = e.get("repo_id")
    if not repo:
        continue
    print("%s\t%s\t%s" % (key, repo, os.path.join(root, key)))
    n += 1
sys.stderr.write("chatbot_arena seed42 repos in manifest: %d\n" % n)
PYEOF
  N_SEED42="$(wc -l < "$TSV" | tr -d '[:space:]')"
  log "               $N_SEED42 chatbot_arena seed42 repos to pull"

  while IFS=$'\t' read -r key repo target; do
    [ -n "$key" ] || continue
    if [ -f "$target/adapter_config.json" ]; then
      log "  [skip] $key (already materialized)"; continue
    fi
    if [ "$DRY_RUN" = 1 ]; then log "  [dry-run] would pull $repo -> $target"; continue; fi
    log "  [pull] $repo"
    mkdir -p "$target"
    "$HF" download "$repo" --local-dir "$target" >>"$LOG_DIR/pull_adapters.log" 2>&1 \
      || warn "download failed for $repo (see logs/pull_adapters.log) — continuing"
  done < "$TSV"

  # Rewrite a partner-local registry so the daemons' path/checkpoint_path point at what we
  # materialized (the committed registry's local paths are our-box absolute paths).
  BASE_REGISTRY="${DEMENTOR_REGISTRY:-$REG_DEFAULT}"
  OUT_REGISTRY="$LOCAL_ADAPTERS_ROOT/registry_partner.json"
  if [ "$DRY_RUN" = 1 ]; then
    log "  [dry-run] would rewrite registry -> $OUT_REGISTRY and export DEMENTOR_REGISTRY"
  else
    N_REWROTE="$("$PY" - "$BASE_REGISTRY" "$TSV" "$OUT_REGISTRY" <<'PYEOF'
import sys, json, os
base, tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
reg = json.load(open(base))
n = miss = nomat = 0
for line in open(tsv):
    line = line.rstrip("\n")
    if not line:
        continue
    key, repo, target = line.split("\t")
    if not os.path.exists(os.path.join(target, "adapter_config.json")):
        nomat += 1
        continue
    if isinstance(reg.get(key), dict):
        reg[key]["path"] = target
        reg[key]["checkpoint_path"] = target
        reg[key]["backend"] = "local"
        n += 1
    else:
        miss += 1
json.dump(reg, open(out, "w"))
sys.stderr.write("registry rewrite: %d local entries repointed, %d not-materialized, %d not-in-registry\n"
                 % (n, nomat, miss))
print(n)
PYEOF
)"
    if [ "${N_REWROTE:-0}" -gt 0 ] 2>/dev/null; then
      export DEMENTOR_REGISTRY="$OUT_REGISTRY"
      log "  registry: repointed $N_REWROTE local entries -> DEMENTOR_REGISTRY=$OUT_REGISTRY"
    else
      log "  registry: 0 adapters materialized — keeping default registry (local track stays empty)"
    fi
  fi
  rm -f "$TSV"
fi

# --------------------------------------------------------------------------- #
# 5. Launch the imitation-eval daemons (setsid nohup, detached, logs/)
# --------------------------------------------------------------------------- #
launch() {
  local name="$1" cmd="$2"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] would launch '$name':"
    echo "            $cmd"
    return 0
  fi
  setsid nohup bash -c "$cmd" >>"$LOG_DIR/${name}.log" 2>&1 </dev/null &
  echo "  launched $name (pid $!) -> $LOG_DIR/${name}.log"
}

log "launching imitation-eval daemons (per PARTNER_SETUP.md §9) ..."
cd "$DEMENTOR_REPO"

# Erosion — Tinker track: remote SAMPLE, then batched local JUDGE on the GPUs.
launch tinker_erosion \
  "cd '$DEMENTOR_REPO' && '$PY' '$PKG/tinker_erosion.py' sample --seed all --sample-workers 64 --max-prompts 300 && '$PY' '$PKG/tinker_erosion.py' judge --seed all --gpus '$DEMENTOR_GPUS' --max-prompts 300"

# Erosion — local track (idle-GPU-polite daemon; auto-empty until §4 adapters land).
launch erosion_daemon \
  "cd '$DEMENTOR_REPO' && '$PY' '$PKG/erosion_daemon.py' --seed seed42 --max-prompts 300"

# Fidelity — remote sample + CPU embed scoring only (never touches a GPU).
launch fidelity_daemon \
  "cd '$DEMENTOR_REPO' && '$PY' '$PKG/fidelity_daemon.py' --no-local --scorer embed --max-prompts 300"

# Prompt rung — Tinker track: remote SAMPLE, then batched local JUDGE.
launch prompt_tinker_erosion \
  "cd '$DEMENTOR_REPO' && '$PY' '$PKG/prompt_tinker_erosion.py' sample --sample-workers 64 --max-prompts 300 && '$PY' '$PKG/prompt_tinker_erosion.py' judge --gpus '$DEMENTOR_GPUS' --max-prompts 300"

# Prompt rung — local track (idle-GPU-polite daemon; baselines are SHARED with erosion).
launch prompt_erosion_daemon \
  "cd '$DEMENTOR_REPO' && '$PY' '$PKG/prompt_erosion_daemon.py' --max-prompts 300"

# --------------------------------------------------------------------------- #
# 6. Monitoring / aggregation hints
# --------------------------------------------------------------------------- #
cat <<EOF

============================================================================
  Imitation eval $([ "$DRY_RUN" = 1 ] && echo "PLAN (dry-run)" || echo "launched").  Monitor with:

    tail -f $LOG_DIR/*.log

    # progress snapshots
    $PY $PKG/tinker_erosion.py status
    $PY $PKG/prompt_tinker_erosion.py status
    $PY $PKG/fidelity_eval.py status

    # aggregate to the result CSVs (safe to run anytime; re-run as work completes)
    $PY $PKG/build_erosion_csv.py --seed all
    $PY $PKG/build_prompt_erosion_csv.py
    $PY $PKG/fidelity_eval.py build-csv --scorer embed --seed all

  Idempotent + resumable: re-run this script anytime — downloads skip cached
  files and the daemons skip finished work via their per-item checkpoints.
============================================================================
EOF
