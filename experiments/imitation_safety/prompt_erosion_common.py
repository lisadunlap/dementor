#!/usr/bin/env python
"""Shared infra for the PROMPT-RUNG imitation safety-erosion evaluation.

This is the disguise-ladder PROMPT rung's analog of the SFT/DPO weight-erosion sweep
(erosion_common.py).  Instead of a LoRA adapter that trained source model A to imitate target B,
here A is PROMPTED to imitate B via one of the 5 registered disguise methods, and we measure how
much A's safety erodes vs A's OWN unadapted baseline:

    erosion = harm(A prompted-as-B) - harm(baseline A)              # harm-axis benchmarks
    over_refusal_delta = over_refusal(prompted) - over_refusal(baseline)   # benign benchmarks

The 5 methods (dementor/methods/get_method.py) each build a disguise SYSTEM PROMPT (+/- few-shot
examples) from the TARGET's chatbot_arena responses, then it is applied to the SOURCE at generation
time (method.forward(prompt) -> [system,user] messages -> rendered with the source's chat template):
  * just_name_it     : "Answer in the style of <B>."  (name only, no data)
  * random_sampling  : k=5 few-shot Q/A examples of B's responses
  * stylistic        : LOCALLY-computed surface-style rules (length/markdown/bullets/...) + examples
  * behavioral       : gpt-4.1-mini analyzer distills B's communication ESSENCE into rules + examples
  * contrastive      : gpt-4.1-mini analyzer contrasts B-vs-A distinctive features + good/bad examples
The behavioral/contrastive analyzer rules are cached per (method,source,target) so a resume never
re-calls the API and the disguise is reproducible.

EVERYTHING HEAVY IS REUSED from erosion_common (EC), never reinvented:
  * benchmark subsamples  -> EC.get_subsample (the SAME cached 300-prompt seed42 core-first sets)
  * local generation load -> EC.load_gen_model (base source, adapter_dir=None; MP-aware)
  * RTL harm judge        -> EC.run_rtl_judge (Qwen3-8B subprocess)
  * canonical graders     -> EC.run_grader (HarmBench-cls / StrongREJECT-ft / Llama-Guard / local)
  * per-benchmark metric  -> EC.benchmark_metric
  * BASELINES             -> REUSED from EC.WORK/baseline_<slug>/metrics.json (model A UNADAPTED,
                            scored on the identical subsamples) -> byte-identical baseline to the
                            SFT/DPO rung, so the two rungs are directly comparable.  We never
                            regenerate baselines here.

The disguise is built from chatbot_arena data (matrix_baselines/<slug>_train.csv); the EVAL is still
the 7 safety benchmarks.  IMPORT-ONLY infra: it only *imports* read-only helpers from EC / the repo;
it never touches sequencer.py, local_backend.py, the steering files, or the running erosion/fidelity
daemons.  It writes ONLY under exp_imitation_safety/work_prompt_erosion + prompt_erosion_* results.
"""
import os, sys, json, time, hashlib, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import erosion_common as EC  # noqa: E402  (no torch at import time; EC adds REPO to sys.path)

# Repo root (dementor imports + repo-relative data). env DEMENTOR_REPO, default: two levels up.
# Reused from EC so this rung is path-portable off our /data box (see PARTNER_SETUP.md).
REPO = EC.REPO

log = EC.log

# ------------------------------------------------------------------ layout (all off the home fs)
WORK = os.path.join(EC.WORK_ROOT, "work_prompt_erosion")   # adapters-analog: <id>/<bench>/...
RULES_CACHE = os.path.join(WORK, "_rules_cache")           # cached behavioral/contrastive analyzer rules
LOG_DIR = os.path.join(HERE, "logs")
# data/ is symlinked -> /data; results land on the big disk (shared safety results dir).
RESULTS = EC.RESULTS_SAFETY
# Target B's chatbot_arena responses used to BUILD the disguise (prompt,model_response,model).
# Off DATA_ROOT (env DEMENTOR_DATA, default <repo>/data) so it follows the big-disk data root;
# override the whole path with DEMENTOR_DISGUISE_DATA on a machine with a different layout.
DISGUISE_DATA = EC._env("DEMENTOR_DISGUISE_DATA",
                        os.path.join(EC.DATA_ROOT, "model-responses", "matrix_baselines", "chatbot_arena"))
for _d in (WORK, RULES_CACHE, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

PY = EC.PY
METHODS = ["just_name_it", "random_sampling", "stylistic", "behavioral", "contrastive"]
ANALYZER_METHODS = {"behavioral", "contrastive"}
DISGUISE_POOL_N = 500          # seeded subsample of B's ~10k chatbot_arena responses for the disguise
DISGUISE_SEED = 42             # deterministic few-shot / analyzer example sampling
DEFAULT_MAX_NEW_TOKENS = 256   # match the erosion rung's generation cap (comparability)
DEFAULT_MAX_PROMPTS = EC.DEFAULT_MAX_PROMPTS
DEFAULT_SUBSAMPLE_SEED = EC.DEFAULT_SUBSAMPLE_SEED

# LOCAL-generation model-parallel gate: mirror EC.NEEDS_MP (gemma-4-31b inference fits ONE card --
# baseline_gemma-4-31b completed single-card in the erosion sweep -- so no extra MP source here).
NEEDS_MP = set(EC.NEEDS_MP)


def _load_env():
    """Load OPENAI_API_KEY (analyzer) + TINKER_API_KEY + HF_TOKEN from the repo .env if unset."""
    p = os.path.join(REPO, ".env")
    if not os.path.exists(p):
        return
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ==================================================================== roster maps
_ROSTER = None


def _roster_maps():
    """{slug: hf_id}, {slug: backend} for every roster+legacy model (from config.yaml)."""
    global _ROSTER
    if _ROSTER is None:
        from dementor import config
        s2id, s2be = {}, {}
        for m in config.roster(include_legacy=True):
            s2id[m["slug"]] = m["id"]
            s2be[m["slug"]] = m.get("backend")
        _ROSTER = (s2id, s2be)
    return _ROSTER


# ==================================================================== pair / item worklist
def build_pair_worklist():
    """Distinct (source, target) chatbot_arena pairs, taken from the SAME adapter registry the
    erosion rung covers -> the prompt rung evaluates the identical pair set for direct comparability.

    Each pair dict carries how to GENERATE A prompted-as-B: source slug/hf/backend + target slug/hf.
    Source backend (local vs tinker) comes from config.yaml (the source model's own backend).
    """
    s2id, s2be = _roster_maps()
    reg = EC.load_registry_entries()
    seen, pairs = set(), []
    for key in reg:
        m = EC.ADAPTER_RE.match(key)
        if not m or m.group(1) != "chatbot_arena":
            continue
        src, tgt = m.group(2), m.group(3)
        if (src, tgt) in seen or src == tgt:
            continue
        src_hf = s2id.get(src)
        tgt_hf = s2id.get(tgt)
        if src_hf is None or tgt_hf is None:
            continue
        # disguise data for B must exist (matrix_baselines chatbot_arena train csv)
        if not os.path.exists(os.path.join(DISGUISE_DATA, f"{tgt}_train.csv")):
            continue
        seen.add((src, tgt))
        pairs.append({
            "source": src, "target": tgt, "source_hf": src_hf, "target_hf": tgt_hf,
            "backend": "tinker" if s2be.get(src) == "tinker" else "local",
            "needs_mp": src_hf in NEEDS_MP,
        })
    return sorted(pairs, key=lambda p: (p["source"], p["target"]))


def item_id(method, source, target):
    return f"prompt_{method}_{source}_as_{target}"


def parse_item_id(iid):
    """('prompt_<method>_<src>_as_<tgt>') -> (method, source, target) or None."""
    if not iid.startswith("prompt_"):
        return None
    rest = iid[len("prompt_"):]
    for meth in sorted(METHODS, key=len, reverse=True):
        if rest.startswith(meth + "_"):
            body = rest[len(meth) + 1:]
            if "_as_" in body:
                s, t = body.split("_as_", 1)
                return meth, s, t
    return None


def build_worklist(methods=None, backend=None):
    """Full (method x pair) item worklist.  methods=None -> all 5; backend in {'local','tinker'}
    filters to that source-generation track.  Each item id = prompt_<method>_<src>_as_<tgt>."""
    methods = methods or METHODS
    items = []
    for p in build_pair_worklist():
        if backend and p["backend"] != backend:
            continue
        for meth in methods:
            items.append({
                "id": item_id(meth, p["source"], p["target"]),
                "kind": "prompt_disguise", "method": meth,
                "source": p["source"], "target": p["target"],
                "base_model": p["source_hf"], "target_hf": p["target_hf"],
                "backend": p["backend"], "needs_mp": p["needs_mp"], "seed": "seed42",
            })
    return items


def find_item(iid):
    parsed = parse_item_id(iid)
    if parsed is None:
        return None
    meth, src, tgt = parsed
    s2id, s2be = _roster_maps()
    src_hf, tgt_hf = s2id.get(src), s2id.get(tgt)
    if src_hf is None or tgt_hf is None:
        return None
    return {"id": iid, "kind": "prompt_disguise", "method": meth, "source": src, "target": tgt,
            "base_model": src_hf, "target_hf": tgt_hf,
            "backend": "tinker" if s2be.get(src) == "tinker" else "local",
            "needs_mp": src_hf in NEEDS_MP, "seed": "seed42"}


def item_dir(iid):
    d = os.path.join(WORK, iid)
    os.makedirs(d, exist_ok=True)
    return d


def done(iid):
    d = os.path.join(WORK, iid)
    return os.path.exists(os.path.join(d, "metrics.json")) or os.path.exists(os.path.join(d, "ERROR.json"))


# ==================================================================== disguise construction
def load_disguise_dfs(source_slug, target_slug, pool_n=DISGUISE_POOL_N, seed=DISGUISE_SEED):
    """(source_df, target_df) for the disguise: a deterministic seeded subsample of each model's
    chatbot_arena responses (matrix_baselines).  target_df has `target_response` (methods' convention);
    source_df keeps `model_response` (contrastive needs it).  Bounded to pool_n rows to keep the
    style/analyzer construction fast + deterministic."""
    import pandas as pd
    tp = os.path.join(DISGUISE_DATA, f"{target_slug}_train.csv")
    sp = os.path.join(DISGUISE_DATA, f"{source_slug}_train.csv")
    tdf = pd.read_csv(tp)
    if len(tdf) > pool_n:
        tdf = tdf.sample(n=pool_n, random_state=seed).reset_index(drop=True)
    tdf = tdf.rename(columns={"model_response": "target_response"})[["prompt", "target_response"]]
    sdf = None
    if os.path.exists(sp):
        sdf = pd.read_csv(sp)
        if len(sdf) > pool_n:
            sdf = sdf.sample(n=pool_n, random_state=seed).reset_index(drop=True)
        sdf = sdf[["prompt", "model_response"]]
    return sdf, tdf


def _rules_cache_path(method, source_slug, target_slug):
    return os.path.join(RULES_CACHE, f"{method}_{source_slug}_as_{target_slug}.json")


def make_disguise(method, source_hf, target_hf, source_slug, target_slug,
                  source_df, target_df, seed=DISGUISE_SEED, logf=None):
    """Build the disguise METHOD instance for (source,target).  behavioral/contrastive analyzer rules
    are cached to disk keyed by (method,source,target); a cache hit builds the method WITHOUT calling
    the gpt-4.1-mini analyzer (rules injected), so a resume never re-hits the API and the disguise is
    reproducible.  Returns a method whose .forward(prompt) yields the disguise messages."""
    from dementor.methods.get_method import get_method

    if method not in ANALYZER_METHODS:
        return get_method(method, source_hf, target_hf, disguise_df=target_df,
                          source_df=source_df, method_kwargs={"seed": seed})

    cache = _rules_cache_path(method, source_slug, target_slug)
    if os.path.exists(cache):
        try:
            rules = json.load(open(cache)).get("rules")
        except Exception:
            rules = None
        if rules:
            if method == "behavioral":
                from dementor.methods.behavioral_based import BehavioralBasedSystemPrompting
                m = BehavioralBasedSystemPrompting(source_hf, target_hf, disguise_df=None, seed=seed)
                m.disguise_df = target_df.copy()
                m.behavior_rules = rules
            else:  # contrastive
                from dementor.methods.contrastive import ContrastiveSystemPrompting
                m = ContrastiveSystemPrompting(source_hf, target_hf, disguise_df=None,
                                               source_df=None, seed=seed)
                m.disguise_df = target_df.copy()
                m.source_df = source_df.copy() if source_df is not None else None
                m.contrastive_features = rules
            log(f"[disguise] {method} {source_slug}->{target_slug}: rules CACHE-HIT", logf)
            return m

    # cache miss: build normally (runs the analyzer once) then persist the rules
    m = get_method(method, source_hf, target_hf, disguise_df=target_df,
                   source_df=source_df, method_kwargs={"seed": seed})
    rules = getattr(m, "behavior_rules", None) if method == "behavioral" \
        else getattr(m, "contrastive_features", None)
    if rules:
        try:
            fd, tmp = tempfile.mkstemp(dir=RULES_CACHE, suffix=".tmp")
            os.close(fd)
            json.dump({"method": method, "source": source_slug, "target": target_slug, "rules": rules},
                      open(tmp, "w"))
            os.replace(tmp, cache)
        except Exception:
            pass
    log(f"[disguise] {method} {source_slug}->{target_slug}: built + cached rules "
        f"({len(rules) if rules else 0} chars)", logf)
    return m


def method_messages(method_obj, prompt):
    """Force clean [system,user] messages from any method (bypass the gemma single-turn pre-wrap in
    MethodBase._wrap by masking self.model), and return (system_prompt, user_prompt).  Rendering into
    the SOURCE model's actual chat template happens in render_disguise (which handles gemma's lack of a
    system role)."""
    method_obj.model = "__generic__"
    msgs = method_obj.forward(prompt)
    if len(msgs) >= 2 and msgs[0].get("role") == "system":
        return msgs[0].get("content", ""), msgs[-1].get("content", str(prompt))
    # defensive: a single (already-merged) message
    return "", msgs[-1].get("content", str(prompt))


def render_disguise(tok, base_model, sys_prompt, user_prompt, chat_kwargs):
    """Render (system,user) into the SOURCE model's chat template with add_generation_prompt.

    Fallback order: (system+user, chat_kwargs) -> (system+user, {}) -> (merged-into-user, chat_kwargs)
    -> (merged-into-user, {}).  The merged-user forms cover models whose template has NO system role
    (gemma), matching MethodBase._wrap's gemma behaviour without double-templating."""
    merged = f"{sys_prompt}\n\n{user_prompt}".strip() if sys_prompt else str(user_prompt)
    combos = [
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        [{"role": "user", "content": merged}],
    ]
    last = None
    for msgs in combos:
        for kw in (chat_kwargs, {}):
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            except Exception as e:  # noqa: BLE001  TypeError(kwarg) or TemplateError(no system role)
                last = e
                continue
    raise last if last else RuntimeError("render_disguise: no chat template worked")


# ==================================================================== local disguised generation
def generate_disguised_local(base_model, method_obj, prompts, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                             batch_size=16, logf=None):
    """Load the BASE source model (adapter_dir=None) ONCE, render each benchmark prompt through the
    disguise, generate greedy (left-padded, OOM-halving), and clean.  Reuses EC.load_gen_model +
    matrix.CHAT_TEMPLATE_KWARGS + clean_response so gpt-oss channels / gemma templates are handled.
    Model is freed by the caller (all benchmarks share one load)."""
    import torch
    from dementor.training.matrix import CHAT_TEMPLATE_KWARGS, clean_response
    tok, mdl, input_dev = _GEN_CACHE.get(base_model, (None, None, None))
    if tok is None:
        raise RuntimeError("call load_source_model() first")
    chat_kwargs = CHAT_TEMPLATE_KWARGS.get(base_model, {})
    texts = []
    for p in prompts:
        sysp, userp = method_messages(method_obj, str(p))
        texts.append(render_disguise(tok, base_model, sysp, userp, chat_kwargs))
    outs, i, bs = [], 0, max(1, batch_size)
    t0 = time.time()
    while i < len(texts):
        chunk = texts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(input_dev)
        try:
            with torch.no_grad():
                seq = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            log(f"[gen] OOM -> batch_size {bs}", logf)
            continue
        new = seq[:, enc["input_ids"].shape[1]:]
        for t in tok.batch_decode(new, skip_special_tokens=False):
            outs.append(clean_response(base_model, t))
        i += len(chunk)
        if i % 64 == 0 or i >= len(texts):
            log(f"[gen] {i}/{len(texts)} ({i/max(time.time()-t0,1e-6):.2f}/s)", logf)
    return outs


_GEN_CACHE = {}


def load_source_model(base_model, logf=None):
    """Load the base source model once and cache it (adapter_dir=None -> A UNADAPTED)."""
    tok, mdl, input_dev = EC.load_gen_model(base_model, None, logf)
    _GEN_CACHE[base_model] = (tok, mdl, input_dev)
    return tok, mdl, input_dev


def free_source_model(base_model):
    _GEN_CACHE.pop(base_model, None)
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ==================================================================== baseline reuse (from EC)
def baseline_metrics(source_slug):
    """The SFT/DPO rung's baseline for source A (model A UNADAPTED, same subsamples) -- REUSED as the
    prompt rung's baseline for a byte-identical, directly-comparable erosion delta.  None if not yet
    computed by the running erosion sweep."""
    p = os.path.join(EC.WORK, f"baseline_{source_slug}", "metrics.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None
