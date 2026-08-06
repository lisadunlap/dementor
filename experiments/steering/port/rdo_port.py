#!/usr/bin/env python
"""Port of the RDO concept-cone refusal method (Wollschlaeger et al. 2502.17420, rdo.py) onto
OUR loader (dementor.steering._common.{load_causal_lm, get_transformer_layers}, transformers 5.5.4).

Faithful re-implementation of rdo.py::refusal_cone_optimization + RefusalCone, using plain PyTorch
forward hooks instead of nnsight (so it loads our 2026 models). The differentiable ablation carries
gradient to the cone's float32 basis vectors (unlike steering_rung.make_ablation_hook, which is
no_grad inference).

Replicated numerics (rdo.py DEFAULT_CONFIG + refusal_cone_optimization):
  * cone = `cone_dim` learnable float32 [hidden] vectors, orthonormal via Gram-Schmidt each step.
  * ablation applied at EVERY decoder layer on: layer INPUT (pre-hook) + self_attn output[0] + mlp
    output.  h <- h - (h.d_hat) d_hat  (beta=1 during training).
  * loss = ablation_lambda(1)*CE_ablate + addition_lambda(0.2)*CE_add + retain_lambda(1)*KL_retain.
    - CE_ablate: ablate cone -> harmful prompt emits the (DIM-ablation-bootstrapped) compliant target.
    - CE_add:    add alpha*d at best_layer -> harmless prompt emits a refusal continuation.
    - KL_retain: ablate cone -> harmless last-30-token logits unchanged vs baseline (fp64 KL).
    For cone_dim>1: sample n_sample directions from the cone (hypersphere) + optimize each basis
    vector directly; losses averaged.
  * AdamW lr=1e-2 betas=(.9,.98) amsgrad=True weight_decay=0; effective_batch_size=16 (grad accum);
    grad projected to the sphere tangent; clip_grad_norm 10; patience=5, n_lr_reduce=2 (lr/=10).
  * Dim search min..max, each dim initialised from the previous dim's lowest-loss basis.

CLI:  rdo_port.py --model <path> --dim-dir <dir with direction.pt+metadata> --out-dir <dir>
                  --min-cone-dim 2 --max-cone-dim 4 [--family qwen2.5|gemma|llama3|auto]
                  [--splits-dir <saladbench_splits>] [--max-train N] [--reuse-targets <json dir>]
"""
import os, sys, json, time, argparse, random, re
import torch
import torch.nn.functional as F

_PORT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PORT)                     # ensure port dir importable
sys.path.insert(0, os.path.dirname(_PORT))    # steering package dir (steer_config)
import steer_config as CFG
import rdo_compat  # transformers-5.5.4 compat shims (LossKwargs / chat_template / nemotron gen)
sys.path.insert(0, CFG.REPO)  # repo root for `dementor` imports (env DEMENTOR_REPO)
from dementor.steering._common import load_causal_lm, get_transformer_layers

# ----------------------------------------------------------------------------- config (rdo defaults)
LR = 1e-2
BETAS = (0.9, 0.98)
EFF_BATCH = 16
N_SAMPLE = 8
NUM_TARGET_TOKENS = 30
ABLATION_LAMBDA = 1.0
ADDITION_LAMBDA = 0.2
RETAIN_LAMBDA = 1.0
PATIENCE = 5
N_LR_REDUCE = 2
EPOCHS = 1
CLIPNORM = 10.0
# Abort cone training when this fraction of ablation targets come back empty. Empty targets make
# the ablation cross-entropy NaN, which silently yields an untrained cone (see build_targets).
EMPTY_TARGET_ABORT = float(os.environ.get("RDO_EMPTY_TARGET_ABORT", "0.5"))
# A generated ablation target must carry real, non-repetitive content. See target_is_usable.
MIN_TARGET_CONTENT = int(os.environ.get("RDO_MIN_TARGET_CONTENT", "12"))
MIN_TARGET_DIVERSITY = float(os.environ.get("RDO_MIN_TARGET_DIVERSITY", "0.35"))


_TAG_RE = re.compile(r"<\|?[^<>]{0,40}?\|?>")
_ROLE_RE = re.compile(r"\b(assistant|user|system|model)\b", re.I)


def _distinct_ngram_ratio(s, n=4):
    """Fraction of character n-grams that are distinct. Near 0 for a repeated short unit."""
    if len(s) <= n:
        return 1.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams)


def target_is_usable(t):
    """True when a generated ablation target carries real content for the cone to regress onto.

    Emptiness is not a sufficient test, and neither is a per-model tag blacklist. Three roster
    models degenerate under all-layer dim-ablation, each in a different surface form:

        qwen3.6-27b   ''  (a single <|im_end|>), or '</think>\nassistant\n<think>' with EOS blocked
        gemma-4-31b   '//(//-//-//-//-//-//-...'
        gemma-4-e4b   '<start_of_turn>model\n<start_of_turn>model\n...'

    Only the first is empty; the other two are non-empty and yield a FINITE cross-entropy, so the
    cone trains happily against noise -- gemma-4-31b's first-step loss is 1297 and gemma-4-e4b's
    is 499, against 4.5-23 for every model that produces real text. What the three share is
    extreme repetition of a short unit, so test for that directly rather than blacklisting each
    model's chat template."""
    s = str(t or "")
    if not s.strip():
        return False
    if _distinct_ngram_ratio(s) < MIN_TARGET_DIVERSITY:
        return False                      # a short unit repeated (punctuation or one tag)
    # Strip chat markup generically -- any <...> or <|...|> span, then the bare role words that
    # survive it -- so a target built only from alternating template tokens reduces to nothing.
    # This is deliberately not a per-model tag list: the three degenerate models use three
    # different templates and a fourth would use a fourth.
    stripped = _TAG_RE.sub(" ", s)
    stripped = _ROLE_RE.sub(" ", stripped)
    return len("".join(ch for ch in stripped if ch.isalnum())) >= MIN_TARGET_CONTENT


# rdo.py hardcoded chat templates (verbatim) -----------------------------------------------------
GEMMA_T = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
QWEN25_T = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n")
LLAMA3_T = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- byte-level decode repair -------------------------------------------------------------------
# Some tokenizers ship a decoder MISMATCHED to their vocab under transformers 5.5.4 (e.g.
# DeepSeek-R1-Distill: a SentencePiece-style "▁"-replace decoder over a byte-level "Ġ" BPE
# vocab). batch_decode then leaks the raw byte-level token strings -- spaces show up as "Ġ" and
# newlines as "Ċ" -- so the decoded text has NO real whitespace. Downstream that silently breaks
# the coherence WORD-COUNT gate (every response splits to 1 "word" -> coh_frac=0 -> matched_harm=NaN)
# and feeds garbled text to the harm judge. fix_bytelevel maps such a string back through the GPT-2
# byte<->unicode table to recover real UTF-8 text; it is a NO-OP on normal decodes (which never
# contain the U+0120/U+010A byte-level markers).
def _byte_level_decoder_map():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_LEVEL_MAP = _byte_level_decoder_map()


def fix_bytelevel(text):
    """Recover real text from a decode that leaked raw byte-level BPE markers (Ġ space / Ċ newline).
    No-op on normal decodes."""
    s = str(text)
    if "Ġ" not in s and "Ċ" not in s:
        return s
    try:
        return bytearray(_BYTE_LEVEL_MAP[c] for c in s).decode("utf-8", "replace")
    except KeyError:
        return s  # not a pure byte-level string -> leave untouched


def load_model_mp_aware(model_path):
    """Load the frozen causal LM + tokenizer for cone training, model-parallel when asked.

    Default (DEMENTOR_MP unset): delegate to dementor.steering._common.load_causal_lm, which puts the
    whole model on one device via .to(device) -- the single-GPU roster path, unchanged.

    DEMENTOR_MP=1 (paired with CUDA_VISIBLE_DEVICES=a,b): shard the weights across the 2 visible GPUs
    with device_map="auto" (HF/accelerate pipeline-parallel) so a 120B-class model fits. We must NOT
    call model.to() on a device_map-sharded model (it breaks accelerate's per-shard dispatch). The
    returned `device` is the INPUT-embedding shard -- where model inputs and the cone params live;
    the per-layer ablation/addition hooks co-locate the cone direction onto each layer's own shard,
    and compute_ce_loss builds labels on the lm_head shard. Mirrors dementor.training.local_backend's
    DEMENTOR_MP path (device_map="auto", no .to(), keep hf_device_map)."""
    if not os.environ.get("DEMENTOR_MP"):
        return load_causal_lm(model_path, padding_side="left", device="cuda", dtype="auto")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # dtype="auto": respects a checkpoint quantization_config (gpt-oss MXFP4 stays 4-bit ~63GB rather
    # than dequantizing to ~233GB bf16), and picks the native dtype (bf16) for dense models (Mixtral).
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype="auto", device_map="auto")
    model.eval()
    device = model.get_input_embeddings().weight.device
    dm = getattr(model, "hf_device_map", {}) or {}
    shards = sorted({str(v) for v in dm.values() if str(v) not in ("cpu", "disk")})
    log(f"[MP] device_map sharded across {len(shards) or 'n/a'} GPUs {shards}; input/cone device={device}")
    return tokenizer, model, device


def proj(a, u):
    """(a . u) u  -- u assumed unit (matches rdo projection_einops at all its unit-direction sites)."""
    return (a * u).sum(-1, keepdim=True) * u


# ============================================================================= ablation controller
class ConeOps:
    """Registers differentiable ablation hooks on every decoder layer (input + attn out + mlp out)
    and a single additive pre-hook at `add_layer`. Gradient flows to `self.direction` (a fn of the
    cone params). Mode gated by flags so the same hooks serve ablate / add / off."""

    def __init__(self, model):
        self.model = model
        self.layers = get_transformer_layers(model)
        self.direction = None      # unit [hidden] float32 on device (set per loss term)
        self.alpha = 1.0
        self.add_layer = 0
        self.ablate_on = False
        self.add_on = False
        self.handles = []
        self._register()

    def _ablate(self, h):
        d = self.direction
        # model-parallel: this layer's activation `h` sits on its own shard while the cone
        # direction lives on the cone/input device. Co-locate d onto h's device (a differentiable
        # copy -- gradient flows back to the cone params on their home device).
        if d.device != h.device:
            d = d.to(h.device)
        hf = h.float()
        if d.dim() == 1:
            # single unit direction, broadcast over the whole batch (rdo semantics)
            hf = hf - proj(hf, d)
        else:
            # batched: d is [B, hidden] -- one unit direction per batch row (perf fast-path that
            # runs the B sampled/basis directions in ONE forward). Per-row projection, identical
            # math to the single-direction path applied independently to each row.
            # h [B, T, hidden]; coeff[b,t] = <hf[b,t], d[b]>; hf -= coeff * d.
            coeff = torch.einsum("bth,bh->bt", hf, d).unsqueeze(-1)
            hf = hf - coeff * d.unsqueeze(1)
        return hf.to(h.dtype)

    def _register(self):
        def pre_hook(module, args, kwargs):
            h = None
            if len(args) > 0 and torch.is_tensor(args[0]):
                h = args[0]; pos = "arg"
            elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                h = kwargs["hidden_states"]; pos = "kw"
            else:
                return None
            new = h
            if self.ablate_on:
                new = self._ablate(new)
            if new is h:
                return None
            if pos == "arg":
                return ((new,) + tuple(args[1:]), kwargs)
            kwargs = dict(kwargs); kwargs["hidden_states"] = new
            return (args, kwargs)

        def add_pre_hook(module, args, kwargs):
            if not self.add_on:
                return None
            # co-locate the cone direction onto the add_layer's shard (model-parallel safe).
            def _add_for(h):
                d = self.direction
                if d.device != h.device:
                    d = d.to(h.device)
                # batched d [B, hidden]: add alpha*d[b] to row b (unsqueeze T for broadcast).
                if d.dim() == 2:
                    return self.alpha * d.unsqueeze(1)
                return self.alpha * d
            if len(args) > 0 and torch.is_tensor(args[0]):
                h = args[0]
                new = (h.float() + _add_for(h)).to(h.dtype)
                return ((new,) + tuple(args[1:]), kwargs)
            if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                h = kwargs["hidden_states"]
                kwargs = dict(kwargs); kwargs["hidden_states"] = (h.float() + _add_for(h)).to(h.dtype)
                return (args, kwargs)
            return None

        def out_hook(module, inputs, output):
            if not self.ablate_on:
                return None
            if isinstance(output, tuple):
                h = output[0]
                return (self._ablate(h),) + tuple(output[1:])
            return self._ablate(output)

        for L in self.layers:
            self.handles.append(L.register_forward_pre_hook(pre_hook, with_kwargs=True))
            attn = getattr(L, "self_attn", None) or getattr(L, "attn", None)
            if attn is not None:
                self.handles.append(attn.register_forward_hook(out_hook))
            mlp = getattr(L, "mlp", None) or getattr(L, "feed_forward", None)
            if mlp is not None:
                self.handles.append(mlp.register_forward_hook(out_hook))
        # additive pre-hook lives on the add_layer only, added lazily in set_add_layer()
        self._add_pre_hook_fn = add_pre_hook
        self._add_handle = None

    def set_add_layer(self, idx):
        self.add_layer = idx
        if self._add_handle is not None:
            self._add_handle.remove()
        self._add_handle = self.layers[idx].register_forward_pre_hook(self._add_pre_hook_fn, with_kwargs=True)

    def set_direction(self, d):
        self.direction = (d / d.norm())

    def set_directions(self, D):
        # batched perf fast-path: D is [B, hidden], one direction per batch row. Per-row normalize
        # (rows are already ~unit; this mirrors set_direction's redundant renormalize so numerics
        # match the single-direction path). Stays differentiable -> grad flows to the cone params.
        self.direction = D / D.norm(dim=-1, keepdim=True)

    def remove(self):
        for h in self.handles:
            h.remove()
        if self._add_handle is not None:
            self._add_handle.remove()


# ============================================================================= the cone (rdo RefusalCone)
class RefusalCone:
    def __init__(self, hidden, cone_dim, device, init_vectors=None):
        self.cone_dim = cone_dim
        self.device = device
        self.fn_vectors = [torch.nn.Parameter(torch.randn(hidden, dtype=torch.float32, device=device),
                                              requires_grad=True) for _ in range(cone_dim)]
        if init_vectors:
            for i, iv in enumerate(init_vectors):
                if i >= cone_dim:
                    break
                iv = iv.to(device=device, dtype=torch.float32)
                self.fn_vectors[i].data = (iv / iv.norm()).clone()
        self.orthogonalize()

    def parameters(self):
        return self.fn_vectors

    def transform(self, sample):
        """sample [cone_dim] -> normalized [hidden] direction inside the cone (rdo.transform)."""
        basis = torch.stack(self.fn_vectors, dim=0)               # [cone_dim, hidden]
        v = torch.matmul(sample.to(basis.dtype), basis)          # [hidden]
        return v / v.norm()

    @torch.no_grad()
    def orthogonalize(self):
        for i in range(self.cone_dim):
            for j in range(i):
                self.fn_vectors[i].data.sub_(proj(self.fn_vectors[i].data, self.fn_vectors[j].data))
            self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())

    def basis(self):
        return torch.stack([p.detach().cpu() for p in self.fn_vectors], dim=0)  # [cone_dim, hidden]


# ============================================================================= helpers
def family_of(model_path, override):
    if override and override != "auto":
        return override
    p = model_path.lower()
    if "gemma" in p:
        return "gemma"
    if "qwen2.5" in p or "qwen2_5" in p:
        return "qwen2.5"
    if "llama-3" in p or "llama3" in p:
        return "llama3"
    return "generic"


def hardcoded_template_matches(hard, tokenizer):
    """Does this checkpoint's own chat template actually use the hardcoded family markup?

    family_of() keys off the model PATH, so every "gemma*" checkpoint gets GEMMA_T -- which is the
    gemma-2 markup, inherited verbatim from rdo.py. gemma-4 uses a different scheme entirely
    (`<|turn>user ... <turn|>`, and the 31B additionally opens a `<|channel>thought` block), so
    gemma-4 checkpoints were being trained on prompts they cannot parse. The model then emits
    garbage, every ablation target is degenerate, and the cone trains on noise -- which is exactly
    how gemma-4-31b and gemma-4-e4b ended up with unusable cones while their EVAL numbers (which
    render through the tokenizer, not through this table) looked perfectly sane.

    Compare the special-token MARKUP, not the literal prefix. The hardcoded templates are
    deliberate simplifications -- LLAMA3_T omits the system block that Llama-3.1's own template
    injects -- so a prefix-substring test would reject them and silently switch working models to
    a different rendering. What actually matters is whether the checkpoint speaks the same turn
    markers at all: gemma-2 emits `<start_of_turn>`, gemma-4 emits `<|turn>`, and a template built
    from the wrong one is unparseable rather than merely differently-worded.
    """
    tags = set(re.findall(r"<\|?[^<>{}]+?\|?>", hard))
    if not tags:
        return True
    try:
        ref = tokenizer.apply_chat_template([{"role": "user", "content": "PROBE"}],
                                            tokenize=False, add_generation_prompt=True)
    except Exception:
        return True          # no usable template to check against -- keep the historical behaviour
    return all(t in ref for t in tags)


def make_render(family, tokenizer):
    hard = {"gemma": GEMMA_T, "qwen2.5": QWEN25_T, "llama3": LLAMA3_T}.get(family)
    if hard is not None and hardcoded_template_matches(hard, tokenizer):
        return lambda ins: hard.format(instruction=ins)
    if hard is not None:
        log(f"[render] hardcoded {family!r} template does NOT match this checkpoint's chat "
            f"template -- falling back to the tokenizer's own. Training on the wrong markup "
            f"produces degenerate ablation targets and an untrained cone.")
    def generic(ins):
        msgs = [{"role": "user", "content": ins}]
        r = None
        for kw in ({"enable_thinking": False}, {"reasoning_effort": "low"}, {}):
            try:
                r = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw); break
            except TypeError:
                continue
        if r is None:
            r = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # gpt-oss (harmony): the assistant turn opens with an "analysis" reasoning channel, so the FIRST
        # generated token is a channel marker (<|channel|>), never the answer's first token -> the
        # P(first-token=="I") refusal probe is permanently blind (refrate=0, false PC_FAILS). Prefill the
        # FINAL channel so generation starts at the answer itself (a refusal -> "I"), which the probe can
        # see. Only fires for the harmony render (ends at "<|start|>assistant"); other families untouched.
        rs = r.rstrip()
        if rs.endswith("<|start|>assistant") and "<|channel|>final<|message|>" not in rs[-64:]:
            r = rs + "<|channel|>final<|message|>"
        return r
    return generic


def refusal_token_ids(family, tokenizer):
    if family == "gemma":
        return [235285]
    if family == "qwen2.5":
        return [40, 2121]
    if family == "llama3":
        return [40]
    # generic: leading tokens of "I" / "As" (first-token refusal openers)
    ids = set()
    for w in ("I", "As"):
        for t in (w, " " + w):
            enc = tokenizer.encode(t, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids) or [tokenizer.encode("I", add_special_tokens=False)[0]]


def refusal_metric(logits, refusal_toks, eps=1e-8):
    logits = logits.to(torch.float64)
    probs = F.softmax(logits, dim=-1)
    rp = probs[:, refusal_toks].sum(dim=-1)
    nrp = torch.ones_like(rp) - rp
    return torch.log(rp + eps) - torch.log(nrp + eps)


def compute_ce_loss(logits, labels):
    # device-safe (model-parallel): logits live on the lm_head shard, labels on the input shard.
    # Build the padded label tensor on logits' device so cross_entropy indexes co-located tensors.
    logits = logits.reshape(-1, logits.size(-1))
    labels = labels.reshape(-1).to(logits.device)
    padding = torch.full((logits.size(0),), -100, device=logits.device, dtype=labels.dtype)
    padding[-labels.size(0):] = labels
    return F.cross_entropy(logits, padding, ignore_index=-100)


def compute_ce_loss_batched(logits, label):
    """Batched CE for the perf fast-path. logits [B, S, V] are B forwards of the SAME prompt (one
    per sampled/basis direction); `label` [S] is that prompt's shared label. Returns the mean over
    the B rows of each row's token-mean CE -- exactly (1/B) * sum_b CE_b, i.e. identical to the
    original per-direction loop that did `CE_b / B` then summed the backward()s. Because every row
    shares the same set of non-ignored label positions, F.cross_entropy's mean over all rows'
    non-ignored tokens equals the mean over rows of per-row token-mean CE."""
    B, S, V = logits.shape
    lab = label.reshape(1, S).expand(B, S).reshape(-1).to(logits.device)
    return F.cross_entropy(logits.reshape(B * S, V), lab, ignore_index=-100)


def kl_div_fn(a, b):
    # device-safe: co-locate b on a's device (both are lm_head-shard logits in the MP case, but the
    # detached baseline and the live pass can differ if a caller ever slices them off-device).
    a = a.to(torch.float64); b = b.to(torch.float64).to(a.device)
    return F.kl_div(F.log_softmax(a, -1), F.softmax(b, -1), reduction="batchmean")


def sample_hypersphere(n, dim, device):
    s = torch.randn(n, dim, dtype=torch.float32, device=device).abs()
    return s / s.norm(dim=1, keepdim=True)


# ============================================================================= generation w/ intervention
@torch.no_grad()
def gen_batch(model, tokenizer, ops, prompts, max_new_tokens, device, batch_size, mode,
              direction=None, min_new_tokens=0):
    """mode in {'off','ablate','add'}. Returns decoded continuations (prompt stripped).

    `min_new_tokens` blocks EOS for that many steps. Needed because some models answer ablation
    by ending the turn immediately rather than by saying anything: qwen3.6-27b emits a single
    <|im_end|> under all-layer dim-ablation, which yields an empty target, an all-ignored label
    batch, and a NaN training loss. Forcing a floor on the generation length recovers a usable
    target under the SAME intervention, rather than weakening the intervention to get one."""
    ops.ablate_on = (mode == "ablate")
    ops.add_on = (mode == "add")
    if direction is not None:
        ops.set_direction(direction.to(device))
    out = [""] * len(prompts)
    tokenizer.padding_side = "left"
    for s in range(0, len(prompts), batch_size):
        batch = prompts[s:s + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           use_cache=False, pad_token_id=tokenizer.pad_token_id,
                           **({"min_new_tokens": min_new_tokens} if min_new_tokens else {}))
        for j, t in enumerate(tokenizer.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                                     skip_special_tokens=True)):
            out[s + j] = fix_bytelevel(t)
    ops.ablate_on = ops.add_on = False
    return out


@torch.no_grad()
def bypass_scores(model, tokenizer, prompts, refusal_toks, device, batch_size):
    tokenizer.padding_side = "left"
    scores = []
    for s in range(0, len(prompts), batch_size):
        batch = prompts[s:s + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        logits = model(**enc, use_cache=False).logits[:, -1]
        scores.append(refusal_metric(logits, refusal_toks).cpu())
    return torch.cat(scores)


# ============================================================================= data / targets pipeline
def build_targets(model, tokenizer, ops, render, dim_dir, refusal_toks, device, batch_size,
                  harmful_ins, harmless_ins, alpha, out_targets_dir, reuse_dir=None):
    os.makedirs(out_targets_dir, exist_ok=True)
    ht_path = os.path.join(out_targets_dir, "harmful_targets.json")
    hl_path = os.path.join(out_targets_dir, "harmless_targets.json")
    # optional reuse of rdo's generated targets (byte-identical training data for validation)
    if reuse_dir:
        rht = os.path.join(reuse_dir, "harmful_targets.json")
        rhl = os.path.join(reuse_dir, "harmless_targets.json")
        if os.path.exists(rht) and os.path.exists(rhl):
            log(f"[targets] reusing rdo targets from {reuse_dir}")
            return json.load(open(rht)), json.load(open(rhl))
    if os.path.exists(ht_path) and os.path.exists(hl_path):
        log("[targets] cached")
        return json.load(open(ht_path)), json.load(open(hl_path))

    harmful_p = [render(x) for x in harmful_ins]
    harmless_p = [render(x) for x in harmless_ins]
    dim = dim_dir.to(device)
    log(f"[targets] generating ablation targets (n={len(harmful_p)})")
    abl = gen_batch(model, tokenizer, ops, harmful_p, NUM_TARGET_TOKENS, device, batch_size,
                    "ablate", direction=dim)
    log(f"[targets] generating addition targets (n={len(harmless_p)})")
    ops.set_add_layer(ops.add_layer)  # ensure hook exists
    add = gen_batch(model, tokenizer, ops, harmless_p, NUM_TARGET_TOKENS, device, batch_size,
                    "add", direction=dim)
    ops.alpha = alpha  # (alpha set on ops before this call)
    log(f"[targets] generating retain targets (n={len(harmless_p)})")
    ret = gen_batch(model, tokenizer, ops, harmless_p, NUM_TARGET_TOKENS - 1, device, batch_size, "off")

    # An ablation target is the text the cone is trained to REPRODUCE.  If the model emits nothing
    # under dim-ablation, every label is -100, F.cross_entropy(ignore_index=-100) over an all-
    # ignored batch returns NaN, and from then on: total is NaN -> `total < lowest` is False for
    # NaN -> `lowest` stays inf -> NO step is ever recorded as best -> the saved "cone" is the
    # random initialisation.  The run still writes cone_dim_*.pt and exits 0, so downstream the
    # model looks merely refusal-RESISTANT rather than untrained.  qwen3.6-27b hit exactly this:
    # ablating the dim direction makes it emit a single <|im_end|> token, so 861/861 of its
    # ablation targets were empty.  Refuse to continue instead of manufacturing a null result.
    n_empty = sum(1 for t in abl if not target_is_usable(t))
    if n_empty / max(1, len(abl)) >= EMPTY_TARGET_ABORT:
        # Retry under the SAME ablation with EOS blocked for a floor of tokens. This keeps the
        # intervention intact (still all-layer, still the dim direction) and only denies the model
        # the option of answering by ending the turn. Weakening the ablation instead would train
        # the cone against a different intervention than the one being measured.
        floor = max(8, NUM_TARGET_TOKENS // 2)
        log(f"[targets] {n_empty}/{len(abl)} ablation targets empty -- retrying with "
            f"min_new_tokens={floor} (EOS blocked, ablation unchanged)")
        abl = gen_batch(model, tokenizer, ops, harmful_p, NUM_TARGET_TOKENS, device, batch_size,
                        "ablate", direction=dim, min_new_tokens=floor)
        n_empty = sum(1 for t in abl if not target_is_usable(t))
        log(f"[targets] after retry: {n_empty}/{len(abl)} still unusable")
    if n_empty:
        frac = n_empty / max(1, len(abl))
        log(f"[targets] WARNING {n_empty}/{len(abl)} ({frac:.0%}) ablation targets are UNUSABLE "
            f"(empty or template spam) -- the model produces no content under dim-ablation")
        if frac >= EMPTY_TARGET_ABORT:
            raise RuntimeError(
                f"{n_empty}/{len(abl)} ({frac:.0%}) ablation targets are unusable (threshold "
                f"{EMPTY_TARGET_ABORT:.0%}). The cone cannot be trained: cross-entropy over an "
                f"all-ignored label batch is NaN, so no checkpoint would ever improve and the "
                f"saved cone would be the initialisation. This model is UNTESTABLE under the "
                f"standard protocol -- report it as such rather than as refusal-resistant.")

    harmful_targets = [{"prompt": harmful_p[i], "ablation": abl[i]} for i in range(len(harmful_p))]
    harmless_targets = [{"prompt": harmless_p[i], "addition": add[i].split(".")[0], "retain": ret[i]}
                        for i in range(len(harmless_p))]
    json.dump(harmful_targets, open(ht_path, "w"))
    json.dump(harmless_targets, open(hl_path, "w"))
    return harmful_targets, harmless_targets


def build_prompt_label(tokenizer, instruction, target, device):
    text = instruction + target
    ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")[0]
    label = ids[1:].clone()
    ins_len = len(tokenizer.encode(instruction, add_special_tokens=True)) - 1
    label[:ins_len] = -100
    return ids.to(device), label.to(device)


# ============================================================================= training (rdo port)
def train_cone(model, tokenizer, ops, render, refusal_toks, device, hidden, best_layer, alpha,
               dataset, cone_dim, init_vectors, n_sample, batch_size_gen):
    """One dim of refusal_cone_optimization. dataset = list of dicts with
    ablation_prompt/ablation_labels/addition_prompt/addition_labels/retain_prompt (+ harmful/harmless
    templated prompts for monitoring). Returns dict(basis, lowest_loss, refusal_scores)."""
    cone = RefusalCone(hidden, cone_dim, device, init_vectors=init_vectors)
    opt = torch.optim.AdamW(cone.parameters(), lr=LR, betas=BETAS, weight_decay=0.0, amsgrad=True)
    ops.set_add_layer(best_layer)
    ops.alpha = float(alpha)
    accum = EFF_BATCH
    if cone_dim == 1:
        n_sample = 0

    order = list(range(len(dataset)))
    random.Random(42).shuffle(order)

    vectors, train_losses, refusal_hist = [], [], []
    lowest = float("inf"); patience_c = 0; lr_reduce_c = 0; step = 0
    bucket = dict(sa=0.0, sad=0.0, sr=0.0, ba=0.0, bad=0.0, br=0.0)
    # A single non-finite loss term poisons the whole step: `total` becomes NaN, `total < lowest`
    # is False for NaN, so `lowest` stays inf, NO checkpoint is ever recorded as best, and the
    # saved "cone" is whatever the initialisation produced -- while the run still exits 0 and
    # writes a cone file. That is exactly how qwen3.6-27b produced a cone that had never trained
    # and a PC_FAILS verdict that looked like refusal-resistance. Name the offending term the
    # first time it appears so the failure is legible instead of silent.
    nonfinite_seen = set()

    def _acc(name, t):
        """Accumulate one loss term, reporting the first non-finite occurrence per term."""
        v = t.item()
        if v != v or v in (float("inf"), float("-inf")):
            if name not in nonfinite_seen:
                nonfinite_seen.add(name)
                log(f"    !! loss term {name!r} is non-finite ({v}) -- the cone cannot train; "
                    f"every subsequent step's total will be NaN and lowest_loss will stay inf")
        bucket[name] += v
    t0 = time.time()

    def forward_logits(ids, n=1):
        # PERF: run the SAME prompt as an n-row batch in ONE forward. The per-row ablation/add hooks
        # apply a different cone direction to each of the n rows, so batching the n_sample sampled
        # directions (and the cone_dim basis vectors) collapses n_sample+cone_dim separate batch-1
        # forwards per loss term into a single batched forward -- the fix for the CPU/launch-bound
        # 100%-CPU / low-GPU regime. n=1 reproduces the original single-prompt forward exactly.
        ids2d = ids.unsqueeze(0).expand(n, -1) if n > 1 else ids.unsqueeze(0)
        return model(input_ids=ids2d, use_cache=False).logits

    # Rebuild the [K, hidden] direction stack fresh before EACH loss term (do NOT reuse one stack
    # across terms): each .backward() frees the cone-params->stack subgraph, so a reused stack would
    # error on the next backward. This mirrors the original, which recomputed cone.transform / p.norm
    # per term. The stacks are tiny (K x hidden) so rebuilding is free.
    def stack_samples(samples):
        return torch.stack([cone.transform(sv) for sv in samples], dim=0)   # [n_sample, hidden]

    def stack_basis():
        return torch.stack([p / p.norm() for p in cone.fn_vectors], dim=0)   # [cone_dim, hidden]

    for epoch in range(EPOCHS):
        for oi in order:
            d = dataset[oi]
            # ---- sampled cone directions (cone_dim>1): all n_sample directions in one batch ----
            if n_sample > 0:
                samples = sample_hypersphere(n_sample, cone_dim, device)
                # ablation CE  (mean over the n_sample rows == original sum of CE_k/n_sample)
                ops.ablate_on = True; ops.add_on = False
                ops.set_directions(stack_samples(samples))
                logits = forward_logits(d["abl_ids"], n_sample)[:, :-1]
                la = compute_ce_loss_batched(logits, d["abl_lab"])
                (ABLATION_LAMBDA * la).backward(); _acc("sa", la)
                # addition CE
                if ADDITION_LAMBDA > 0:
                    ops.ablate_on = False; ops.add_on = True
                    ops.set_directions(stack_samples(samples))
                    logits = forward_logits(d["add_ids"], n_sample)[:, :-1]
                    lad = compute_ce_loss_batched(logits, d["add_lab"])
                    (ADDITION_LAMBDA * lad).backward(); _acc("sad", lad)
                # retain KL  (baseline is direction-independent -> compute once, broadcast to rows)
                ops.ablate_on = False; ops.add_on = False
                with torch.no_grad():
                    base = forward_logits(d["ret_ids"], 1)[:, -NUM_TARGET_TOKENS:].detach()
                ops.ablate_on = True
                ops.set_directions(stack_samples(samples))
                ret = forward_logits(d["ret_ids"], n_sample)[:, -NUM_TARGET_TOKENS:]
                lr_ = kl_div_fn(base.expand(n_sample, -1, -1), ret)
                (RETAIN_LAMBDA * lr_).backward(); _acc("sr", lr_)
                ops.ablate_on = False
            # ---- basis vectors directly: all cone_dim basis vectors in one batch ----
            ops.ablate_on = True; ops.add_on = False
            ops.set_directions(stack_basis())
            logits = forward_logits(d["abl_ids"], cone_dim)[:, :-1]
            ba = compute_ce_loss_batched(logits, d["abl_lab"])
            (ABLATION_LAMBDA * ba).backward(); _acc("ba", ba)
            if ADDITION_LAMBDA > 0:
                ops.ablate_on = False; ops.add_on = True
                ops.set_directions(stack_basis())
                logits = forward_logits(d["add_ids"], cone_dim)[:, :-1]
                bad = compute_ce_loss_batched(logits, d["add_lab"])
                (ADDITION_LAMBDA * bad).backward(); _acc("bad", bad)
            ops.ablate_on = False; ops.add_on = False
            with torch.no_grad():
                base = forward_logits(d["ret_ids"], 1)[:, -NUM_TARGET_TOKENS:].detach()
            ops.ablate_on = True
            ops.set_directions(stack_basis())
            ret = forward_logits(d["ret_ids"], cone_dim)[:, -NUM_TARGET_TOKENS:]
            br = kl_div_fn(base.expand(cone_dim, -1, -1), ret)
            (RETAIN_LAMBDA * br).backward(); _acc("br", br)
            ops.ablate_on = False

            step += 1
            if step % accum == 0:
                for p in cone.fn_vectors:
                    if p.grad is not None:
                        p.grad.sub_(proj(p.grad, p.data / p.data.norm()))
                        p.grad.div_(accum)
                torch.nn.utils.clip_grad_norm_(cone.parameters(), CLIPNORM)
                opt.step(); opt.zero_grad()
                cone.orthogonalize()
                total = sum(bucket.values()) / accum
                train_losses.append(total); vectors.append(cone.basis())
                # monitor: cone-ablation refusal bypass on harmful (lower = more erosion).
                # batched: one forward with the cone_dim basis vectors as per-row directions.
                with torch.no_grad():
                    ops.ablate_on = True
                    Dm = torch.stack([p / p.norm() for p in cone.fn_vectors], dim=0)
                    ops.set_directions(Dm)
                    lg = forward_logits(d["harmful_ids"], cone_dim)[:, -1]  # [cone_dim, vocab]
                    bset = refusal_metric(lg, refusal_toks).tolist()
                    ops.ablate_on = False
                refusal_hist.append(bset)
                if total < lowest:
                    lowest = total; patience_c = 0
                else:
                    patience_c += 1
                if (step // accum) % 5 == 0 or patience_c >= PATIENCE:
                    log(f"    step {step//accum} loss={total:.4f} bypass={[round(x,2) for x in bset]} "
                        f"({(time.time()-t0)/60:.1f}m)")
                if patience_c >= PATIENCE:
                    if lr_reduce_c >= N_LR_REDUCE:
                        log("    early stop"); break
                    lr_reduce_c += 1
                    for g in opt.param_groups:
                        g["lr"] /= 10
                    patience_c = 0
                    log(f"    reduce lr -> {opt.param_groups[0]['lr']:.1e}")
                bucket = dict.fromkeys(bucket, 0.0)
        else:
            continue
        break

    idx = int(torch.argmin(torch.tensor(train_losses))) if train_losses else -1
    best_basis = vectors[idx] if vectors else cone.basis()
    return {"basis": best_basis, "lowest_loss": lowest,
            "refusal_scores": refusal_hist, "all_bases": vectors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dim-dir", required=True, help="dir with direction.pt + direction_metadata.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-cone-dim", type=int, default=2)
    ap.add_argument("--max-cone-dim", type=int, default=4)
    ap.add_argument("--family", default="auto")
    ap.add_argument("--splits-dir", default=CFG.SPLITS_DIR)
    ap.add_argument("--max-train", type=int, default=0, help=">0 caps filtered training examples")
    ap.add_argument("--reuse-targets", default=None, help="dir with rdo harmful/harmless_targets.json")
    ap.add_argument("--gen-batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cones_dir = os.path.join(args.out_dir, "cones"); os.makedirs(cones_dir, exist_ok=True)
    random.seed(42); torch.manual_seed(42)

    direction = torch.load(os.path.join(args.dim_dir, "direction.pt"), map_location="cpu").float()
    meta = json.load(open(os.path.join(args.dim_dir, "direction_metadata.json")))
    best_layer = int(meta["layer"]); alpha = float(direction.norm())
    log(f"DIM: layer={best_layer} |dir|(alpha)={alpha:.3f} hidden={direction.numel()}")

    tokenizer, model, device = load_model_mp_aware(args.model)
    model.requires_grad_(False)
    hidden = direction.numel()
    layers = get_transformer_layers(model)
    # clamp DIM's best_layer to model depth (robust for shallow models / reused dim-dirs).
    best_layer = rdo_compat.clamp_layers([best_layer], len(layers))[0]
    family = family_of(args.model, args.family)
    render = make_render(family, tokenizer)
    refusal_toks = refusal_token_ids(family, tokenizer)
    log(f"family={family} refusal_toks={refusal_toks} n_layers={len(layers)}")

    ops = ConeOps(model)
    ops.set_add_layer(best_layer)

    # ---- data: saladbench, filter by bypass score (rdo filter_data) ----
    ht = json.load(open(os.path.join(args.splits_dir, "harmful_train.json")))
    hl = json.load(open(os.path.join(args.splits_dir, "harmless_train.json")))
    harmful_ins = [d["instruction"] for d in ht]
    harmless_ins = [d["instruction"] for d in hl][:len(harmful_ins)]

    harmful_p = [render(x) for x in harmful_ins]
    harmless_p = [render(x) for x in harmless_ins]
    log("[filter] scoring bypass on harmful/harmless train")
    hs = bypass_scores(model, tokenizer, harmful_p, refusal_toks, device, args.gen_batch)
    ls = bypass_scores(model, tokenizer, harmless_p, refusal_toks, device, args.gen_batch)
    hs_l = hs.tolist(); ls_l = ls.tolist()
    fh = [i for i, s in enumerate(hs_l) if s > 0]
    fl = [i for i, s in enumerate(ls_l) if s < 0]
    # fallback for models with weak/atypical refusal-token signal (e.g. reasoning distills): if the
    # sign filter keeps too few, rank by bypass score so training data is non-empty (else 0 triples).
    if len(fh) < 64:
        log(f"[filter] weak harmful refusal signal (kept {len(fh)}); ranking by bypass instead")
        fh = sorted(range(len(hs_l)), key=lambda i: -hs_l[i])
    if len(fl) < 64:
        log(f"[filter] weak harmless signal (kept {len(fl)}); ranking by bypass instead")
        fl = sorted(range(len(ls_l)), key=lambda i: ls_l[i])
    n = min(len(fh), len(fl))
    fh, fl = fh[:n], fl[:n]
    harmful_ins = [harmful_ins[i] for i in fh]
    harmless_ins = [harmless_ins[i] for i in fl]
    if args.max_train and len(harmful_ins) > args.max_train:
        harmful_ins = harmful_ins[:args.max_train]; harmless_ins = harmless_ins[:args.max_train]
    log(f"[filter] kept {len(harmful_ins)} harmful / {len(harmless_ins)} harmless")

    ops.alpha = alpha
    harmful_targets, harmless_targets = build_targets(
        model, tokenizer, ops, render, direction, refusal_toks, device, args.gen_batch,
        harmful_ins, harmless_ins, alpha, os.path.join(args.out_dir, "targets"),
        reuse_dir=args.reuse_targets)

    # ---- build tokenized dataset ----
    log("[data] building tokenized prompts/labels")
    dataset = []
    m = min(len(harmful_targets), len(harmless_targets))
    for i in range(m):
        hp = harmful_targets[i]["prompt"]; abl = harmful_targets[i]["ablation"]
        lp = harmless_targets[i]["prompt"]; add = harmless_targets[i]["addition"]; ret = harmless_targets[i]["retain"]
        abl_ids, abl_lab = build_prompt_label(tokenizer, hp, abl, device)
        add_ids, add_lab = build_prompt_label(tokenizer, lp, add, device)
        ret_ids = tokenizer.encode(lp + ret, add_special_tokens=True, return_tensors="pt")[0].to(device)
        harmful_ids = tokenizer.encode(hp, add_special_tokens=True, return_tensors="pt")[0].to(device)
        dataset.append(dict(abl_ids=abl_ids, abl_lab=abl_lab, add_ids=add_ids, add_lab=add_lab,
                            ret_ids=ret_ids, harmful_ids=harmful_ids))
    log(f"[data] {len(dataset)} training triples")

    # ---- dim search ----
    init_vectors = []
    for cd in range(args.min_cone_dim, args.max_cone_dim + 1):
        cpath = os.path.join(cones_dir, f"cone_dim_{cd}.pt")
        if os.path.exists(cpath):
            log(f"=== cone_dim {cd} cached -> load init from it ===")
            init_vectors = list(torch.load(cpath, map_location="cpu")["basis"])
            continue
        log(f"=== training cone_dim {cd} (init from {len(init_vectors)} prev vectors) ===")
        res = train_cone(model, tokenizer, ops, render, refusal_toks, device, hidden, best_layer,
                         alpha, dataset, cd, init_vectors, N_SAMPLE, args.gen_batch)
        torch.save({"basis": res["basis"], "cone_dim": cd, "best_layer": best_layer,
                    "alpha": alpha, "lowest_loss": res["lowest_loss"],
                    "refusal_scores": res["refusal_scores"]}, cpath)
        log(f"[save] {cpath} lowest_loss={res['lowest_loss']:.4f}")
        init_vectors = list(res["basis"])

    ops.remove()
    log("DONE port training")


if __name__ == "__main__":
    main()
