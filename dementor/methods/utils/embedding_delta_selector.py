from __future__ import annotations

"""
Embedding-delta selector for example selection.

Selects K examples that maximize coverage of the delta directions between
target and source responses in embedding space.

Algorithm
- Align responses by prompt via inner join of target_df and source_df
- Compute embeddings e_s for source responses and e_t for target responses
- Compute deltas d_i = e_t - e_s and scores s_i = ||d_i||
- Build a candidate pool: top M = pool_multiplier * K by s_i
- Normalize candidate deltas and run KMeans with K clusters
- Pick one representative per cluster (closest to centroid)

Notes
- Uses a lightweight HF encoder via transformers (no sentence-transformers dependency)
- Deterministic with optional seed
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer, AutoModel


class _HFEmbedder:
    """Minimal embedding wrapper using transformers with mean pooling."""

    def __init__(self, model_name: str = "intfloat/e5-small-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Some embedding models (e.g., E5) benefit from instruction prefixes
        self._prefix = None
        name_lower = model_name.lower()
        if "e5" in name_lower:
            # E5 uses query/document prefixes for best results; these are generic answers, use "passage:"
            self._prefix = "passage: "

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            if self._prefix:
                chunk = [self._prefix + (t or "") for t in chunk]
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state  # [B, T, H]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            vec = (summed / counts).cpu().numpy()
            # L2 normalize
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
            embs.append(vec)
        if not embs:
            return np.zeros((0, getattr(self.model.config, "hidden_size", 768)))
        return np.vstack(embs)


class EmbeddingDeltaSelector:
    """
    Select examples by diversity and magnitude of embedding deltas.

    Inputs
    - target_responses_df: DataFrame with columns ['prompt', 'target_response'] (or 'model_response')
    - source_responses_df: DataFrame with columns ['prompt', 'model_response'] (or 'target_response')

    Returns
    - DataFrame subset of target_responses_df with selected rows
    """

    def __init__(
        self,
        target_responses_df: pd.DataFrame,
        source_responses_df: pd.DataFrame,
        num_examples: int = 5,
        embedding_model: str = "intfloat/e5-small-v2",
        pool_multiplier: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self.num_examples = int(max(1, num_examples))
        self.pool_multiplier = int(max(1, pool_multiplier))
        self.embedding_model = embedding_model
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Normalize column names and align by prompt
        tgt = target_responses_df.copy()
        if "target_response" not in tgt.columns and "model_response" in tgt.columns:
            tgt = tgt.rename(columns={"model_response": "target_response"})

        src = source_responses_df.copy()
        if "model_response" not in src.columns and "target_response" in src.columns:
            src = src.rename(columns={"target_response": "model_response"})

        # Inner join on prompt to ensure pairs
        self._joined = pd.merge(
            tgt[[c for c in ["prompt", "target_response"] if c in tgt.columns]],
            src[[c for c in ["prompt", "model_response"] if c in src.columns]],
            on="prompt",
            how="inner",
        )

        self._tgt_ref = tgt  # original target df for returning selected rows by prompt
        self._embedder = _HFEmbedder(embedding_model)

    def _compute_deltas(self) -> Dict[str, np.ndarray]:
        texts_t = self._joined["target_response"].astype(str).fillna("").tolist()
        texts_s = self._joined["model_response"].astype(str).fillna("").tolist()
        E_t = self._embedder.encode(texts_t)
        E_s = self._embedder.encode(texts_s)
        deltas = E_t - E_s
        norms = np.linalg.norm(deltas, axis=1)
        return {"deltas": deltas, "norms": norms}

    def select_examples(self, prompt: Optional[str] = None) -> pd.DataFrame:
        if len(self._joined) == 0:
            # Fallback: return empty slice of target df
            return self._tgt_ref.head(0)

        mats = self._compute_deltas()
        deltas = mats["deltas"]
        norms = mats["norms"]

        # Candidate pool
        k = min(self.num_examples, len(self._joined))
        M = min(self.pool_multiplier * k, len(self._joined))
        top_idx = np.argsort(-norms)[:M]

        pool_d = deltas[top_idx]
        # Normalize to unit vectors for directional clustering
        pool_d = pool_d / (np.linalg.norm(pool_d, axis=1, keepdims=True) + 1e-12)

        n_clusters = min(k, len(top_idx))
        if n_clusters <= 0:
            return self._tgt_ref.head(0)

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.seed)
        labels = km.fit_predict(pool_d)
        centers = km.cluster_centers_

        # Pick closest to centroid per cluster (medoid-like)
        chosen_pool_idx: List[int] = []
        for c in range(n_clusters):
            c_mask = labels == c
            idxs = np.where(c_mask)[0]
            if len(idxs) == 0:
                continue
            diffs = np.linalg.norm(pool_d[idxs] - centers[c], axis=1)
            best_local = idxs[int(np.argmin(diffs))]
            chosen_pool_idx.append(best_local)

        # Map back to global indices
        chosen_global_idx = [int(top_idx[i]) for i in chosen_pool_idx]
        selected_prompts = self._joined.iloc[chosen_global_idx]["prompt"].tolist()

        # Return corresponding rows from original target df, preserving order
        sel = self._tgt_ref[self._tgt_ref["prompt"].isin(selected_prompts)]
        # Stable order by prompt occurrence in selected_prompts
        order = {p: i for i, p in enumerate(selected_prompts)}
        sel = sel.assign(_ord=sel["prompt"].map(order)).sort_values("_ord").drop(columns=["_ord"])  # type: ignore
        return sel.head(k)

    def get_selection_statistics(self) -> Dict[str, float | int]:
        return {
            "num_candidates": int(len(self._joined)),
            "num_examples": int(self.num_examples),
            "pool_multiplier": int(self.pool_multiplier),
            "embedding_model": self.embedding_model,
        }

