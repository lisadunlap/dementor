"""
Feature-based clustering methods for selecting representative target responses.

This module predates the streamlined interface, so we refresh the imports and
handle modern naming (`behavioral` instead of `vibe`).
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd

try:  # Prefer package-relative imports when invoked via scripts.*
    from ..base import MethodBase
    from ...utils import get_token_count
    from ..utils.clustering import (
        get_cluster_to_idx,
        kmeans_clustering,
        kmodes_clustering,
        sample_from_clusters,
    )
    from ..utils.extract_styles import extract_style_features, get_model_embeddings
    from ..utils.extract_vibe_features import extract_vibe_features
except ImportError:  # pragma: no cover - fall back when executed out of tree
    from scripts.methods.base import MethodBase  # type: ignore
    from scripts.utils import get_token_count  # type: ignore
    from scripts.methods.utils.clustering import (  # type: ignore
        get_cluster_to_idx,
        kmeans_clustering,
        kmodes_clustering,
        sample_from_clusters,
    )
    from scripts.methods.utils.extract_styles import (  # type: ignore
        extract_style_features,
        get_model_embeddings,
    )
    from scripts.methods.utils.extract_vibe_features import extract_vibe_features  # type: ignore


MethodName = Literal["stylistic", "behavioral", "embedding"]


class FeatureClustering(MethodBase):
    """Cluster target responses and sample representative examples for prompting."""

    def __init__(
        self,
        model: str,
        disguise_as: str,
        *,
        num_samples_per_disguise: int = 5,
        seed: int | None = None,
        method: MethodName = "stylistic",
        sample_at_init: bool = True,
        disguise_df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(model, disguise_as)

        if disguise_df is None or disguise_df.empty:
            raise ValueError("FeatureClustering requires a non-empty disguise_df.")

        required = {"prompt", "target_response"}
        missing = required - set(disguise_df.columns)
        if missing:
            raise ValueError(f"disguise_df missing required columns: {sorted(missing)}")

        if method in {"behavioral", "embedding"} and "source_response" not in disguise_df.columns:
            raise ValueError(f"{method} clustering requires 'source_response' in disguise_df.")

        self.method: MethodName = method
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.sample_at_init = sample_at_init

        self.disguise_df = disguise_df.copy()
        self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(get_token_count)
        self.num_samples = len(self.disguise_df)

        self.clusters_df = self.init_clusters()
        if self.sample_at_init:
            self.sampled_df = sample_from_clusters(self.clusters_df, self.seed)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def init_clusters(self) -> pd.DataFrame:
        """Compute or load cached clusters."""
        dataset = os.environ.get("DEMENTOR_DATASET", "generic")
        model_id = self.model.replace("/", "_")
        target_id = self.disguise_as.replace("/", "_")

        if self.method == "stylistic":
            save_dir = f"data/results/{dataset}/clusters/stylistic"
            filename = f"{target_id}_clusters-{self.num_samples}.csv"
            axes_filename = None
            ratings_filename = None
        elif self.method == "behavioral":
            save_dir = f"data/results/{dataset}/clusters/behavioral/{model_id}"
            filename = f"{model_id}_disguised-{target_id}_clusters-{self.num_samples}.csv"
            axes_filename = f"{model_id}_disguised-{target_id}_axes-{self.num_samples}.json"
            ratings_filename = f"{model_id}_disguised-{target_id}_ratings-{self.num_samples}.csv"
        elif self.method == "embedding":
            save_dir = f"data/results/{dataset}/clusters/embedding/{model_id}"
            filename = f"{model_id}_disguised-{target_id}_clusters-{self.num_samples}.csv"
            axes_filename = None
            ratings_filename = None
        else:  # pragma: no cover
            raise ValueError(f"Unsupported clustering method '{self.method}'")

        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, filename)
        if os.path.exists(cache_path):
            print(f"Loading existing clusters from {cache_path}")
            return pd.read_csv(cache_path)

        responses_target = self.disguise_df["target_response"].apply(self._truncate_if_needed)
        responses_source = (
            self.disguise_df["source_response"].apply(self._truncate_if_needed)
            if "source_response" in self.disguise_df.columns
            else None
        )

        if self.method == "stylistic":
            style_features = np.array([extract_style_features(resp) for resp in responses_target])
            cluster_labels, _ = kmodes_clustering(style_features, self.num_samples_per_disguise)
            cluster_to_idx = get_cluster_to_idx(cluster_labels)
        elif self.method == "behavioral":
            assert responses_source is not None
            vibe_features, response_indices = extract_vibe_features(
                responses_source,
                responses_target,
                n_samples=10,
                n_axes=10,
                path_to_axes=os.path.join(save_dir, axes_filename),
                path_to_ratings=os.path.join(save_dir, ratings_filename),
            )
            vibe_features = np.array(vibe_features)
            cluster_labels, _ = kmeans_clustering(vibe_features, self.num_samples_per_disguise)
            cluster_to_idx = get_cluster_to_idx(cluster_labels, response_indices)
        else:  # embedding
            assert responses_source is not None
            embeddings_source = get_model_embeddings(self.disguise_df["source_response"])
            embeddings_target = get_model_embeddings(self.disguise_df["target_response"])
            epsilon = 1e-8
            embeddings_diff = embeddings_source - embeddings_target
            embeddings_diff_norm = embeddings_diff / (np.linalg.norm(embeddings_diff, axis=1, keepdims=True) + epsilon)
            cluster_labels, _ = kmeans_clustering(embeddings_diff_norm, self.num_samples_per_disguise)
            cluster_to_idx = get_cluster_to_idx(cluster_labels)

        clusters = []
        prompts = []
        responses = []
        for cluster, indices in cluster_to_idx.items():
            rows = self.disguise_df.iloc[indices]
            prompts.extend(rows["prompt"].tolist())
            responses.extend(rows["target_response"].tolist())
            clusters.extend([cluster] * len(rows))

        samples_df = pd.DataFrame({"cluster": clusters, "prompt": prompts, "target_response": responses})
        samples_df.to_csv(cache_path, index=False, encoding="utf-8")
        print(f"Clusters saved to {cache_path}")
        return samples_df

    @staticmethod
    def _truncate_if_needed(text: str, token_limit: int = 256) -> str:
        if get_token_count(text) > token_limit:
            return f"{text[: token_limit * 4]}...(truncated)"
        return text

    # ------------------------------------------------------------------ #
    # MethodBase API
    # ------------------------------------------------------------------ #
    def forward(self, prompt: str):
        if self.sample_at_init:
            self.sampled_df["token_length"] = self.sampled_df["target_response"].apply(get_token_count)
            disguise_prompt = self.make_disguise_prompt(self.sampled_df, prompt)
        else:
            sampled_df = sample_from_clusters(self.clusters_df, self.seed)
            sampled_df["token_length"] = sampled_df["target_response"].apply(get_token_count)
            disguise_prompt = self.make_disguise_prompt(sampled_df, prompt)

        return [{"role": "system", "content": disguise_prompt}, {"role": "user", "content": prompt}]

