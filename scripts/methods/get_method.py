"""
Complete method registry for all disguise methods.
"""
try:
    from .contrastive import ContrastiveSystemPrompting
    from .behavioral_based import BehavioralBasedSystemPrompting
    from .random_sampling import RandomSamplingSystemPrompting
    from .stylistic import StylisticSystemPrompting
except ImportError:
    # Fallbacks if relative imports fail
    try:
        from scripts.methods.contrastive import ContrastiveSystemPrompting
        from scripts.methods.behavioral_based import BehavioralBasedSystemPrompting
        from scripts.methods.random_sampling import RandomSamplingSystemPrompting
        from scripts.methods.stylistic import StylisticSystemPrompting
    except ImportError:
        from contrastive import ContrastiveSystemPrompting
        from behavioral_based import BehavioralBasedSystemPrompting
        from random_sampling import RandomSamplingSystemPrompting
        from stylistic import StylisticSystemPrompting

import pandas as pd


def _merge_source_target(source_df: pd.DataFrame | None, target_df: pd.DataFrame | None) -> pd.DataFrame:
    if source_df is None or target_df is None:
        raise ValueError("Clustering methods require both source_df and disguise_df.")

    if "prompt" not in source_df.columns or "prompt" not in target_df.columns:
        raise ValueError("Both source_df and disguise_df must contain a 'prompt' column.")

    src = source_df[["prompt", "model_response"]].rename(columns={"model_response": "source_response"})
    tgt = target_df[["prompt", "target_response"]]
    merged = pd.merge(src, tgt, on="prompt", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping prompts between source and target responses for clustering.")

    return merged


def get_method(method_name, model, disguise_as, disguise_df=None, source_df=None, method_kwargs=None):
    """
    Get a method instance.
    
    Args:
        method_name: One of the four core methods
        model: Source model name
        disguise_as: Target model to disguise as
        disguise_df: DataFrame with target model responses
        source_df: DataFrame with source model responses (required for contrastive)
    
    Available methods:
        - contrastive: Learn differences between models
        - behavioral_based: Capture personality and communication essence
        - stylistic: Focus on measurable surface-level style patterns
        - random_sampling: Example-based disguise
        - stylistic_clustering: Cluster target responses on surface features
        - stylistic_clustering_resample: Same as above but resample each forward pass
        - embedding_clustering: Cluster on embedding deltas between source and target
        - behavioral_clustering: Cluster on behavioral axes (formerly vibe clustering)
        - just_name_it: Straightforward legacy instruction
    """
    # Clean, concise method names
    method_kwargs = method_kwargs or {}

    if method_name == "contrastive":
        if source_df is None:
            raise ValueError("contrastive method requires source_df parameter")
        return ContrastiveSystemPrompting(model, disguise_as, disguise_df=disguise_df, source_df=source_df)
    
    elif method_name == "behavioral_based":
        return BehavioralBasedSystemPrompting(model, disguise_as, disguise_df=disguise_df)
    
    elif method_name == "stylistic":
        return StylisticSystemPrompting(model, disguise_as, disguise_df=disguise_df)
    
    elif method_name == "random_sampling":
        return RandomSamplingSystemPrompting(model, disguise_as, disguise_df=disguise_df)

    elif method_name == "just_name_it":
        try:
            from .extras.legacy_simple_methods import JustNameIt
        except ImportError:
            from scripts.methods.extras.legacy_simple_methods import JustNameIt  # type: ignore
        return JustNameIt(model, disguise_as)

    elif method_name in {"stylistic_clustering", "stylistic_clustering_resample", "embedding_clustering", "behavioral_clustering"}:
        try:
            from .extras.feature_clustering import FeatureClustering
        except ImportError:
            try:
                from scripts.methods.extras.feature_clustering import FeatureClustering  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency missing
                raise ModuleNotFoundError(
                    "Feature clustering methods require the 'kmodes' package. "
                    "Install dependencies via `pip install -r requirements.txt`."
                ) from exc

        merged_df = _merge_source_target(source_df, disguise_df)
        cluster_kwargs = {
            "model": model,
            "disguise_as": disguise_as,
            "disguise_df": merged_df,
        }
        cluster_kwargs.update(method_kwargs)

        if method_name == "stylistic_clustering":
            return FeatureClustering(method="stylistic", sample_at_init=True, **cluster_kwargs)
        if method_name == "stylistic_clustering_resample":
            return FeatureClustering(method="stylistic", sample_at_init=False, **cluster_kwargs)
        if method_name == "embedding_clustering":
            return FeatureClustering(method="embedding", sample_at_init=True, **cluster_kwargs)
        if method_name == "behavioral_clustering":
            return FeatureClustering(method="behavioral", sample_at_init=True, **cluster_kwargs)
    
    # Legacy support for old verbose names
    elif method_name in ["contrastive_system_prompting", "behavioral_based_system_prompting",
                        "stylistic_system_prompting", "random_sampling_system_prompting"]:
        # Redirect to clean names
        clean_name = method_name.replace("_system_prompting", "").replace("random_sampling_system_prompting", "random_sampling")
        return get_method(clean_name, model, disguise_as, disguise_df, source_df)
    
    else:
        available_methods = [
            "contrastive",
            "behavioral_based",
            "stylistic",
            "random_sampling",
            "just_name_it",
            "stylistic_clustering",
            "stylistic_clustering_resample",
            "embedding_clustering",
            "behavioral_clustering",
        ]
        raise ValueError(f"Method '{method_name}' not found. Available methods: {available_methods}")
