"""
Complete method registry for all disguise methods.
"""
try:
    from .contrastive import ContrastiveSystemPrompting
    from .vibe_based import VibeBasedSystemPrompting
    from .random_sampling import RandomSamplingSystemPrompting
    from .stylistic import StylisticSystemPrompting
    from .contrastive_with_al_examples import ContrastiveWithALExamples
except ImportError:
    # Fallbacks if relative imports fail
    from contrastive import ContrastiveSystemPrompting
    from vibe_based import VibeBasedSystemPrompting
    from random_sampling import RandomSamplingSystemPrompting
    from stylistic import StylisticSystemPrompting
    from contrastive_with_al_examples import ContrastiveWithALExamples


def get_method(method_name, model, disguise_as, disguise_df=None, source_df=None, method_kwargs=None):
    """
    Get a method instance.
    
    Args:
        method_name: One of the five core methods
        model: Source model name
        disguise_as: Target model to disguise as
        disguise_df: DataFrame with target model responses
        source_df: DataFrame with source model responses (required for contrastive)
    
    Available methods:
        - contrastive: Learn differences between models
        - vibe_based: Capture personality and communication essence
        - stylistic: Focus on measurable surface-level style patterns
        - random_sampling: Example-based disguise
        - contrastive_with_al_examples: Contrastive rules + AL-selected examples
    """
    # Clean, concise method names
    method_kwargs = method_kwargs or {}

    if method_name == "contrastive":
        if source_df is None:
            raise ValueError("contrastive method requires source_df parameter")
        return ContrastiveSystemPrompting(model, disguise_as, disguise_df=disguise_df, source_df=source_df)
    
    elif method_name == "vibe_based":
        return VibeBasedSystemPrompting(model, disguise_as, disguise_df=disguise_df)
    
    elif method_name == "stylistic":
        return StylisticSystemPrompting(model, disguise_as, disguise_df=disguise_df)
    
    elif method_name == "random_sampling":
        return RandomSamplingSystemPrompting(model, disguise_as, disguise_df=disguise_df)
        
    # Note: active_learning is no longer exposed as a standalone method. Use
    # the composite method contrastive_with_al_examples (or contrastive_al) to
    # apply AL-based example selection together with contrastive rules.

    elif method_name in ("contrastive_with_al_examples", "contrastive_al"):
        if source_df is None or disguise_df is None:
            raise ValueError("contrastive_with_al_examples requires disguise_df and source_df")
        # Prepare kwargs for AL selector
        al_kwargs = {
            'd_regular': method_kwargs.get('al_d_regular', 3),
            'p_threshold': method_kwargs.get('al_p_threshold', 0.1),
            'q_threshold': method_kwargs.get('al_q_threshold', 0.1),
            'batch_size': method_kwargs.get('al_batch_size', 10),
            'max_iterations': method_kwargs.get('al_max_iterations', 5),
            'relaxation_factor': method_kwargs.get('al_relaxation_factor', 1.2),
            'seed': method_kwargs.get('al_seed'),
        }
        return ContrastiveWithALExamples(
            model,
            disguise_as,
            disguise_df=disguise_df,
            source_df=source_df,
            num_examples=method_kwargs.get('al_num_examples', 5),
            al_kwargs=al_kwargs,
            selector=method_kwargs.get('example_selector', 'al'),
        )
    
    # Legacy support for old verbose names
    elif method_name in ["contrastive_system_prompting", "vibe_based_system_prompting", 
                        "stylistic_system_prompting", "random_sampling_system_prompting"]:
        # Redirect to clean names
        clean_name = method_name.replace("_system_prompting", "").replace("random_sampling_system_prompting", "random_sampling")
        return get_method(clean_name, model, disguise_as, disguise_df, source_df)
    
    else:
        available_methods = ["contrastive", "vibe_based", "stylistic", "random_sampling", "contrastive_with_al_examples", "contrastive_al"]
        raise ValueError(f"Method '{method_name}' not found. Available methods: {available_methods}")
