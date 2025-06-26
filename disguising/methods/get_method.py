from methods.base import RandomSampleDisguise, JustNameIt, VibeBasedDisguise, VibeBasedDisguiseOneSided
from methods.feature_clustering import FeatureClustering

def get_method(method_name, model, disguise_as, disguise_df=None):
    """
    Get a method from the methods module.
    """
    if method_name == "random_sample_1_example":
        return RandomSampleDisguise(model, disguise_as, num_samples_per_disguise=1, disguise_df=disguise_df)
    elif method_name == "random_sample_3_examples":
        return RandomSampleDisguise(model, disguise_as, num_samples_per_disguise=3, disguise_df=disguise_df)
    elif method_name == "random_sample_5_examples":
        return RandomSampleDisguise(model, disguise_as, num_samples_per_disguise=5, disguise_df=disguise_df)
    elif method_name == "just_name_it":
        return JustNameIt(model, disguise_as)
    elif method_name == "vibe_based_disguise":
        return VibeBasedDisguise(model, disguise_as, disguise_df=disguise_df)
    elif method_name == "stylistic_clustering":
        return FeatureClustering(model, disguise_as, num_samples_per_disguise=5, method='stylistic', sample_at_init=True, disguise_df=disguise_df)
    elif method_name == "vibe_clustering":
        return FeatureClustering(model, disguise_as, num_samples_per_disguise=5, method='vibe', sample_at_init=True, disguise_df=disguise_df)
    elif method_name == "stylistic_clustering_resample":
        return FeatureClustering(model, disguise_as, num_samples_per_disguise=5, method='stylistic', sample_at_init=False, disguise_df=disguise_df)
    elif method_name == "embedding_clustering":
        return FeatureClustering(model, disguise_as, num_samples_per_disguise=5, method='embedding', sample_at_init=True, disguise_df=disguise_df)
    elif method_name == "vibe_based_disguise_one_sided":
        return VibeBasedDisguiseOneSided(model, disguise_as, disguise_df=disguise_df)
    else:
        raise ValueError(f"Method {method_name} not found")