from methods.base import MethodBase, RandomSampleDisguise, JustNameIt, VibeBasedDisguise

def get_method(method_name, model, disguise_as, num_samples=None):
    """
    Get a method from the methods module.
    """
    if method_name == "random_sample_1_example":
        return RandomSampleDisguise(model, disguise_as, num_samples_per_disguise=1, num_samples=num_samples)
    elif method_name == "random_sample_3_examples":
        return RandomSampleDisguise(model, disguise_as, num_samples_per_disguise=3, num_samples=num_samples)
    elif method_name == "random_sample_5_examples":
        return RandomSampleDisguise(model, disguise_as, num_samples_per_disguise=5, num_samples=num_samples)
    elif method_name == "just_name_it":
        return JustNameIt(model, disguise_as)
    elif method_name == "vibe_based_disguise":
        return VibeBasedDisguise(model, disguise_as, num_samples=num_samples)
    else:
        raise ValueError(f"Method {method_name} not found")