# Disguise Methods

## Baseline Methods

### `random_sample_{i}_example` (`i=1,3,5`)
Randomly samples `i` examples from the target model's responses

### `just_name_it`
Simply instructs the model to act like the target model

## Vibe-Based Methods

### `vibe_based_disguise`
Uses GPT-4o to identify the differences in vibes between 2 models responses and use the differencing vibes in system prompt

### `vibe_based_disguise_one_sided`
Uses GPT-4o to identify the vibes in target model responses and use these vibes in system prompt

## Clustering Methods

### `stylistic_clustering`
Clusters responses by stylistic features (length, formatting, etc.)
- **Method**: K-modes clustering on style features (binary, categorical)
- **Parameters**: `num_samples_per_disguise=5`, `sample_at_init=True`

### `stylistic_clustering_resample`
Same as stylistic_clustering but resamples at each forward pass
- **Method**: K-modes clustering on style features
- **Parameters**: `num_samples_per_disguise=5`, `sample_at_init=False`

### `vibe_clustering`
Clusters responses by vibe features (distinctive axes between models): 
(1) GPT-4o-mini identifies 10 vibes that differentiate the 2 models; 
(2) GPT-4o-mini rates every target model response based on the 10 axes of vibes from 1 to 10 (vibe features)
- **Method**: K-means clustering on vibe features
- **Parameters**: `num_samples_per_disguise=5`, `sample_at_init=True`

### `embedding_clustering`
Clusters responses by text embedding differences between source and target, representative sampling based on semantic differences
- **Method**: K-means clustering on embedding differences
- **Parameters**: `num_samples_per_disguise=5`, `sample_at_init=True`