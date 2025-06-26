import numpy as np
from kmodes.kmodes import KModes
import pandas as pd
from sklearn.cluster import KMeans

def kmeans_clustering(features, n_clusters=5, n_init=5):
    km = KMeans(n_clusters, n_init=n_init, verbose=1)
    cluster_labels = km.fit_predict(features)
    centroids = km.cluster_centers_
    return cluster_labels, centroids

def kmodes_clustering(features, n_clusters=5, n_init=5):
    km = KModes(n_clusters, init='Huang', n_init=n_init, verbose=1)
    cluster_labels = km.fit_predict(features)
    centroids = km.cluster_centroids_
    return cluster_labels, centroids

def get_cluster_to_idx(cluster_labels, response_indices = None):
    '''
    Returns a dictionary mapping each cluster index to a list of response indices.
    If response_indices is provided, the indices are the indices of the target responses successfully rated (e.g. vibes clustering).
    '''
    if response_indices is not None:
        assert len(response_indices) == len(cluster_labels), "Number of response indices must match number of cluster labels"

    unique_clusters = np.unique(cluster_labels)
    
    cluster_to_idx = {}
    for cluster in unique_clusters:
        cluster_indices = np.nonzero(cluster_labels == cluster)[0]
        if response_indices is not None:
            cluster_to_idx[cluster] = np.array(response_indices)[cluster_indices].tolist()
        else:
            cluster_to_idx[cluster] = cluster_indices
        
    return cluster_to_idx

def sample_from_clusters(df_clusters, seed, n_samples=1):
    '''
    Returns a df with columns 'cluster', 'row_index', 'prompt', 'target_response'
    '''
    unique_clusters = df_clusters['cluster'].unique()
    
    sampled_rows = []
    for cluster in unique_clusters:
        cluster_rows = df_clusters[df_clusters['cluster'] == cluster]
        sampled_row = cluster_rows.sample(n=n_samples, random_state=seed).iloc[0]
        sampled_rows.append({
            'cluster': cluster,
            'row_index': int(sampled_row.name),
            'prompt': sampled_row['prompt'],
            'target_response': sampled_row['target_response']
        })
    sampled_df = pd.DataFrame(sampled_rows)
    return sampled_df