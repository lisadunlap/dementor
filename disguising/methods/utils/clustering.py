import numpy as np
from kmodes.kmodes import KModes
import pandas as pd

def kmodes_clustering(features, n_clusters=5, n_init=5):
    km = KModes(n_clusters, init='Huang', n_init=n_init, verbose=1)
    cluster_labels = km.fit_predict(features)
    centroids = km.cluster_centroids_
    return cluster_labels, centroids

def get_cluster_to_idx(cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    
    cluster_to_idx = {}
    for cluster in unique_clusters:
        cluster_indices = np.nonzero(cluster_labels == cluster)[0]
        cluster_to_idx[cluster] = cluster_indices
        
    return cluster_to_idx

def sample_from_clusters(df_clusters, seed, n_samples=1):
    '''
    Returns a df with columns 'cluster', 'row_index', 'prompt', 'model_response'
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
            'model_response': sampled_row['model_response']
        })
    sampled_df = pd.DataFrame(sampled_rows)
    return sampled_df