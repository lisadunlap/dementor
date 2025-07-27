
import pandas as pd
from methods.base import MethodBase
from utils import get_token_count
from methods.utils.extract_styles import extract_style_features, get_model_embeddings
from methods.utils.clustering import kmodes_clustering, get_cluster_to_idx, sample_from_clusters, kmeans_clustering
import numpy as np
import os
from methods.utils.extract_vibe_features import extract_vibe_features

class FeatureClustering(MethodBase):
    """
    Disguise the prompt by clustering the base model's responses by stylistic features and then sampling from the clusters.
    """
    def __init__(self, model: str, disguise_as: str, num_samples_per_disguise: int = 5, seed: int = None,
                 method: str = 'stylistic', sample_at_init: bool = True, disguise_df: pd.DataFrame = None) -> None:
        """
        Num_samples_per_disguise is the number of samples to use for each disguise.
        Seed is the seed to use for the random sampling, if it is None, then no seed is used and the samples will be different each time.
        Method is the method to use for clustering. Options: stylistic, vibes.
        Sample_at_init is a true if we sample the in-context examples from the clusters at initialization. If false, the examples are sampled at each disguise.
        """
        super().__init__(model, disguise_as)
        self.num_samples = len(disguise_df)
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.method = method
        self.sample_at_init = sample_at_init
        self.disguise_df = disguise_df.copy()
        self.disguise_df["token_length"] = self.disguise_df["target_response"].apply(lambda x: get_token_count(x))
        # truncate the responses to 256 tokens
        # self.disguise_df["target_response"] = self.disguise_df["target_response"].apply(lambda x: f"{x[:256]}...(truncated)" if get_token_count(x) > 256 else x)
        
        # initialize clusters
        self.clusters_df = self.init_clusters()
        if self.sample_at_init:
            self.sampled_df = sample_from_clusters(self.clusters_df, self.seed)

    def init_clusters(self):
        """
        Initialize the clusters by clustering the responses by stylistic features.
        self.num_samples_per_disguise is the number of clusters. We sample 1 response from each cluster.
        """
        # Check if clusters already exist
        if self.method == 'stylistic':
            save_dir = f"disguising/model-responses/clusters/stylistic"
            filename = f"{self.disguise_as.replace('/', '_')}_clusters-{self.num_samples}.csv" # stylistic clusters are only based on the disguise_as model
        elif self.method == 'vibe':
            save_dir = f"disguising/model-responses/clusters/vibe/{self.model.replace('/', '_')}"
            filename = f"{self.model.replace('/', '_')}_disguised-{self.disguise_as.replace('/', '_')}_clusters-{self.num_samples}.csv"
            axes_filename = f"{self.model.replace('/', '_')}_disguised-{self.disguise_as.replace('/', '_')}_axes-{self.num_samples}.json"
            ratings_filename = f"{self.model.replace('/', '_')}_disguised-{self.disguise_as.replace('/', '_')}_ratings-{self.num_samples}.csv"
        elif self.method == 'embedding':
            save_dir = f"disguising/model-responses/clusters/embedding/{self.model.replace('/', '_')}"
            filename = f"{self.model.replace('/', '_')}_disguised-{self.disguise_as.replace('/', '_')}_clusters-{self.num_samples}.csv"
        else:
            raise ValueError(f"Method {self.method} not supported")

        # Return existing clusters if they exist
        if os.path.exists(f"{save_dir}/{filename}"):
            print(f"Loading existing clusters from {save_dir}/{filename}")
            return pd.read_csv(f"{save_dir}/{filename}")
        
        os.makedirs(save_dir, exist_ok=True)

        # Do clustering
        responses_source = self.disguise_df["source_response"]
        responses_target = self.disguise_df["target_response"]
        responses_source = responses_source.apply(lambda x: f"{x[:256]}...(truncated)" if get_token_count(x) > 256 else x)
        responses_target = responses_target.apply(lambda x: f"{x[:256]}...(truncated)" if get_token_count(x) > 256 else x)
        if self.method == 'stylistic':
            # Extract style features    
            style_features = []
            for response in responses_target:
                features = extract_style_features(response)
                style_features.append(features)
            style_features = np.array(style_features)
            # Use kmodes clustering for categorical features
            cluster_labels, _ = kmodes_clustering(style_features, self.num_samples_per_disguise)
            cluster_to_idx = get_cluster_to_idx(cluster_labels)
        elif self.method == 'vibe':
            # Extract vibe features (based on most distinctive axes between source and target models)
            vibe_features, response_indices = extract_vibe_features(responses_source, responses_target, n_samples=10, n_axes=10, path_to_axes=f"{save_dir}/{axes_filename}", path_to_ratings=f"{save_dir}/{ratings_filename}")
            vibe_features = np.array(vibe_features)
            cluster_labels, _ = kmeans_clustering(vibe_features, self.num_samples_per_disguise)
            cluster_to_idx = get_cluster_to_idx(cluster_labels, response_indices)
        elif self.method == 'embedding':
            responses_source = self.disguise_df["source_response"]
            responses_target = self.disguise_df["target_response"]
            embeddings_source = get_model_embeddings(responses_source)
            embeddings_target = get_model_embeddings(responses_target)
            embeddings_diff = embeddings_source - embeddings_target
            epsilon = 1e-8 # to avoid division by zero
            embeddings_diff_norm = embeddings_diff / (np.linalg.norm(embeddings_diff, axis=1, keepdims=True) + epsilon)
            cluster_labels, _ = kmeans_clustering(embeddings_diff_norm, self.num_samples_per_disguise)
            cluster_to_idx = get_cluster_to_idx(cluster_labels)
        else:
            raise ValueError(f"Method {self.method} not supported")

        clusters = []
        prompts = []
        responses = []
        for cluster, indices in cluster_to_idx.items():
            rows = self.disguise_df.iloc[indices]
            r = rows["target_response"].tolist()
            p = rows['prompt'].tolist()
            clusters.extend([cluster] * len(r))
            prompts.extend(p)
            responses.extend(r)

        samples_df = pd.DataFrame({'cluster': clusters, 'prompt': prompts, 'target_response': responses})

        # Save cluster samples to CSV
        samples_df.to_csv(f"{save_dir}/{filename}", index=False, encoding='utf-8')
        print(f"Clusters saved to {save_dir}/{filename}")

        return samples_df

    def forward(self, prompt: str) -> str:
        """
        Given a prompt, return the disguised prompt to use for the model.
        """
        if self.sample_at_init:
            self.sampled_df["token_length"] = self.sampled_df["target_response"].apply(lambda x: get_token_count(x))
            disguise_prompt = self.make_disguise_prompt(self.sampled_df, prompt)
        else:
            sampled_df = sample_from_clusters(self.clusters_df, self.seed)
            sampled_df["token_length"] = sampled_df["target_response"].apply(lambda x: get_token_count(x))
            disguise_prompt = self.make_disguise_prompt(sampled_df, prompt)
        return  [{"role": "system", "content": disguise_prompt}, {"role": "user", "content": prompt}]
        