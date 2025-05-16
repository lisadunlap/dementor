
import pandas as pd
from methods.base import MethodBase
from utils import get_token_count
from methods.utils.extract_styles import extract_style_features
from methods.utils.clustering import kmodes_clustering, get_cluster_to_idx, sample_from_clusters
import numpy as np
import os

class FeatureClustering(MethodBase):
    """
    Disguise the prompt by clustering the base model's responses by stylistic features and then sampling from the clusters.
    """
    def __init__(self, model: str, disguise_as: str, num_samples: int = 1000, num_samples_per_disguise: int = 5, seed: int = None,
                 method: str = 'stylistic', sample_at_init: bool = True, save_clusters: bool = True, disguise_df: pd.DataFrame = None) -> None:
        """
        Num_samples is the number of samples to use from the base model, used to read in the responses from the model-responses/base folder.
        Num_samples_per_disguise is the number of samples to use for each disguise.
        Seed is the seed to use for the random sampling, if it is None, then no seed is used and the samples will be different each time.
        Method is the method to use for clustering. Options: stylistic, vibes.
        Sample_at_init is a true if we sample the in-context examples from the clusters at initialization. If false, the examples are sampled at each disguise.
        Save_clusters is a true if we save the clusters to a csv file.
        """
        super().__init__(model, disguise_as)
        self.num_samples = num_samples
        self.num_samples_per_disguise = num_samples_per_disguise
        self.seed = seed
        self.method = method
        self.sample_at_init = sample_at_init
        self.save_clusters = save_clusters
        self.disguise_df = disguise_df
        self.disguise_df["token_length"] = self.disguise_df["model_response"].apply(lambda x: get_token_count(x))
        # truncate the responses to 256 tokens
        self.disguise_df["model_response"] = self.disguise_df["model_response"].apply(lambda x: f"{x[:256]}...(truncated)" if get_token_count(x) > 256 else x)
        
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
        elif self.method == 'vibes':
            save_dir = f"disguising/model-responses/clusters/vibes"
            filename = f"{self.model.replace('/', '_')}_disguised-{self.disguise_as.replace('/', '_')}_clusters.csv"
        else:
            raise ValueError(f"Method {self.method} not supported")

        # Return existing clusters if they exist
        if os.path.exists(f"{save_dir}/{filename}"):
            print(f"Loading existing clusters from {save_dir}/{filename}")
            return pd.read_csv(f"{save_dir}/{filename}")
        
        # Do clustering
        responses = self.disguise_df["model_response"]
        if self.method == 'stylistic':
            # Extract style features    
            style_features = []
            for response in responses:
                features = extract_style_features(response)
                style_features.append(features)

            style_features = np.array(style_features)
            cluster_labels, _ = kmodes_clustering(style_features, self.num_samples_per_disguise)
            cluster_to_idx = get_cluster_to_idx(cluster_labels)

        elif self.method == 'vibes':
            pass
        else:
            raise ValueError(f"Method {self.method} not supported")

        clusters = []
        prompts = []
        responses = []
        for cluster, indices in cluster_to_idx.items():
            rows = self.disguise_df.iloc[indices]
            r = rows["model_response"].tolist()
            p = rows['prompt'].tolist()
            clusters.extend([cluster] * len(r))
            prompts.extend(p)
            responses.extend(r)

        samples_df = pd.DataFrame({'cluster': clusters, 'prompt': prompts, 'model_response': responses})

        # Save cluster samples to CSV
        if self.save_clusters:
            os.makedirs(save_dir, exist_ok=True)
            samples_df.to_csv(f"{save_dir}/{filename}", index=False, encoding='utf-8')
            print(f"Clusters saved to {save_dir}/{filename}")

        return samples_df

    def forward(self, prompt: str) -> str:
        """
        Given a prompt, return the disguised prompt to use for the model.
        """
        if self.sample_at_init:
            disguise_prompt = self.make_disguise_prompt(self.sampled_df, prompt)
        else:
            sampled_df = sample_from_clusters(self.clusters_df, self.seed)
            disguise_prompt = self.make_disguise_prompt(sampled_df, prompt)
        return  [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": disguise_prompt}]