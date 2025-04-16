import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stylistic_analysis_fixed import *
import argparse
from kmodes.kmodes import KModes

def extract_style_features(text):
    style_functions = {
        "Markdown": has_markdown,
        "List": contains_list,
        "Header/Title": contains_header,
        "Code Formatting": contains_code,
        "Links": contains_link,
        "Greeting (Start)": starts_with_greeting,
        "Sign-off (End)": ends_with_signoff,
        "Emojis": contains_emoji,
        "Bullets": contains_bullets,
        "Questions": contains_question,
        "Parentheses": uses_parentheses,
        "Exclamations": contains_exclamation,
        "Long Sentences": has_long_sentences,
        "Starts with List": starts_with_list,
        "Math Symbols": contains_math_symbols,
        "Blockquotes": contains_blockquote,
        "Repetition": contains_repetition,
        "Numbered Steps": contains_numbered_steps,
        "ALL CAPS": contains_all_caps,
        "First-Person Pronouns": uses_first_person
    }

    return np.array([1 if func(text) else 0 for func in style_functions.values()])

def kmodes_clustering(features, n_clusters=3):
    km = KModes(n_clusters, init='Huang', n_init=5, verbose=1)
    cluster_labels = km.fit_predict(features)
    centroids = km.cluster_centroids_
    return cluster_labels, centroids

def sample_indices_from_clusters(cluster_labels, n_samples):
    unique_clusters = np.unique(cluster_labels)
    
    cluster_to_idx = {}
    for cluster in unique_clusters:
        cluster_indices = np.nonzero(cluster_labels == cluster)[0]
        sample_indices = np.random.choice(cluster_indices, size=n_samples, replace=False)
        cluster_to_idx[cluster] = sample_indices
        
    return cluster_to_idx

def main(args):
    df = pd.read_csv(args.data)
    responses = df[args.col_name]

    # Extract style features    
    style_features = []
    for response in responses:
        if not pd.isna(response):
            features = extract_style_features(response)
            style_features.append(features)

    style_features = np.array(style_features)

    # Cluster
    if args.method == 'kmodes':
        cluster_labels, centroids = kmodes_clustering(style_features, args.n_clusters)
        cluster_to_idx = sample_indices_from_clusters(cluster_labels, args.n_samples)

    # Save samples to CSV
    clusters = []
    prompts = []
    responses = []
    for cluster, indices in cluster_to_idx.items():
        rows = df.iloc[indices]
        r = rows[args.col_name].tolist()
        p = rows['prompt'].tolist()
        clusters.extend([cluster] * len(r))
        prompts.extend(p)
        responses.extend(r)

    samples_df = pd.DataFrame({'Cluster': clusters, 'Prompt': prompts, 'Response': responses})
    samples_df.to_csv(args.output, index=False)
    print(f'Samples saved to {args.output}')



if __name__ == "__main__":
    '''
    python clustering.py --method kmodes --n_clusters 5 --n_samples 2 --data old_comparison_results.csv --col_name gpt4omini_response --output gpt4omini_clustering.csv
    '''
    parser = argparse.ArgumentParser(description='Cluster sampling')
    parser.add_argument('--method', type=str, default='kmodes', choices=['kmodes'],
                      help='Clustering method to use (default: kmodes)')
    parser.add_argument('--n_clusters', type=int, default=5,
                      help='Number of clusters (default: 5)')
    parser.add_argument('--n_samples', type=int, default=1,
                      help='Number of samples per cluster (default: 1)')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to input data file')
    parser.add_argument('--col_name', type=str, required=True,
                      help='Model response column name')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output data file')
    
    args = parser.parse_args()
    
    main(args)
