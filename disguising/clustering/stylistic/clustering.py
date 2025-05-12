import pandas as pd
import numpy as np
import argparse
from kmodes.kmodes import KModes
import re
import json


def has_markdown(text):
    markdown_patterns = [
        r'\*\*.*?\*\*', r'\*.*?\*', r'\_\_.*?\_\_', r'\_.*?\_',
        r'\#', r'\[.*?\]\(.*?\)', r'\!\[.*?\]\(.*?\)',
        r'\`{1,3}.*?\`{1,3}', r'\>\s', r'\-\s', r'\+\s', r'\*\s'
    ]
    return any(re.search(p, text) for p in markdown_patterns)

def contains_list(text):
    patterns = [
        r'(^|\n)[\-\*\+]\s+', r'(^|\n)\d+\.\s+',
        r'<(ul|ol|li)[^>]*>', r'(^|\n)\s*[•‣◦▪–—-→➡️🔹📌✅➤]\s+',
        r'(^|\n)[a-zA-Z][\.\)]\s+'
    ]
    return any(re.search(p, text) for p in patterns)

def contains_header(text):
    patterns = [
        r'(^|\n)#{1,6}\s', r'<h[1-6][^>]*>', r'^[A-Z][^a-z]{3,}'
    ]
    return any(re.search(p, text) for p in patterns)

def contains_code(text):
    patterns = [
        r'`{1,3}.*?`{1,3}', r'<code>.*?</code>', r'```[\s\S]*?```'
    ]
    return any(re.search(p, text) for p in patterns)

def contains_link(text):
    patterns = [
        r'\[.*?\]\(http.*?\)', r'<a\s+href="[^"]+">', r'https?://[^\s\)]+'
    ]
    return any(re.search(p, text) for p in patterns)

def starts_with_greeting(text):
    return bool(re.match(r'^(hi|hello|hey|greetings|dear)\b', text.strip(), re.IGNORECASE))

def ends_with_signoff(text):
    return bool(re.search(r'(let me know|hope this helps|cheers|best regards|thanks|sincerely)[\.\!]*$', text.strip(), re.IGNORECASE))

def contains_emoji(text):
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]'
    return bool(re.search(emoji_pattern, text))

def contains_bullets(text):
    return bool(re.search(r'(^|\n)\s*[•‣◦▪–—\-*+→➡️🔹📌✅➤]\s+', text))

def contains_question(text):
    return '?' in text

def uses_parentheses(text):
    return bool(re.search(r'\(.*?\)', text))

def contains_exclamation(text):
    return '!' in text

def average_sentence_length(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    total_words = sum(len(s.split()) for s in sentences)
    return total_words / len(sentences)

def has_long_sentences(text, threshold=20):
    return average_sentence_length(text) > threshold

def starts_with_list(text):
    return bool(re.match(r'^\s*[\-\*\+\d\•\‣\◦a-zA-Z][\.\)]?\s+', text.strip()))

def contains_math_symbols(text):
    return bool(re.search(r'\$.*?\$|[=+\-*/<>^%]', text))

def contains_blockquote(text):
    return bool(re.search(r'(^|\n)\s*>\s+', text))

def contains_repetition(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words) != len(set(words)) and len(words) > 10

def contains_numbered_steps(text):
    return bool(re.search(r'(^|\n)\s*\d+\.\s+\w+', text))

def contains_all_caps(text):
    return bool(re.search(r'\b[A-Z]{2,}\b', text))

def uses_first_person(text):
    return bool(re.search(r'\b(I|we|my|our|me|us)\b', text, re.IGNORECASE))

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
        if n_samples == -1:
            sample_indices = cluster_indices
        else:
            sample_indices = np.random.choice(cluster_indices, size=n_samples, replace=False)
        cluster_to_idx[cluster] = sample_indices
        
    return cluster_to_idx

def sample_from_clusters(df_clusters, n_samples=1):
    unique_clusters = df_clusters['Cluster'].unique()
    
    cluster_to_sample = {}
    for cluster in unique_clusters:
        cluster_rows = df_clusters[df_clusters['Cluster'] == cluster]
        sampled_row = cluster_rows.sample(n=n_samples).iloc[0]
        cluster_to_sample[str(cluster)] = {
            'row_index': int(sampled_row.name),
            'prompt': sampled_row['Prompt'], 
            'response': sampled_row['Response']
        }

    return cluster_to_sample

def clean_csv_string(text):
    if not isinstance(text, str):
        return text
    text = text.replace(r'\\"', '"')
    text = text.replace('\\n', '\n') 
    text = text.strip('"')
    text = text.strip()
    return text
    
def main(args):
    df = pd.read_csv(args.data, encoding='utf-8')
    # Filter out rows with NA responses
    df = df[~pd.isna(df[args.col_name])]
    # Clean df
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_csv_string)
    responses = df[args.col_name]

    # Extract style features    
    style_features = []
    for response in responses:
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

    model_name = args.data.split('/')[-1].replace('_responses.csv', '')
    if args.output is None:
        args.output = f'./clusters/{model_name}_clustering.csv'
    if args.output_samples is None:
        args.output_samples = f'./samples/{model_name}_samples.json'

    samples_df.to_csv(args.output, index=False, encoding='utf-8')
    print(f'Clusters saved to {args.output}')

    print(f'Sampling 1 from each cluster from {args.output}...')
    sampled = sample_from_clusters(samples_df)
    with open(args.output_samples, 'w', encoding='utf-8') as f:
        json.dump(sampled, f)
    print(f'Samples saved to {args.output_samples}')


if __name__ == "__main__":
    '''
    python clustering.py --method kmodes --n_clusters 5 --n_samples -1 \
        --data ../../model-responses/gpt-3.5_responses.csv
    '''
    parser = argparse.ArgumentParser(description='Cluster sampling')
    parser.add_argument('--method', type=str, default='kmodes', choices=['kmodes'],
                      help='Clustering method to use (default: kmodes)')
    parser.add_argument('--n_clusters', type=int, default=5,
                      help='Number of clusters (default: 5); -1 for all')
    parser.add_argument('--n_samples', type=int, default=1,
                      help='Number of samples per cluster (default: 1)')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to input data file')
    parser.add_argument('--col_name', type=str, default='model_response',
                      help='Model response column name')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to output data file')
    parser.add_argument('--output_samples', type=str, default=None,
                      help='Path to output samples file')
    
    args = parser.parse_args()
    
    main(args)
