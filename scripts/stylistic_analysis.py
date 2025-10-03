import csv
import re
from rich.table import Table
from rich.console import Console
import tqdm
import numpy as np
import wandb
import pandas as pd

# ----------------------------
# Style Detection Functions
# ----------------------------

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


# ----------------------------
# Analysis + Table
# ----------------------------

def analyze_style(feature_func, responses):
    return [feature_func(r) for r in responses.values()]

def compute_percentage(matches):
    return (sum(matches) / len(matches)) * 100 if matches else 0

def run_all_analyses(response_35, response_4o, response_35_repr, scores_35=None, scores_4o=None, scores_35_repr=None):
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

    table = Table(title="Stylistic Feature Analysis")
    table.add_column("Feature", style="bold cyan")
    table.add_column("GPT-3.5 (%)", justify="right")
    table.add_column("GPT-4o (%)", justify="right")
    table.add_column("GPT-3.5 Repr (%)", justify="right")

    for feature, func in tqdm.tqdm(style_functions.items()):
        matches_35 = analyze_style(func, response_35)
        matches_4o = analyze_style(func, response_4o)
        matches_35_repr = analyze_style(func, response_35_repr)

        val_35 = compute_percentage(matches_35)
        val_4o = compute_percentage(matches_4o)
        val_35_repr = compute_percentage(matches_35_repr)

        # Compare distance to 4o
        dist_original = abs(val_35 - val_4o)
        dist_repr = abs(val_35_repr - val_4o)

        row_style = "green" if dist_repr < dist_original else "red"

        table.add_row(
            feature,
            f"{val_35:.2f}",
            f"{val_4o:.2f}",
            f"{val_35_repr:.2f}",
            style=row_style
        )

    # Add LLM scoring metrics if available
    if scores_35 and scores_4o and scores_35_repr:
        # Add semantic scores
        table.add_row(
            "Semantic Score (LLM)",
            f"{np.mean(scores_35['semantic']):.2f}",
            f"{np.mean(scores_4o['semantic']):.2f}",
            f"{np.mean(scores_35_repr['semantic']):.2f}",
            style="bold yellow"
        )
        
        # Add stylistic scores
        table.add_row(
            "Stylistic Score (LLM)",
            f"{np.mean(scores_35['stylistic']):.2f}",
            f"{np.mean(scores_4o['stylistic']):.2f}",
            f"{np.mean(scores_35_repr['stylistic']):.2f}",
            style="bold yellow"
        )
        
        # Add similarity scores
        table.add_row(
            "Similarity Score (LLM)",
            f"{np.mean(scores_35['similarity']):.2f}",
            f"{np.mean(scores_4o['similarity']):.2f}",
            f"{np.mean(scores_35_repr['similarity']):.2f}",
            style="bold yellow"
        )

    # Add aggregated style similarity scores
    for model in ["GPT-3.5", "GPT-4o", "GPT-3.5 Repr"]:
        feature_values[model] = np.array(feature_values[model])
    
    # Compute style similarity as 1 - normalized L2 distance
    style_similarity_35 = 1 - np.linalg.norm(feature_values["GPT-3.5"] - feature_values["GPT-4o"]) / np.linalg.norm(feature_values["GPT-4o"])
    style_similarity_35_repr = 1 - np.linalg.norm(feature_values["GPT-3.5 Repr"] - feature_values["GPT-4o"]) / np.linalg.norm(feature_values["GPT-4o"])
    
    table.add_row(
        "Aggregated Style Similarity",
        f"{style_similarity_35:.2f}",
        "1.00",
        f"{style_similarity_35_repr:.2f}",
        style="bold green" if style_similarity_35_repr > style_similarity_35 else "bold red"
    )

    console = Console()
    console.print(table)

def compute_heuristics(responses1, responses2):
    """
    Compute heuristic-based style differences between two lists of responses.
    Returns a DataFrame with boolean matches for each response pair.
    
    Args:
        responses1: List of first responses
        responses2: List of second responses (must be same length as responses1)
        
    Returns:
        DataFrame with one row per response pair and boolean columns for each style feature
    """
    assert len(responses1) == len(responses2), "Response lists must be the same length"
    
    style_functions = {
        "markdown": has_markdown,
        "list": contains_list,
        "header": contains_header,
        "code": contains_code,
        "links": contains_link,
        "greeting": starts_with_greeting,
        "signoff": ends_with_signoff,
        "emojis": contains_emoji,
        "bullets": contains_bullets,
        "questions": contains_question,
        "parentheses": uses_parentheses,
        "exclamations": contains_exclamation,
        "long_sentences": has_long_sentences,
        "starts_list": starts_with_list,
        "math_symbols": contains_math_symbols,
        "blockquotes": contains_blockquote,
        "repetition": contains_repetition,
        "numbered_steps": contains_numbered_steps,
        "all_caps": contains_all_caps,
        "first_person": uses_first_person
    }

    # Initialize results for each response pair
    results = []
    
    # Compute features for each pair
    for resp1, resp2 in zip(responses1, responses2):
        pair_result = {}
        for feature_name, feature_func in style_functions.items():
            resp1_has = feature_func(resp1)
            resp2_has = feature_func(resp2)
            # Store boolean match (True if both have feature or both don't have it)
            pair_result[feature_name] = resp1_has == resp2_has
        results.append(pair_result)

    return pd.DataFrame(results)

# ----------------------------
# Run it!
# ----------------------------

# # Load the 3.5 vs. 4o-mini results:
# response_35 = {}
# response_4o = {}
# response_35_repr = {}
# scores_35 = {"semantic": [], "stylistic": [], "similarity": []}
# scores_4o = {"semantic": [], "stylistic": [], "similarity": []}
# scores_35_repr = {"semantic": [], "stylistic": [], "similarity": []}

# with open ('/home/davidchan/Projects/dementor/disguising/old_comparison_results.csv', 'r') as f:
#     # Columns are prompt,gpt35_response,gpt4omini_response,comparison_results
#     for row in csv.DictReader(f):
#         response_35[row['prompt']] = row['gpt35_response']
#         response_4o[row['prompt']] = row['gpt4omini_response']
#         # Parse scores from comparison_results
#         if 'comparison_results' in row:
#             semantic, stylistic = parse_score(row['comparison_results'])
#             if semantic is not None and stylistic is not None:
#                 scores_35['semantic'].append(semantic)
#                 scores_35['stylistic'].append(stylistic)
#                 scores_4o['semantic'].append(semantic)
#                 scores_4o['stylistic'].append(stylistic)

# with open('/home/davidchan/Projects/dementor/disguising/new_comparison_results.csv', 'r') as f:
#     # Columns are prompt,gpt35_reprompted,gpt4omini_response,comparison_results
#     for row in csv.DictReader(f):
#         response_35_repr[row['prompt']] = row['gpt35_reprompted']
#         response_4o[row['prompt']] = row['gpt4omini_response']
#         # Parse scores from comparison_results
#         if 'comparison_results' in row:
#             semantic, stylistic = parse_score(row['comparison_results'])
#             if semantic is not None and stylistic is not None:
#                 scores_35_repr['semantic'].append(semantic)
#                 scores_35_repr['stylistic'].append(stylistic)
#                 scores_4o['semantic'].append(semantic)
#                 scores_4o['stylistic'].append(stylistic)

# def parse_score(score):
#     # get first line of score
#     score = score.split("\n")[0]
    
#     # First try to match the exact format "Scores: X/Y, Z/W"
#     match = re.search(r'Scores:\s*(\d+)/(\d+),\s*(\d+)/(\d+)', score)
#     if match:
#         return int(match.group(1)), int(match.group(3))
    
#     # Try to find two numbers in the format X/Y, Z/W
#     match = re.search(r'(\d+)/(\d+),\s*(\d+)/(\d+)', score)
#     if match:
#         return int(match.group(1)), int(match.group(3))
    
#     # Try to find two numbers separated by comma or slash
#     match = re.search(r'(\d+)[,/]\s*(\d+)', score)
#     if match:
#         return int(match.group(1)), int(match.group(2))
    
#     # If that fails, try to find any two numbers in the line
#     matches = re.findall(r'\b(\d+)\b', score)
#     if len(matches) >= 2:
#         return int(matches[0]), int(matches[1])
    
#     print(f"Failed to parse score from: {score.splitlines()[0]}")
#     return None, None

# # Run the analysis
# run_all_analyses(response_35, response_4o, response_35_repr, scores_35, scores_4o, scores_35_repr)
