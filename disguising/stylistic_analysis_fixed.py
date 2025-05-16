import csv
import re
import numpy as np
import tqdm
from rich.table import Table
from rich.console import Console

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
        r'`{1,3}.*?\`{1,3}', r'<code>.*?</code>', r'```[\s\S]*?```'
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

def uses_questions_at_end(text):
    return bool(re.search(r'\?\s*$', text.strip()))

def uses_passive_voice(text):
    return bool(re.search(r'\b(is|was|were|are|been|being)\s+\w+ed\b', text, re.IGNORECASE))

def uses_contractions(text):
    return bool(re.search(r"\b(can't|won't|it's|don't|I'm|you're|they're|we're|isn't|aren't|didn't|wasn't|couldn't|wouldn't|shouldn't)\b", text, re.IGNORECASE))

def has_inline_html(text):
    return bool(re.search(r'<[^>]+>', text))

def uses_enumerated_sections(text):
    return bool(re.search(r'(Section|Part|Step)\s+\d+', text, re.IGNORECASE))

def has_ascii_art_or_tables(text):
    return bool(re.search(r'(\+[-=]+\+)|(\|[^\n]+\|)', text))

def uses_em_dashes(text):
    return '—' in text or '--' in text

def uses_parentheticals_with_emphasis(text):
    return bool(re.search(r'\(.*?[!?].*?\)', text))

def uses_definitions_or_glosses(text):
    return bool(re.search(r'".+?"\s+means|\brefers to\b|\bdefined as\b', text, re.IGNORECASE))

def uses_ellipsis(text):
    return '...' in text

# ----------------------------
# Analysis Helper Functions
# ----------------------------
def analyze_style(feature_func, responses):
    # Compute a list of booleans (one per prompt)
    return [feature_func(r) for r in responses.values()]

def compute_percentage(matches):
    return (sum(matches) / len(matches)) * 100 if matches else 0

# ----------------------------
# Additional Alignment Metrics
# ----------------------------
def cohen_kappa(rater1, rater2):
    """
    Compute Cohen's kappa for two raters (lists of booleans).
    """
    n = len(rater1)
    # Convert booleans to integers (0 or 1)
    r1 = [int(x) for x in rater1]
    r2 = [int(x) for x in rater2]
    observed = sum(1 for a, b in zip(r1, r2) if a == b) / n
    p_a = sum(r1) / n
    p_b = sum(r2) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return 1.0
    kappa = (observed - expected) / (1 - expected)
    return kappa

def cronbach_alpha(data):
    """
    Compute Cronbach's alpha given a 2D NumPy array of shape (subjects, items).
    Here, subjects are prompts and items are the scores (e.g. from different models).
    """
    # Number of items (columns, e.g. raters)
    k = data.shape[1]
    variances = np.var(data, axis=0, ddof=1)
    total_scores = np.sum(data, axis=1)
    var_total = np.var(total_scores, ddof=1)
    if var_total == 0:
        return 1.0
    alpha = (k / (k - 1)) * (1 - np.sum(variances) / var_total)
    return alpha

# ----------------------------
# Original Analysis (Table of Style Percentages and Differences)
# ----------------------------
def run_all_analyses(response_35, response_4o, response_35_repr):
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
        "First-Person Pronouns": uses_first_person,
        "Question at End": uses_questions_at_end,
        "Passive Voice": uses_passive_voice,
        "Contractions": uses_contractions,
        "Inline HTML": has_inline_html,
        "Enumerated Sections": uses_enumerated_sections,
        "ASCII Art / Tables": has_ascii_art_or_tables,
        "Em Dashes": uses_em_dashes,
        "Emphatic Parentheticals": uses_parentheticals_with_emphasis,
        "Definitions/Glosses": uses_definitions_or_glosses,
        "Ellipses": uses_ellipsis
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

        # Compare the distance to GPT-4o
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

    Console().print(table)

# ----------------------------
# New Alignment Analysis: Cohen's Kappa and Cronbach's Alpha
# ----------------------------
def run_alignment_analysis(response_35, response_4o, response_35_repr):
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
        "First-Person Pronouns": uses_first_person,
        "Question at End": uses_questions_at_end,
        "Passive Voice": uses_passive_voice,
        "Contractions": uses_contractions,
        "Inline HTML": has_inline_html,
        "Enumerated Sections": uses_enumerated_sections,
        "ASCII Art / Tables": has_ascii_art_or_tables,
        "Em Dashes": uses_em_dashes,
        "Emphatic Parentheticals": uses_parentheticals_with_emphasis,
        "Definitions/Glosses": uses_definitions_or_glosses,
        "Ellipses": uses_ellipsis
    }

    # Table for Cohen's Kappa per stylistic feature.
    kappa_table = Table(title="Cohen's Kappa for Stylistic Features")
    kappa_table.add_column("Feature", style="bold cyan")
    kappa_table.add_column("GPT-3.5 vs GPT-4o", justify="right")
    kappa_table.add_column("GPT-3.5 Repr vs GPT-4o", justify="right")
    kappa_table.add_column("GPT-3.5 vs GPT-3.5 Repr", justify="right")

    # Prepare accumulators for overall averages.
    sum_35_4o = 0.0
    sum_35repr_4o = 0.0
    sum_35_35repr = 0.0
    n_features = len(style_functions)

    # Prepare for overall aggregated scores (for Cronbach's Alpha).
    aggregated_scores_35 = {}
    aggregated_scores_4o = {}
    aggregated_scores_35_repr = {}

    for feature, func in tqdm.tqdm(style_functions.items()):
        f_35 = analyze_style(func, response_35)
        f_4o = analyze_style(func, response_4o)
        f_35_repr = analyze_style(func, response_35_repr)

        kappa_35_4o = cohen_kappa(f_35, f_4o)
        kappa_35repr_4o = cohen_kappa(f_35_repr, f_4o)
        kappa_35_35repr = cohen_kappa(f_35, f_35_repr)

        # Accumulate kappas for the overall row.
        sum_35_4o += kappa_35_4o
        sum_35repr_4o += kappa_35repr_4o
        sum_35_35repr += kappa_35_35repr

        # Color the row green when kappa for GPT-3.5 repr vs. GPT-4o exceeds GPT-3.5 vs. GPT-4o.
        row_style = "green" if kappa_35repr_4o > kappa_35_4o else "red"

        kappa_table.add_row(
            feature,
            f"{kappa_35_4o:.2f}",
            f"{kappa_35repr_4o:.2f}",
            f"{kappa_35_35repr:.2f}",
            style=row_style
        )

        # For overall aggregated style scores, increment counts per prompt.
        for prompt, value in zip(response_35.keys(), f_35):
            aggregated_scores_35.setdefault(prompt, 0)
            aggregated_scores_35[prompt] += int(value)
        for prompt, value in zip(response_4o.keys(), f_4o):
            aggregated_scores_4o.setdefault(prompt, 0)
            aggregated_scores_4o[prompt] += int(value)
        for prompt, value in zip(response_35_repr.keys(), f_35_repr):
            aggregated_scores_35_repr.setdefault(prompt, 0)
            aggregated_scores_35_repr[prompt] += int(value)

    # Compute overall (equally weighted) averages.
    overall_35_4o = sum_35_4o / n_features
    overall_35repr_4o = sum_35repr_4o / n_features
    overall_35_35repr = sum_35_35repr / n_features

    # Decide the overall row color based on the condition.
    overall_row_style = "green" if overall_35repr_4o > overall_35_4o else "red"

    # Add an overall row to the bottom of the table.
    kappa_table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{overall_35_4o:.2f}[/bold]",
        f"[bold]{overall_35repr_4o:.2f}[/bold]",
        f"[bold]{overall_35_35repr:.2f}[/bold]",
        style=overall_row_style
    )

    Console().print(kappa_table)

    # Now, compute Cronbach's alpha on the aggregated style scores.
    all_prompts = list(response_35.keys())
    score_matrix = []
    for prompt in all_prompts:
        score_35 = aggregated_scores_35.get(prompt, 0)
        score_4o = aggregated_scores_4o.get(prompt, 0)
        score_35_repr = aggregated_scores_35_repr.get(prompt, 0)
        score_matrix.append([score_35, score_4o, score_35_repr])
    score_matrix = np.array(score_matrix)

    overall_alpha = cronbach_alpha(score_matrix)
    Console().print(f"[bold magenta]Overall Cronbach's Alpha (across aggregated style scores): {overall_alpha:.3f}[/bold magenta]")


# ----------------------------
# Load Data from CSV Files
# ----------------------------
response_35 = {}
response_4o = {}
response_35_repr = {}

with open('/home/davidchan/Projects/dementor/disguising/old_comparison_results.csv', 'r') as f:
    # CSV columns: prompt,gpt35_response,gpt4omini_response,comparison_results
    for row in csv.DictReader(f):
        response_35[row['prompt']] = row['gpt35_response']
        response_4o[row['prompt']] = row['gpt4omini_response']

with open('/home/davidchan/Projects/dementor/disguising/new_comparison_results.csv', 'r') as f:
    # CSV columns: prompt,gpt35_reprompted,gpt4omini_response,comparison_results
    for row in csv.DictReader(f):
        response_35_repr[row['prompt']] = row['gpt35_reprompted']
        # Overwrite or reconfirm GPT-4o responses as needed:
        response_4o[row['prompt']] = row['gpt4omini_response']

# ----------------------------
# Run the Analyses
# ----------------------------
run_all_analyses(response_35, response_4o, response_35_repr)
run_alignment_analysis(response_35, response_4o, response_35_repr)
