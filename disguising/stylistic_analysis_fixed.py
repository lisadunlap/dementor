


import csv
import re
from rich.table import Table
from rich.console import Console
import tqdm

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

    console = Console()
    console.print(table)

# ----------------------------
# Run it!
# ----------------------------
# Make sure response_35, response_4o, response_35_repr are defined dicts
# Example usage:
# run_all_analyses(response_35, response_4o, response_35_repr)


# Load the 3.5 vs. 4o-mini results:
response_35 = {}
response_4o = {}
response_35_repr = {}
with open ('/home/davidchan/Projects/dementor/disguising/old_comparison_results.csv', 'r') as f:
    # Columns are prompt,gpt35_response,gpt4omini_response,comparison_results
    for row in csv.DictReader(f):
        response_35[row['prompt']] = row['gpt35_response']
        response_4o[row['prompt']] = row['gpt4omini_response']

with open('/home/davidchan/Projects/dementor/disguising/new_comparison_results.csv', 'r') as f:
    # Columns are prompt,gpt35_reprompted,gpt4omini_response,comparison_results
    for row in csv.DictReader(f):
        response_35_repr[row['prompt']] = row['gpt35_reprompted']
        response_4o[row['prompt']] = row['gpt4omini_response']


# Run the analysis
run_all_analyses(response_35, response_4o, response_35_repr)
