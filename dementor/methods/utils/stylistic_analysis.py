import re
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


def compute_heuristics(responses1, responses2):
    """
    Compute heuristic-based style differences between two lists of responses.
    Returns a DataFrame with boolean matches for each response pair.
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
        "first_person": uses_first_person,
    }

    results = []
    for resp1, resp2 in zip(responses1, responses2):
        pair_result = {}
        for feature_name, feature_func in style_functions.items():
            resp1_has = feature_func(resp1)
            resp2_has = feature_func(resp2)
            pair_result[feature_name] = (resp1_has == resp2_has)
        # Aggregate simple match score
        feature_matches = list(pair_result.values())
        pair_result["heuristic_match_score"] = sum(feature_matches) / len(feature_matches) if feature_matches else 0.0
        results.append(pair_result)
    return pd.DataFrame(results)

