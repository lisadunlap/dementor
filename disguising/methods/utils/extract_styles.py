import numpy as np
import re
from tqdm import tqdm
from litellm import embedding
from typing import List

def get_model_embeddings(embeddings: List[str]) -> np.ndarray:
    """
    Get the embeddings for the source and target model's responses using litellm
    """
    embeddings_list = []
    for row in tqdm(embeddings, desc="Getting embeddings"):
        embedding_vec = embedding(
            model="text-embedding-3-small",
            input=row,
            caching=True,
        )
        embedding_vec = embedding_vec["data"][0]["embedding"]
        embeddings_list.append(embedding_vec)
    return np.array(embeddings_list)

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