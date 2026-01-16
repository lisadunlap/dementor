from __future__ import annotations

from typing import Iterable, List, Optional

import re
import numpy as np


class PosTagger:
    """Convert text to space-delimited POS tag sequences using spaCy."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            import spacy  # type: ignore

            self._nlp = spacy.load(self.model_name, disable=["ner", "lemmatizer"])
        return self._nlp

    def fit(self, _: Iterable[str], __: Optional[Iterable[str]] = None):
        self._load()
        return self

    def transform(self, texts: Iterable[str]) -> List[str]:
        nlp = self._load()
        tagged = []
        for doc in nlp.pipe(texts, batch_size=32):
            tagged.append(" ".join(tok.pos_ for tok in doc))
        return tagged


def normalize_text(
    text: str,
    prompt: Optional[str],
    strip_prompt_echo: bool,
    strip_answer_prefix: bool,
    strip_leading_markers: bool,
    strip_markdown: bool,
    strip_punctuation: bool,
    strip_reasoning_phrases: bool,
    collapse_whitespace: bool,
    lowercase: bool,
    max_chars: Optional[int],
) -> str:
    cleaned = text.strip()

    if strip_prompt_echo and prompt:
        prompt_clean = prompt.strip()
        if prompt_clean and cleaned.startswith(prompt_clean):
            cleaned = cleaned[len(prompt_clean) :].lstrip("\n\r\t ")

    if strip_answer_prefix:
        for prefix in ("Answer:", "Final answer:", "Final Answer:"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip()
                break

    if strip_leading_markers:
        for prefix in (
            "Let's",
            "To",
            "First,",
            "Second,",
            "Third,",
            "Finally,",
            "Thus,",
            "Therefore,",
            "So,",
            "Hence,",
            "In summary,",
            "We can",
            "We",
            "I",
        ):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip(" ,:-\n\t")
                break

    if strip_markdown:
        # Drop fenced code blocks and remove common markdown markers.
        lines = []
        for line in cleaned.splitlines():
            if line.strip().startswith("```"):
                continue
            line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
            line = re.sub(r"^\s{0,3}>\s+", "", line)
            line = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", line)
            lines.append(line)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)

    if strip_punctuation:
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    if strip_reasoning_phrases:
        reasoning_patterns = [
            r"let's break down the problem step by step[:\\s]*",
            r"let's break down the problem[:\\s]*",
            r"let's solve this step by step[:\\s]*",
            r"let's solve step by step[:\\s]*",
            r"step[-\\s]by[-\\s]step[:\\s]*",
            r"to find out[:\\s]*",
            r"to find the[:\\s]*",
            r"to find[:\\s]*",
            r"we need to[:\\s]*",
            r"we need[:\\s]*",
        ]
        for pat in reasoning_patterns:
            cleaned = re.sub(rf"^(?:{pat})", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.lstrip(" ,:-\n\t")

    if collapse_whitespace:
        cleaned = " ".join(cleaned.split())

    if lowercase:
        cleaned = cleaned.lower()

    if max_chars is not None and max_chars > 0:
        cleaned = cleaned[:max_chars]

    return cleaned


def normalize_texts(
    texts: List[str],
    prompts: Optional[List[Optional[str]]],
    config: dict,
) -> List[str]:
    strip_prompt_echo = config.get("strip_prompt_echo", False)
    strip_answer_prefix = config.get("strip_answer_prefix", False)
    strip_leading_markers = config.get("strip_leading_markers", False)
    strip_markdown = config.get("strip_markdown", False)
    strip_punctuation = config.get("strip_punctuation", False)
    strip_reasoning_phrases = config.get("strip_reasoning_phrases", False)
    collapse_whitespace = config.get("collapse_whitespace", False)
    lowercase = config.get("lowercase", False)
    max_chars = config.get("max_chars")

    if prompts is None:
        prompts = [None] * len(texts)

    normalized = []
    for text, prompt in zip(texts, prompts):
        normalized.append(
            normalize_text(
                text,
                prompt,
                strip_prompt_echo,
                strip_answer_prefix,
                strip_leading_markers,
                strip_markdown,
                strip_punctuation,
                strip_reasoning_phrases,
                collapse_whitespace,
                lowercase,
                max_chars,
            )
        )
    return normalized


class FormatFeatures:
    """Extract simple markdown/list structure counts."""

    def fit(self, _: Iterable[str], __: Optional[Iterable[str]] = None):
        return self

    def transform(self, texts: Iterable[str]):
        rows = []
        for text in texts:
            lines = text.splitlines()
            num_lines = max(len(lines), 1)
            line_lengths = [len(line) for line in lines] or [len(text)]
            avg_line_len = sum(line_lengths) / len(line_lengths)
            bullet_count = sum(
                1
                for line in lines
                if re.match(r"^\s*([-*•]|\d+[.)])\s+", line)
            )
            header_count = sum(1 for line in lines if re.match(r"^\s{0,3}#{1,6}\s+", line))
            blockquote_count = sum(1 for line in lines if re.match(r"^\s{0,3}>\s+", line))
            code_fence_count = text.count("```")
            inline_code_count = len(re.findall(r"`[^`]+`", text))
            bold_count = len(re.findall(r"(\\*\\*[^*]+\\*\\*|__[^_]+__)", text))
            italic_count = len(re.findall(r"(\\*[^*]+\\*|_[^_]+_)", text))
            rows.append(
                [
                    num_lines,
                    avg_line_len,
                    bullet_count / num_lines,
                    header_count / num_lines,
                    blockquote_count / num_lines,
                    code_fence_count,
                    inline_code_count,
                    bold_count,
                    italic_count,
                ]
            )
        return np.array(rows, dtype=float)


class StructureFeatures:
    """Extract coarse structural features (length, punctuation, casing)."""

    def fit(self, _: Iterable[str], __: Optional[Iterable[str]] = None):
        return self

    def transform(self, texts: Iterable[str]):
        rows = []
        for text in texts:
            if not text:
                rows.append([0.0] * 7)
                continue
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
            sentence_splits = re.split(r"[.!?]+", text)
            sentence_lens = [len(s.strip()) for s in sentence_splits if s.strip()]
            sent_count = len(sentence_lens)
            avg_sent_len = sum(sentence_lens) / max(sent_count, 1)
            punct_count = len(re.findall(r"[^\w\s]", text))
            digit_count = sum(c.isdigit() for c in text)
            upper_count = sum(c.isupper() for c in text)
            rows.append(
                [
                    word_count,
                    char_count,
                    avg_word_len,
                    sent_count,
                    avg_sent_len,
                    punct_count / max(char_count, 1),
                    digit_count / max(char_count, 1),
                    upper_count / max(char_count, 1),
                ]
            )
        return np.array(rows, dtype=float)


class PosDistributionFeatures:
    """POS tag distribution over a fixed tag list."""

    POS_TAGS = [
        "NOUN",
        "VERB",
        "PRON",
        "ADP",
        "PUNCT",
        "DET",
        "AUX",
        "ADJ",
        "ADV",
        "PART",
        "CCONJ",
        "SCONJ",
        "PROPN",
        "NUM",
        "X",
        "INTJ",
        "SYM",
    ]

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            import spacy  # type: ignore

            self._nlp = spacy.load(self.model_name, disable=["ner", "lemmatizer"])
        return self._nlp

    def fit(self, _: Iterable[str], __: Optional[Iterable[str]] = None):
        self._load()
        return self

    def transform(self, texts: Iterable[str]):
        nlp = self._load()
        rows = []
        for doc in nlp.pipe(texts, batch_size=32):
            counts = {tag: 0 for tag in self.POS_TAGS}
            total = 0
            for tok in doc:
                if tok.pos_ in counts:
                    counts[tok.pos_] += 1
                total += 1
            if total == 0:
                rows.append([0.0 for _ in self.POS_TAGS])
            else:
                rows.append([counts[tag] / total for tag in self.POS_TAGS])
        return np.array(rows, dtype=float)


class ToneFeatures:
    """Extract lightweight discourse tone signals (hedging, politeness, directness)."""

    HEDGES = {
        "maybe",
        "perhaps",
        "likely",
        "possibly",
        "i think",
        "i believe",
        "it seems",
        "it appears",
        "i guess",
        "probably",
    }
    BOOSTERS = {
        "definitely",
        "clearly",
        "certainly",
        "undoubtedly",
        "obviously",
        "surely",
    }
    POLITENESS = {
        "please",
        "thanks",
        "thank you",
        "sorry",
        "apologies",
        "kindly",
    }
    DIRECTIVES = {
        "you should",
        "you must",
        "do this",
        "do that",
        "make sure",
        "need to",
        "must",
        "should",
    }

    def fit(self, _: Iterable[str], __: Optional[Iterable[str]] = None):
        return self

    def _count_phrases(self, text: str, phrases: set[str]) -> int:
        lowered = text.lower()
        return sum(lowered.count(p) for p in phrases)

    def transform(self, texts: Iterable[str]):
        rows = []
        for text in texts:
            word_count = max(len(text.split()), 1)
            hedge = self._count_phrases(text, self.HEDGES)
            boost = self._count_phrases(text, self.BOOSTERS)
            polite = self._count_phrases(text, self.POLITENESS)
            direct = self._count_phrases(text, self.DIRECTIVES)
            rows.append(
                [
                    hedge / word_count,
                    boost / word_count,
                    polite / word_count,
                    direct / word_count,
                ]
            )
        return np.array(rows, dtype=float)
