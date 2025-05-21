"""
Helper functions for museum topic analysis app.

This module contains text cleaning, visualization, and modeling helpers
for use in the Streamlit interface.
"""

import re
from typing import List, Tuple, Dict
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud


def clean_text(text: str) -> str:
    """
    Cleans text by removing punctuation, digits, and extra whitespace.

    Args:
        text: Raw input string.

    Returns:
        A cleaned and lowercased string.
    """
    text = re.sub(r"[-+&]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return text.lower().strip()

def get_custom_stop_words(additional: set[str] = None) -> set[str]:
    """
    Return a consistent set of stop words for topic modeling and word clouds.

    Args:
        additional (set[str], optional): Additional stop words to include.

    Returns:
        set[str]: A unified set of stop words.
    """
    base_stop_words = set(ENGLISH_STOP_WORDS)
    project_specific = {"somewhat", "very", "deeply", "moved", "not at all", "s", "felt", "experienced"}
    combined = base_stop_words.union(project_specific)
    if additional:
        combined = combined.union(additional)
    return combined

def generate_wordcloud(text: str, stopwords: set = None) -> plt.Figure:
    """
    Generates a matplotlib word cloud figure.

    Args:
        text: Concatenated input text.
        stopwords: Optional set of stopwords to remove.

    Returns:
        A matplotlib Figure object.
    """
    if stopwords is None:
        stopwords = set()

    words = " ".join(
        [word for word in text.lower().split() if word not in stopwords]
    )
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


def get_top_word_frequencies(text: str, n: int = 20, stop_words: set[str] = None) -> Tuple[List[str], List[int]]:
    """
    Computes top N word frequencies from cleaned text, excluding stop words.

    Args:
        text: Preprocessed text string.
        n: Number of top words to return.
        stop_words: Set of stop words to exclude.

    Returns:
        A tuple of (words, frequencies) lists.
    """
    cleaned = clean_text(text)
    words = cleaned.split()

    if stop_words:
        words = [word for word in words if word not in stop_words]

    freq = Counter(words)
    top_words = freq.most_common(n)

    if not top_words:
        return [], []

    words, counts = zip(*top_words)
    return list(words), list(counts)


def plot_word_frequencies(words: List[str], counts: List[int]) -> plt.Figure:
    """
    Creates a horizontal bar plot of word frequencies.

    Args:
        words: List of words.
        counts: Corresponding frequency counts.

    Returns:
        A matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette="viridis", ax=ax)
    ax.set_title(f"Top {len(words)} Words in Combined Responses")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Words")
    return fig
