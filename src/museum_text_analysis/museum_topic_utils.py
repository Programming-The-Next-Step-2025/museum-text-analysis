"""museum_topic_utils.py

Helper functions for cleaning text, visualizing word distributions,
and supporting topic modeling for the museum response analysis app.

This module includes:
- Text cleaning and stopword custimization
- Word cloud generation
- Bar plot generation
- Frequency analysis and visualization
"""

# Standard library
import re
from collections import Counter
from typing import List, Tuple, Dict

# Third-party
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

def clean_text(text: str) -> str:
    """Cleans text by removing punctuation, digits, and extra whitespace.

    This function is useful for preparing text data for analysis, 
    such as topic modeling or word cloud generation.

    Args:
        text: Raw input string.

    Returns:
        A cleaned and lowercased string with punctuation and digits removed.

    Examples:
        >>> clean_text("Hello, World! 1234")
        'hello world'
        >>> clean_text("Museum & Artifacts - A Journey")
        'museum artifacts journey'
        >>> clean_text("Deeply moved by the experience!")
        'deeply moved by the experience'
    """
    text = re.sub(r"[-+&]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return text.lower().strip()

def get_custom_stop_words(additional: set[str] = None) -> set[str]:
    """Returns a consistent set of stop words for text anlaysis.

    Combines standard English stop words with project-specific terms that should
    be excluded from analyses (e.g., for word clouds or topic modeling). Additional
    stop words can be optionally passed in.

    Args:
        additional: Optional set of additional stop words to include.

    Returns:
        set[str]: A unified set of stop words including base English stop words
        and project-specific stop words.
    
    Examples:
        >>> custom_stop_words = get_custom_stop_words({"exhibition", "museum"})
        >>> print(custom_stop_words)
        {'the', 'and', 'is', 'exhibition', 'museum', ...}
    """
    base_stop_words = set(ENGLISH_STOP_WORDS)
    project_specific = {
        "somewhat", "very", "deeply", "moved", "not at all", "s", "felt", "experienced"
    }
    combined = base_stop_words.union(project_specific)
    if additional:
        combined = combined.union(additional)
    return combined

def generate_wordcloud(text: str, stopwords: set = None) -> plt.Figure:
    """Generate a matplotlib word cloud figure from input text.

    Args:
        text: Concatenated input text.
        stopwords: Optional set of stopwords to remove from word cloud.

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object containing the word cloud.

    Examples:
        >>> text = "museum exhibits are fascinating i felt deeply moved"
        >>> fig = generate_wordcloud(text)
        >>> plt.show(fig)
    """
    if stopwords is None:
        stopwords = set()

    words = " ".join(
        [word for word in text.lower().split() if word not in stopwords]
    )
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="#0e1117", 
        colormap="cividis",
    ).generate(words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


def get_top_word_frequencies(
    text: str, 
    n: int = 20, 
    stop_words: set[str] = None
    ) -> Tuple[List[str], List[int]]:
    """Computes top N word frequencies from cleaned text, excluding stop words.

    This function processes the input text to extract the most common words,
    excluding any specified stop words. It returns the top N words and their
    corresponding frequencies.

    Args:
        text: Preprocessed text string.
        n: Number of top words to return.
        stop_words: Optional set of stop words to exclude.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing:
            - A list of the most frequent words.
            - A list of their corresponding counts.
    
    Examples:
        >>> text = "sadness sadness hope anger anger anger"
        >>> get_top_word_frequencies(text, n=2)
        (['anger', 'sadness'], [3, 2])
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
    """Create a horizontal bar plot of word frequencies.

    Args:
        words: List of words.
        counts: Corresponding frequency counts.

    Returns:
         matplotlib.figure.Figure: A matplotlib Figure object containing the horizontal bar chart.

    Examples:
        >>> words = ['museum', 'artifacts', 'history']
        >>> counts = [10, 5, 8]
        >>> fig = plot_word_frequencies(words, counts)
        >>> plt.show(fig)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette="viridis", ax=ax)
    ax.set_title(f"Top {len(words)} Words in Combined Responses")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Words")
    return fig
