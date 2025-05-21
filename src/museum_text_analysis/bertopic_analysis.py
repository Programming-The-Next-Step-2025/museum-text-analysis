from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from museum_text_analysis.museum_topic_utils import get_custom_stop_words
from typing import Dict
from sentence_transformers import SentenceTransformer


def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load and prepare the data you want to analyse.

    This function reads a CSV file and combines three open-ended text response 
    columns into a single column for further text analysis.

    Args:
        uploaded_file: A file-like object containing the CSV data.

    Returns:
        pd.DataFrame: A DataFrame containing the combined text responses.
    
    Raises:
        ValueError: If the expected columns are not found in the dataset.
    """
    # Read CSV file from the uploaded file object
    df = pd.read_csv(uploaded_file, sep=",")

    text_columns = [
        "What kind of emotions did the exhibit trigger in you?",
        "Is there an item or story from the exhibit that stayed with you? If so, why?",
        "What is your key takeaway from this exhibition?",
        "To what extent did the exhibition move you?"
    ]

    for col in text_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    # Combine the three text columns into a single column
    df["combined_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

    return df

def run_bertopic(texts: list[str]) -> tuple[list[int], BERTopic]:
    if not texts or not isinstance(texts, list):
        raise ValueError("Input texts must be a non-empty list.")

    """
    Fit BERTopic to a list of texts and return topics + model.
    This function uses a custom CountVectorizer with a list of stop words
    and a seed topic list to guide the topic modeling process.

    Args:
        texts (list[str]): A list of text responses to analyze.

    Returns:
        tuple[list[int], BERTopic]: A tuple containing the list of topics 
        assigned to each text and the fitted BERTopic model.
   
    Raises: 
        ValueError: If the input texts are empty or not a list.
    """

    # Custom vectorizer with stop words
    vectorizer_model = CountVectorizer(stop_words=list(get_custom_stop_words()))

    # Custom embedding model for better quality
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Custom dimensionality reduction
    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.3, metric="cosine")

    # Use HDBSCAN for more aggressive topic merging
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom")

    # Seed topics
    seed_topic_list = [
        ["children", "sad", "anger", "cry"],
        ["fear", "hope", "inspiration"],
        ["fear", "shock", "sadness", "U.S"],
        ["never again", "warning", "repeat", "history"],
        ["USA", "Sobibor", "Trump", "don't forget"],
        ["resist", "kind", "aware", "sadness"]
    ]

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        seed_topic_list=seed_topic_list,
        min_topic_size=10,
        verbose=True
    )

    topics, _ = model.fit_transform(texts)

    # Force reduction to fewer topics if needed
    model.reduce_topics(texts, nr_topics=6)

    return topics, model

def run_bertopic_per_column(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """
    Run BERTopic separately for each column in a DataFrame of text responses.

    Args:
        df: DataFrame where each column contains open-text responses.

    Returns:
        A dictionary where each key is a column name, and the value is another
        dictionary containing:
            - "model": the BERTopic model trained on that column
            - "topics": the list of topic labels per document
    """
    results = {}
    for col in df.columns:
        texts = df[col].fillna("").astype(str).tolist()
        topics, model = run_bertopic(texts)
        results[col] = {"model": model, "topics": topics}
    return results
