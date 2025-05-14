# src/museum_text_analysis/bertopic_analysis.py

from bertopic import BERTopic
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

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
    df = pd.read_csv(uploaded_file, sep=";")

    text_columns = [
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
    
    # Define stop words, including the answers to question 3, and convert to list
    custom_stop_words = ENGLISH_STOP_WORDS.union({
        "the", "of", "that", "to", "for", "is", "and",
        "somewhat", "very", "deeply", "moved", "people", 
        "just", "like"
    })
    stop_words_list = list(custom_stop_words)

    # Use custom CountVectorizer with stop words
    vectorizer_model = CountVectorizer(stop_words=stop_words_list)
    
    # Define seed topics: keywords that represent topics we’re interested in
    seed_topic_list = [
        ["children", "sad", "anger", "cry"],
        ["fear", "hope", "inspiration"],
        ["fear", "shock", "sadness", "U.S"],
        ["never again", "warning", "repeat", "history"],
        ["USA", "Sobibor", "Trump", "don't forget"],
        ["resist", "kind", "aware", "sadness"]
    ]

   # Create BERTopic model with custom vectorizer and seed topics
    model = BERTopic(
        vectorizer_model=vectorizer_model,
        seed_topic_list=seed_topic_list,
        verbose=True
    )

    # Fit the model to the texts
    topics, _ = model.fit_transform(texts)
    return topics, model