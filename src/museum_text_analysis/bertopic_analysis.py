# src/museum_text_analysis/bertopic_analysis.py

from bertopic import BERTopic
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

def load_data(uploaded_file) -> pd.DataFrame:
    """Load and prepare the data you want to analyze.

    This function reads a CSV file and combines three open-ended text response 
    columns into a single column for further text analysis.

    Args:
        uploaded_file: A file-like object containing the CSV data to be processed.

    Returns:
        df (pd.DataFrame): A DataFrame containing the combined text responses.
    
    Raises:
        ValueError: If the expected columns are not found in the dataset.

    Examples:
        >>> from io import StringIO
        >>> csv_data = StringIO(
        ...     "What kind of emotions did the exhibit trigger in you?,"
        ...     "Is there an item or story from the exhibit that stayed with you? If so, why?,"
        ...     "What is your key takeaway from this exhibition?,"
        ...     "To what extent did the exhibition move you?\\n"
        ...     "Sadness,The suitcase of a child,Reminded me of the importance of remembering,Very much"
        ... )
        >>> df = load_data(csv_data)
        >>> print(df['combined_text'].iloc[0])
        Sadness The suitcase of a child Reminded me of the importance of remembering Very much
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
    """Fit BERTopic to a list of texts and return topics + model.
    
    This function performs topic modeling using BERTopic with a custom pipeline:
    - A CountVectorizer using custom stop words.
    - A sentence embedding model (MiniLM).
    - UMAP for dimensionality reduction.
    - HDBSCAN for clustering.
    - A predefined seed topic list to guide topic formation.

    Args:
        texts (list[str]): A list of open-ended text responses to analyze.

    Returns:
        tuple[list[int], BERTopic]: A tuple containing the list of topics 
        assigned to each text and the fitted BERTopic model.
   
    Raises: 
        ValueError: If the input texts are empty or not a list.

    Examples:
        >>> texts = [
        ...     "I felt sadness and anger at the exhibit.",
        ...     "The suitcase of a child really moved me.",
        ...     "My key takeaway is the importance of remembering history."
        ... ]
        >>> topics, model = run_bertopic(texts)
        >>> print(topics)
        [0, 1, 2]
    """
    if not texts or not isinstance(texts, list):
        raise ValueError("Input texts must be a non-empty list.")

    # Custom vectorizer with stop words
    vectorizer_model = CountVectorizer(stop_words=list(get_custom_stop_words()))

    # Custom embedding model for better quality
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Custom dimensionality reduction
    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.3, metric="cosine")

    # Use UMAP for more focused clusters
    umap_model = UMAP(
        n_neighbors=15,         
        n_components=5,
        min_dist=0.2,           
        metric="cosine"
    )

    # Use HDBSCAN for stricter cluster formation
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,   
        metric="euclidean",
        cluster_selection_method="eom"
    )

    # Seed topics to guide the model
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

    # Force reduction to fewer topics if needed
    model.reduce_topics(texts, nr_topics=5)

    return topics, model

def run_bertopic_per_column(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Run BERTopic separately for each column in a DataFrame of text responses.

    This function iterates over each column in the DataFrame, applies the
    BERTopic model to the open-text responses, and returns a dictionary of results.

    Args:
        df (pd.DataFrame): DataFrame where each column contains open-text responses.

    Returns:
        Dict[str, Dict[str, object]]: A dictionary where keys are column names and 
        values are dictionaries with the following structure:
            - "model" (BERTopic): The fitted BERTopic model for that column.
            - "topics" (list[int]): The list of topic labels assigned to each document in that column.
    
    Examples:
        >>> import pandas as pd
        >>> data = {
        ...     "response1": ["I felt sadness", "The suitcase moved me"],
        ...     "response2": ["My key takeaway is history", "I was inspired"]
        ... }
        >>> df = pd.DataFrame(data)
        >>> results = run_bertopic_per_column(df)
        >>> print(results.keys())
        dict_keys(['emotions', 'takeaway'])
        >>> print(results["emotions"]["topics"])
        [0, 1]
    """
    results = {}
    for col in df.columns:
        texts = df[col].fillna("").astype(str).tolist()
        topics, model = run_bertopic(texts)
        results[col] = {"model": model, "topics": topics}
    return results
