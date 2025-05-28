"""Run BERTopic analysis on sample survey responses."""

# Standard library
import os
import sys
from pathlib import Path

# Third-party
import pandas as pd

# Add the src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from museum_text_analysis.bertopic_analysis import load_data, run_bertopic

# Constants
SAMPLE_FILE = Path("sample_data/sample_responses.csv")
TEXT_COLUMNS = [
    "What kind of emotions did the exhibit trigger in you?",
    "Is there an item or story from the exhibit that stayed with you? If so, why?",
    "What is your key takeaway from this exhibition?"
]


def analyze_sample_file(filepath: Path = SAMPLE_FILE,
                        output_csv: bool = True):
    """Run BERTopic on combined text columns from a CSV file.

    This function loads a CSV file containing visitor responses, combines specified text columns,
    and runs BERTopic to identify topics in the responses. It also provides an option to save the topic
    summary to a CSV file.

    Args:
        filepath (Path): Path to the input CSV file. Default is 'sample_data/sample_responses.csv'.
        output_csv (bool): Whether to save the topic summary to a CSV file. Default is True.

    Returns:
        Tuple[BERTopic, pd.DataFrame]: A tuple containing:
            - model (BERTopic): The trained BERTopic model.
            - topic_info (pd.DataFrame): DataFrame containing topic information.

    Examples:
        >>> model, topic_info = analyze_sample_file()
        >>> print(topic_info.head())
        Top Topics Summary:
           Topic  Count  Name
        0      0    50  Topic 0
        1      1    30  Topic 1
        ...
    """
    print(f"Loading data from: {filepath}")
    df = load_data(filepath)

    print("Combining text columns...")
    df["combined_responses"] = df[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)

    print("Running BERTopic...")
    _, model = run_bertopic(df["combined_responses"].tolist())

    topic_info = model.get_topic_info()

    print("\nTop Topics Summary:")
    print(topic_info.head(10))

    if output_csv:
        output_path = Path("sample_data/sample_topic_summary.csv")
        topic_info.to_csv(output_path, index=False)
        print(f"\nTopic summary saved to: {output_path}")

    return model, topic_info


if __name__ == "__main__":
    analyze_sample_file()