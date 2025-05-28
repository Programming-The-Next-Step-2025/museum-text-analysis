# Museum Text Analysis

This project is a Streamlit-based application for analyzing open-ended questionnaire responses from museum visitors using BERTopic.  
Developed as part of the course *Programming: The Next Step 2025*, this tool helps researchers and museum staff extract and visualize thematic insights from qualitative data.

## Features

- Upload and process CSV files containing text responses
- Topic modeling using [BERTopic](https://doi.org/10.48550/arXiv.2203.05794)
- Visualize topics and word distributions interactively.
- Download topic summaries and visualizations for reporting purposes.
- User-friendly interface built with Streamlit.
- Includes a sample dataset for quick testing and demonstration.

## Installation

Install the package and its dependencies in editable mode:

```bash
pip install -e src
```

Ensure you have the required dependencies installed (see pyproject.toml for dependencies list).

## How to Use

### Option 1: Use the Interactive Streamlit App

1. Run the Streamlit app:

```python
streamlit run src/museum_text_analysis/app.py
```

2. In the browser window that opens:
    - Upload a CSV file containing open-text responses (see sample_data/sample_responses.csv for format).
    - The app will automatically detect text, preprocess it, and run BERTopic.
    - Explore the generated topics and keyword visualizations interactively.
    - Download the results as CSV files or visualizations for further analysis or reporting.

### Option 2: Run the Analysis Script Directly

1. If you prefer not to use the Streamlit app, you can run the BERTopic analysis directly from the command line.

```python
python src/museum_text_analysis/bertopic_analysis.py --run_analysis
```

or, if you prefer, you can run the provided script from the project root:

```python
python run_analysis.py
```

This will:
    - Load the sample data from `sample_data/sample_responses.csv`.
    - Preprocess the text data.
    - Run BERTopic to generate topics.
    - Save the topic summary in the `sample_data` folder.

## Sample Data
A sample dataset is included in the sample_data folder as sample_responses.csv to help you get started quickly.

## License

This project is licensed under the MIT Licens (https://choosealicense.com/licenses/mit/). You are free to use, modify, and distribute this code, provided that appropriate credit is given to the original author.