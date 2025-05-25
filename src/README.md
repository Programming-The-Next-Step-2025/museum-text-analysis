# Museum Text Analysis

This project is a Streamlit-based application for analyzing open-ended questionnaire responses from museum visitors using BERTopic.  
Developed as part of the course *Programming: The Next Step 2025*, this tool helps researchers and museum staff extract and visualize thematic insights from qualitative data.

## Features

- Upload and process CSV files containing text responses
- Topic modeling using [BERTopic](https://doi.org/10.48550/arXiv.2203.05794)
- Visualize topics and word distributions interactively.
- Download topic summaries and visualizations

## Installation

Install the package and its dependencies in editable mode:

```bash
pip install -e .
```
Ensure you have the required dependencies installed (see pyproject.toml for dependencies list).

## Usage

Run the Streamlit app:
```python
streamlit run src/museum_text_analysis/app.py
```

Alternatively, run the topic modeling analysis script directly without launching the app:
```python
python src/museum_text_analysis/bertopic_analysis.py --run_analysis
```
or, if you prefer, you can run the provided script from the project root:
```python
python run_analysis.py
```
You can install the package directly from GitHub using pip:
```python
pip install git+https://github.com/YourUserName/your-repo-name.git#egg=museum_text_analysis
```

Replace YourUserName and your-repo-name with your GitHub details.

After installation, you can import and use the package in your own Python projects:
```python
from museum_text_analysis.bertopic_analysis import run_bertopic
```
A sample dataset is included in the sample_data folder as sample_responses.csv to help you get started quickly.

## License

This project is licensed under the MIT Licens (https://choosealicense.com/licenses/mit/). You are free to use, modify, and distribute this code, provided that appropriate credit is given to the original authors.