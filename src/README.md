# Museum Text Analysis

This project is a Streamlit-based application for analysing open-ended questionnaire responses from museum visitors using BERTopic.  
It is developed as part of the course *Programming: The Next Step 2025*.

## Project Overview

This project aims to support topic modeling of qualitative survey data using BERTopic. It is designed to help users, such as researchers or museum staff, explore themes in open-ended text responses.

This repository is structured as a Python package and will be extended over the course of four weeks.

## Features (Planned)

- Upload and process CSV files with text data
- Topic modeling using [BERTopic](https://doi.org/10.48550/arXiv.2203.05794)
- Visualise data and topic summaries
- Download reports

## Directory Structure

museum-text-analysis/
├── src/
│ └── museum_text_analysis/
│ ├── init.py
│ ├── app.py
│ └── bertopic_analysis.py
├── sample_data/
│ └── sample_responses.csv
├── outputs/
│ └── topics.png
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

## License

This project is licensed under the MIT Licens (https://choosealicense.com/licenses/mit/). You are free to use, modify, and distribute this code, provided that appropriate credit is given to the original authors.