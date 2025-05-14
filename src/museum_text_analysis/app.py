import streamlit as st
from museum_text_analysis.bertopic_analysis import load_data, run_bertopic
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Streamlit app for BERTopic analysis
st.title("Museum Visitor Response Topic Explorer")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    try:
        # Load data from the uploaded file
        df = load_data(uploaded_file)

        # Display the first few rows of the DataFrame
        st.write("### Data Preview")
        st.write(df.head())

        # Run BERTopic on the combined text column
        texts = df["combined_text"].tolist()
        topics, model = run_bertopic(texts)
        st.success("Topic modeling complete!")

        # Display the topics
        st.write("### Top Topics")
        st.dataframe(model.get_topic_info().head())

        # Phrase frequency analysis
        st.write("### Phrase Counts")
        for phrase in ["deeply moved", "very moved", "somewhat moved", "not at all"]:
            count = df["combined_text"].str.lower().str.count(phrase).sum()
            st.write(f"{phrase.title()}: {int(count)} occurrence(s)")

        # Word cloud
        st.write("### Word Cloud of Responses")
        # Apply stop word filtering manually for word cloud
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop_words = ENGLISH_STOP_WORDS.union({"somewhat", "very", "deeply", "moved"})
        text_blob = " ".join(df["combined_text"].dropna().tolist())
        filtered_words = " ".join([word for word in text_blob.lower().split() if word not in stop_words])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_words)
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

        # BERTopic visualization
        st.write("### Topic Distribution")
        fig_topics = model.visualize_barchart(top_n_topics=10)
        st.plotly_chart(fig_topics)

    except Exception as e:
        st.error(f"Error: {e}")