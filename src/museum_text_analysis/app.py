"""app.py

Streamlit app for uploading and analyzing museum visitor responses using BERTopic.

Steps:
1. Upload CSV with free-text responses.
2. Run topic modeling using BERTopic.
3. Display topics, both overall and per question.
4. Generate word clouds and frequency plots.
"""

# Standard library
import pandas as pd

# Third-party
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Local
from museum_text_analysis.bertopic_analysis import load_data, run_bertopic, run_bertopic_per_column
from museum_text_analysis.museum_topic_utils import (
    get_custom_stop_words,
    generate_wordcloud,
    get_top_word_frequencies,
    plot_word_frequencies,
)

# Streamlit app for BERTopic analysis
st.title("Museum Visitor Response Topic Explorer")

# Add sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload a CSV file with visitor responses.
2. Make sure it contains the expected question columns.
3. Wait for topic modeling to complete.
4. Explore topics and representative responses.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())

        bertopic_text_columns = [
            "What kind of emotions did the exhibit trigger in you?",
            "Is there an item or story from the exhibit that stayed with you? If so, why?",
            "What is your key takeaway from this exhibition?"
        ]
        moved_col = "To what extent did the exhibition move you?"

        # Combine all text responses into one column
        df["combined_responses"] = df[bertopic_text_columns].astype(str).agg(" ".join, axis=1)

        # Run overall topic modeling on all combined text
        st.subheader("Overall Topic Summary (All Responses Combined)")
        with st.spinner("Analyzing overall topics..."):
            overall_model = run_bertopic(df["combined_responses"].tolist())[1]
            overall_topic_info = overall_model.get_topic_info()
            st.dataframe(overall_topic_info)

            # Download button
            csv = overall_topic_info.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Overall Topic Summary CSV",
                data=csv,
                file_name="overall_topic_summary.csv",
                mime="text/csv"
            )

            # Topic distribution pie chart
            st.markdown("### Topic Distribution")
            fig = px.pie(
                overall_topic_info,
                values="Count",
                names="Topic",
                title="Overall Topic Distribution",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig)

        # Individual column topic modeling
        st.success("Individual Topic Modeling Complete! Explore Below:")

        results_per_col = run_bertopic_per_column(df[bertopic_text_columns].fillna(""))

        for col in bertopic_text_columns:
            st.header(f"Topics for: *{col}*")
            model = results_per_col[col]["model"]
            topics_df = model.get_topic_info()
            topics_df = topics_df[topics_df["Topic"] != -1].copy()
            topics_df["Clean Name"] = topics_df["Name"].str.replace(r"^\d+_", "", regex=True)

            fig = px.bar(
                topics_df,
                x="Clean Name",
                y="Count",
                title=f"Top Topics",
                color="Count",
                text="Count"
            )
            fig.update_layout(xaxis_tickangle=-45, title_x=0.3)
            st.plotly_chart(fig)

            st.subheader("Top Keywords per Topic")
            for _, row in topics_df.iterrows():
                topic_id = row["Topic"]
                keywords = model.get_topic(topic_id)
                keyword_str = ", ".join([word for word, _ in keywords[:10]])
                st.markdown(f"**Topic {topic_id}**: {keyword_str}")

            st.subheader("Representative Responses")
            for _, row in topics_df.iterrows():
                topic_id = row["Topic"]
                st.markdown(f"**Topic {topic_id}**:")
                try:
                    docs = model.get_representative_docs(topic_id)
                    if docs:
                        st.write(f"\u2022 {docs[0]}")
                    else:
                        st.write("No representative examples found.")
                except Exception:
                    st.write("No representative examples found.")

        # Phrase frequency bar chart for moved_col
        st.write(f"### Phrase Frequencies in '{moved_col}'")
        cleaned = df[moved_col].dropna().str.strip().str.lower()
        freq = cleaned.value_counts()
        expected_phrases = ["deeply moved", "very moved", "somewhat moved", "not at all"]
        freq_full = {phrase: freq.get(phrase, 0) for phrase in expected_phrases}

        for phrase, count in freq_full.items():
            st.write(f"{phrase.title()}: {count} occurrence(s)")

        st.bar_chart(pd.Series(freq_full))

        # Word cloud of combined responses
        st.write("### Word Cloud of Responses")
        combined_text_str = " ".join(df["combined_responses"]).lower()
        stop_words = get_custom_stop_words()
        fig_wc = generate_wordcloud(combined_text_str, stop_words)
        st.pyplot(fig_wc)

        # Optional frequency plot
        if st.checkbox("Show Top Word Frequencies"):
            stop_words = get_custom_stop_words()
            words, counts = get_top_word_frequencies(combined_text_str, stop_words=stop_words)
            fig = plot_word_frequencies(words, counts)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")