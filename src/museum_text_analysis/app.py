import streamlit as st
from museum_text_analysis.bertopic_analysis import load_data, run_bertopic, run_bertopic_per_column
from museum_text_analysis.museum_topic_utils import get_custom_stop_words, generate_wordcloud, get_top_word_frequencies, plot_word_frequencies
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Streamlit app for BERTopic analysis
st.title("Museum Visitor Response Topic Explorer")

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

        # BERTopic Combined
        combined_texts = (
            df[bertopic_text_columns[0]] + " " + df[bertopic_text_columns[1]]
        ).tolist()
        combined_topics, combined_model = run_bertopic(combined_texts)
        st.success("Text Topic Modeling Complete! Hang tight...")

        st.write("### Topic Modeling Per Question")
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
                title=f"Topic Frequencies for: {col}",
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
                    if docs:  # Ensure the list isn't empty
                        st.write(f"\u2022 {docs[0]}")  # Only the most representative doc
                    else:
                        st.write("No representative examples found.")
                except Exception:
                    st.write("No representative examples found.")

        st.write(f"### Phrase frequencies in '{moved_col}'")
        cleaned = df[moved_col].dropna().str.strip().str.lower()
        freq = cleaned.value_counts()

        expected_phrases = ["deeply moved", "very moved", "somewhat moved", "not at all"]
        freq_full = {phrase: freq.get(phrase, 0) for phrase in expected_phrases}

        for phrase, count in freq_full.items():
            st.write(f"{phrase.title()}: {count} occurrence(s)")

        st.bar_chart(pd.Series(freq_full))

        st.write("### Word Cloud of Responses")
        combined_text_str = " ".join(combined_texts).lower()
        stop_words = get_custom_stop_words()
        fig_wc = generate_wordcloud(combined_text_str, stop_words)
        st.pyplot(fig_wc)

        if st.checkbox("Show Top Word Frequencies"):
            stop_words = get_custom_stop_words()
            words, counts = get_top_word_frequencies(combined_text_str, stop_words=stop_words)
            fig = plot_word_frequencies(words, counts)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
