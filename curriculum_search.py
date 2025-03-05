import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import nltk
from nltk.corpus import wordnet
import re

# Ensure WordNet data is available
nltk.download('wordnet')
nltk.download('omw-1.4')  # Download the Open Multilingual WordNet if needed

# Load CSV file
@st.cache_data
def load_data():
    return pd.read_csv("reach_higher_curriculum_all_units.csv", dtype=str, keep_default_na=False)

df = load_data()

# Define columns based on search type
VOCAB_COLS = ["CONTENT VOCABULARY", "ACADEMIC VOCABULARY"]
SKILL_COLS = ["LANGUAGE SKILL", "THINKING MAP SKILL", "READING SKILL", "PHONICS SKILL", "GRAMMAR SKILL", "ORAL LANGUAGE PROJECT", "WRITING PROJECT"]
GENRE_COL = ["GENRES"]
ALL_COLS = VOCAB_COLS + SKILL_COLS + GENRE_COL

# Function to get synonyms and related words
def expand_query(query):
    words = query.lower().split()
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return list(expanded)

# Function to perform fuzzy search
def fuzzy_search(query, category):
    expanded_terms = expand_query(query)
    search_cols = VOCAB_COLS if category == "Vocabulary" else SKILL_COLS if category == "Skill" else GENRE_COL
    
    exact_matches = []
    fuzzy_matches = []

    for _, row in df.iterrows():
        for col in search_cols:
            text = row[col]
            if text:
                # Check for exact matches first
                if re.search(rf"\b{re.escape(query)}\b", text, re.IGNORECASE):  # Exact match
                    exact_matches.append((100, row["LEVEL"], row["UNIT"], row["TOPIC AND CONTENT AREA"], row["PART"], col, text))
                else:
                    scores = [fuzz.partial_ratio(term, text) for term in expanded_terms]
                    max_score = max(scores, default=0)
                    if max_score > 60:  # Threshold for fuzzy match
                        fuzzy_matches.append((max_score, row["LEVEL"], row["UNIT"], row["TOPIC AND CONTENT AREA"], row["PART"], col, text))
    
    # Combine exact matches first, followed by fuzzy matches (exact match first with score of 100)
    all_matches = sorted(exact_matches, key=lambda x: -x[0]) + sorted(fuzzy_matches, key=lambda x: -x[0])  # Ensure exact matches come first
    return all_matches

# Streamlit UI
st.title("Reach Higher Curriculum Search")
search_query = st.text_input("Enter a topic or concept:")
category = st.radio("Search for:", ["Vocabulary", "Skill", "Genre"])

if search_query:
    matches = fuzzy_search(search_query, category)
    if matches:
        st.write("### Search Results:")
        for score, level, unit, topic, part, matched_col, matched_content in matches:
            # Format the first line to match the requested output format
            formatted_result = f"Reach Higher {level} Unit {unit}: {topic} - {part}"
            
            # Highlight the matching content in yellow
            highlighted_text = re.sub(f"({search_query})", r'<mark>\1</mark>', matched_content, flags=re.IGNORECASE)
            
            # Improved result format
            st.markdown(f"#### {formatted_result}")
            st.markdown(f"  - **Matched Column**: {matched_col}")
            st.markdown(f"  - *Matched Content*: {highlighted_text}", unsafe_allow_html=True)
    else:
        st.warning("No exact matches found. Try simplifying your search or using different keywords.")
