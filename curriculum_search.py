import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import nltk
from nltk.corpus import wordnet
import re

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
    
    results = []
    for _, row in df.iterrows():
        for col in search_cols:
            text = row[col]
            if text:
                scores = [fuzz.partial_ratio(term, text) for term in expanded_terms]
                max_score = max(scores, default=0)
                if max_score > 60:  # Threshold for match
                    results.append((max_score, row["LEVEL"], row["UNIT"], row["TOPIC AND CONTENT AREA"], row["PART"], col, text))
    
    results = sorted(results, key=lambda x: -x[0])[:max(5, len(results))]  # Ensure at least 5 results
    return results

# Streamlit UI
st.title("Reach Higher Curriculum Search")
search_query = st.text_input("Enter a topic or concept:")
category = st.radio("Search for:", ["Vocabulary", "Skill", "Genre"])

if search_query:
    matches = fuzzy_search(search_query, category)
    if matches:
        st.write("### Search Results:")
        results_df = pd.DataFrame(matches, columns=["Relevance", "Level", "Unit", "Topic & Content Area", "Part", "Matched Column", "Matched Content"])
        for _, row in results_df.iterrows():
            highlighted_text = re.sub(f"({search_query})", r"**\1**", row["Matched Content"], flags=re.IGNORECASE)
            st.markdown(f"- **{row['Level']} {row['Unit']} ({row['Topic & Content Area']}) Part {row['Part']}**")
            st.markdown(f"  - *{row['Matched Column']}*: {highlighted_text}")
    else:
        st.warning("No exact matches found. Try simplifying your search or using different keywords.")
