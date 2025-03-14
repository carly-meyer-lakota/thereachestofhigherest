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
VOCAB_COLS = ["LEVEL", "UNIT", "TOPIC AND CONTENT AREA", "PART", "CONTENT VOCABULARY", "ACADEMIC VOCABULARY"]
SKILL_COLS = ["LEVEL", "UNIT", "TOPIC AND CONTENT AREA", "PART", "LANGUAGE SKILL", "THINKING MAP SKILL", "READING SKILL", "PHONICS SKILL", "GRAMMAR SKILL", "ORAL LANGUAGE PROJECT", "WRITING PROJECT"]
GENRE_COL = ["LEVEL", "UNIT", "TOPIC AND CONTENT AREA", "PART", "GENRES"]

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
                    exact_matches.append((100, row["LEVEL"], row["UNIT"], row["TOPIC AND CONTENT AREA"], row["PART"], col, text, query))
                else:
                    scores = [(term, fuzz.partial_ratio(term, text)) for term in expanded_terms]
                    max_term, max_score = max(scores, key=lambda x: x[1], default=(None, 0))
                    if max_score > 60:  # Threshold for fuzzy match
                        fuzzy_matches.append((max_score, row["LEVEL"], row["UNIT"], row["TOPIC AND CONTENT AREA"], row["PART"], col, text, max_term))
    
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
        
        # Create a DataFrame for the results
        if category == "Vocabulary":
            results_df = pd.DataFrame(matches, columns=["Score", "Level", "Unit", "Topic", "Part", "Matched Column", "Matched Content", "Matched Term"])
        elif category == "Skill":
            results_df = pd.DataFrame(matches, columns=["Score", "Level", "Unit", "Topic", "Part", "Matched Column", "Matched Content", "Matched Term"])
            # Filter to only include matched skill columns
            results_df = results_df[["Score", "Level", "Unit", "Topic", "Part", "Matched Column", "Matched Content", "Matched Term"]]
        else:  # Genre
            results_df = pd.DataFrame(matches, columns=["Score", "Level", "Unit", "Topic", "Part", "Matched Column", "Matched Content", "Matched Term"])
        
        # Combine "Matched Column" and "Matched Content" columns
        results_df["Matched Content"] = results_df.apply(lambda row: f"{row['Matched Column']} - {row['Matched Content']}", axis=1)
        
        # Highlight the matching content in yellow
        results_df["Matched Content"] = results_df.apply(lambda row: re.sub(f"({re.escape(row['Matched Term'])})", r'<span style="background-color: yellow">\1</span>', row["Matched Content"], flags=re.IGNORECASE), axis=1)
        
        # Drop the redundant columns
        results_df = results_df.drop(columns=["Score", "Matched Column", "Matched Term"])
        
        # Display the results in a table
        st.write(results_df.to_html(index=False, escape=False), unsafe_allow_html=True)
    else:
        st.warning("No exact matches found. Try simplifying your search or using different keywords.")
