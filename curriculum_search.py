import os
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import nltk
from nltk.corpus import wordnet
import re

# Ensure WordNet data is available only if not already downloaded
def ensure_nltk_downloads():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

ensure_nltk_downloads()

# Set NLTK_DATA environment variable (useful for deployment)
os.environ['NLTK_DATA'] = '/path/to/nltk_data'

# Load CSV file
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def load_data():
    return pd.read_csv("reach_higher_curriculum_all_units.csv", dtype=str, keep_default_na=False)

df = load_data()

# Define columns based on search type
VOCAB_COLS = ["LEVEL", "UNIT", "TOPIC AND CONTENT AREA", "PART", "CONTENT VOCABULARY", "ACADEMIC VOCABULARY"]
SKILL_COLS = ["LEVEL", "UNIT", "TOPIC AND CONTENT AREA", "PART", "LANGUAGE SKILL", "THINKING MAP SKILL", "READING SKILL", "PHONICS SKILL", "GRAMMAR SKILL", "ORAL LANGUAGE PROJECT", "WRITING PROJECT"]
GENRE_COL = ["LEVEL", "UNIT", "TOPIC AND CONTENT AREA", "PART", "GENRES"]

# Common stop words to ignore in search queries
STOP_WORDS = {"a", "an", "and", "the", "in", "on", "at", "to", "for", "of", "with", "by", "about"}

# Function to get synonyms and related words
def expand_query(query):
    words = [word for word in query.lower().split() if word not in STOP_WORDS]
    expanded = set(words)
    
    for word in words:
        root = wordnet.morphy(word) or word  # Get base form
        for syn in wordnet.synsets(root):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    
    return list(expanded)

# Function to perform fuzzy search
def fuzzy_search(query, category):
    expanded_terms = expand_query(query)
    search_cols = {"Vocabulary": VOCAB_COLS, "Skill": SKILL_COLS, "Genre": GENRE_COL}[category]
    
    exact_matches, fuzzy_matches = [], []
    
    for _, row in df.iterrows():
        row_text = " ".join(str(row[col]) for col in search_cols if row[col])  # Merge columns into one string
        
        # Check exact match
        if re.search(rf"\b{re.escape(query)}\b", row_text, re.IGNORECASE):
            exact_matches.append((100, *row[["LEVEL", "UNIT", "TOPIC AND CONTENT AREA", "PART"]], row_text, query))
        else:
            best_match = process.extractOne(row_text, expanded_terms, scorer=fuzz.partial_ratio)
            if best_match and best_match[1] > 60:
                fuzzy_matches.append((best_match[1], *row[["LEVEL", "UNIT", "TOPIC AND CONTENT AREA", "PART"]], row_text, best_match[0]))
    
    return sorted(exact_matches, key=lambda x: -x[0]) + sorted(fuzzy_matches, key=lambda x: -x[0])

# Streamlit UI
st.title("Reach Higher Curriculum Search")
search_query = st.text_input("Enter a topic or concept:")
category = st.radio("Search for:", ["Vocabulary", "Skill", "Genre"])

if search_query:
    matches = fuzzy_search(search_query, category)
    if matches:
        st.write("### Search Results:")
        
        # Create a results DataFrame dynamically
        columns = ["Score", "Level", "Unit", "Topic", "Part", "Matched Content", "Matched Term"]
        results_df = pd.DataFrame(matches, columns=columns)
        
        # Highlight search term in results
        def highlight_match(row):
            return re.sub(f"({re.escape(row['Matched Term'])})", r'<span style="background-color: yellow">\1</span>', row["Matched Content"], flags=re.IGNORECASE)
        
        results_df["Matched Content"] = results_df.apply(highlight_match, axis=1)
        
        # Drop redundant columns
        results_df.drop(columns=["Score", "Matched Term"], inplace=True)
        
        # Display results
        st.markdown(results_df.to_html(index=False, escape=False), unsafe_allow_html=True)
    else:
        st.warning("No exact matches found. Try simplifying your search or using different keywords.")
