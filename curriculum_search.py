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

# Define columns based on search type, excluding "TOPIC AND CONTENT AREA" and including "WORD WORK"
VOCAB_COLS = ["CONTENT VOCABULARY", "ACADEMIC VOCABULARY", "WORD WORK"]
SKILL_COLS = ["LANGUAGE SKILL", "THINKING MAP SKILL", "READING SKILL", "PHONICS SKILL", "GRAMMAR SKILL", "ORAL LANGUAGE PROJECT", "WRITING PROJECT", "WORD WORK"]
GENRE_COLS = ["GENRES", "WORD WORK"]

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
    # Select the relevant columns for each category
    search_cols = {
        "Vocabulary": VOCAB_COLS,
        "Skill": SKILL_COLS,
        "Genre": GENRE_COLS,
    }[category]
    
    matches = []
    
    for _, row in df.iterrows():
        for col in search_cols:
            if not row[col]:  # Skip empty cells
                continue
            cell_content = str(row[col])
            
            # Check for exact match
            if re.search(rf"\b{re.escape(query)}\b", cell_content, re.IGNORECASE):
                matches.append((
                    row["LEVEL"], row["UNIT"], row["TOPIC AND CONTENT AREA"],
                    row["PART"], col, cell_content, query
                ))
            else:
                # Fuzzy match
                best_match = process.extractOne(cell_content, expanded_terms, scorer=fuzz.partial_ratio)
                if best_match and best_match[1] > 60:  # Threshold for fuzzy match
                    matches.append((
                        row["LEVEL"], row["UNIT"], row["TOPIC AND CONTENT AREA"],
                        row["PART"], col, cell_content, best_match[0]
                    ))
    
    return matches

# Streamlit UI
st.title("Reach Higher Curriculum Search")
search_query = st.text_input("Enter a topic or concept:")
category = st.radio("Search for:", ["Vocabulary", "Skill", "Genre"])

if search_query:
    matches = fuzzy_search(search_query, category)
    if matches:
        # Prepare the results DataFrame
        columns = ["Level", "Unit", "Topic", "Part", "Match Found", "Matched Content", "Matched Term"]
        results_df = pd.DataFrame(matches, columns=columns)
        
        # Highlight the matched term in the "Matched Content" column
        def highlight_match(row):
            return re.sub(
                rf"({re.escape(row['Matched Term'])})",
                r'<span style="background-color: yellow">\1</span>',
                row["Matched Content"], flags=re.IGNORECASE
            )
        
        results_df["Matched Content"] = results_df.apply(highlight_match, axis=1)
        
        # Drop "Matched Term" column before displaying
        results_df.drop(columns=["Matched Term"], inplace=True)
        
        # Display results
        st.markdown("### Search Results:")
        st.markdown(results_df.to_html(index=False, escape=False), unsafe_allow_html=True)
    else:
        st.warning("No matches found. Try simplifying your search or using different keywords.")
