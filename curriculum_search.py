import streamlit as st
import pandas as pd
from rapidfuzz import process
import difflib

# Load your data
curriculum_df = pd.read_csv('reach_higher_curriculum_all_units.csv')
rh3A_6B_df = pd.read_excel('rh3A-6B.xlsx')

# Function to perform fuzzy search
def fuzzy_search(query, column_data, scorer, limit=5):
    results = process.extract(query, column_data, limit=limit, scorer=scorer)
    return results

# Function to find related words and synonyms using WordNet (optional)
# You can extend this function to include lemmatization, synonyms, etc.

# Define scoring function based on search type
def get_scorer(search_type):
    if search_type == 'topic':
        return process.default_scorer
    elif search_type == 'concept':
        return difflib.SequenceMatcher(None, query, column_data)

# Main Streamlit app
def main():
    st.title("Curriculum Search")

    # Input for search query and selection of search type
    query = st.text_input("Enter a topic or concept to search for:")
    search_type = st.radio("Select search type", ('topic', 'concept'))

    # Search the data
    if query:
        # Search the CSV file (vocabulary and skill columns)
        if search_type == 'topic':
            vocabulary_column_data = curriculum_df['Vocabulary Words'].dropna().tolist()
            results = fuzzy_search(query, vocabulary_column_data, scorer=get_scorer('topic'))
        elif search_type == 'concept':
            skill_columns = ['Language Skill', 'Thinking Map Skill', 'Reading Skill', 'Grammar Skill']
            concept_column_data = pd.concat([curriculum_df[col] for col in skill_columns], axis=0).dropna().tolist()
            results = fuzzy_search(query, concept_column_data, scorer=get_scorer('concept'))

        if results:
            # Display results in a table
            st.write("### Search Results")
            results_df = pd.DataFrame(results, columns=["Match", "Score", "Index"])
            results_df['Unit Name'] = curriculum_df.iloc[results_df['Index']]['Unit']
            results_df['Vocabulary Words'] = curriculum_df.iloc[results_df['Index']]['Vocabulary Words']
            st.dataframe(results_df[['Unit Name', 'Match', 'Score', 'Vocabulary Words']])
        else:
            st.write("No matches found. Try refining your search.")
            st.write("Tips for effective search:")
            st.write("- Try a more specific term.")
            st.write("- Use simpler words or synonyms.")
            st.write("- Make sure the spelling is correct.")
    
    st.sidebar.write("### About")
    st.sidebar.write("This app helps you search for topics and concepts in the curriculum.")

if __name__ == "__main__":
    main()
