import pandas as pd
import fuzzywuzzy
from fuzzywuzzy import fuzz, process

# Load the CSV file with the curriculum data
curriculum_df = pd.read_csv('reach_higher_curriculum_all_units.csv')

# Example function to perform fuzzy search on a column
def fuzzy_search(query, column_name):
    # Extract the relevant column for searching
    column_data = curriculum_df[column_name].dropna().unique()
    # Use fuzzywuzzy to find the best matches
    results = process.extract(query, column_data, scorer=fuzz.partial_ratio)
    return results

# Function to find matching themes based on search input
def find_matching_themes(query):
    # Search in relevant columns (for example, 'Unit' and 'Vocabulary Words')
    unit_results = fuzzy_search(query, 'Unit')
    vocab_results = fuzzy_search(query, 'Vocabulary Words')
    
    # Combine and rank results by score
    all_results = unit_results + vocab_results
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    # Filter to return top 5 results
    return all_results[:5]

# Example of how you can search
query = "environment"
matching_themes = find_matching_themes(query)

# Print the results
for result in matching_themes:
    print(f"Match: {result[0]}, Score: {result[1]}")
