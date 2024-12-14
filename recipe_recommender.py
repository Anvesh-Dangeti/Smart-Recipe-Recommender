# recipe_recommender.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv('data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')

# Initialize TfidfVectorizer on the cleaned ingredients column of the dataset
vectorizer = TfidfVectorizer()
data['Cleaned_Ingredients'] = data['Cleaned_Ingredients'].fillna('')  # Handle any missing values
data_vectors = vectorizer.fit_transform(data['Cleaned_Ingredients'])

def find_similar_recipes(ingredients):
    """
    Find and return top 5 recipes based on similarity to the given ingredients list.
    """
    # Join ingredients into a single string for the query
    ingredients_query = " ".join(ingredients).lower()  # Convert to lowercase for consistency
    
    # Transform the query into the same vector space as the dataset
    query_vector = vectorizer.transform([ingredients_query])

    # Calculate cosine similarity between the query vector and all recipe vectors
    similarity_scores = cosine_similarity(query_vector, data_vectors).flatten()
    
    # Get the indices of the top 5 highest similarity scores
    top_indices = similarity_scores.argsort()[-5:][::-1]

    # Return the top 5 recipes with Title, Ingredients, and Instructions
    top_recipes = data.iloc[top_indices][['Title', 'Ingredients', 'Instructions']].copy()
    top_recipes['Similarity'] = similarity_scores[top_indices]  # Optional: add similarity scores
    
    return top_recipes