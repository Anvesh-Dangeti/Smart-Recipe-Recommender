# Required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load dataset
file_path = 'C:/Users/bunny/Downloads/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'  
df = pd.read_csv(file_path)

# Cleaned_Ingredients are stored as strings, so convert them to lists for easier manipulation
df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: ast.literal_eval(x))

# Function to match user's available ingredients with recipes
def find_matching_recipes(user_ingredients, df, top_n=5):
    user_ingredients_str = ' '.join(user_ingredients)
    
    # Convert the ingredients in the dataset and user's input to a format suitable for vectorization
    recipe_ingredients = df['Cleaned_Ingredients'].apply(lambda x: ' '.join(x))
    
    # Use CountVectorizer to convert text data into a frequency matrix
    vectorizer = CountVectorizer().fit_transform(recipe_ingredients.tolist() + [user_ingredients_str])
    
    # Calculate cosine similarity between user ingredients and each recipe
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    
    # The last element in cosine_sim corresponds to user_ingredients
    user_similarity_scores = cosine_sim[-1][:-1]
    
    # Get top N similar recipes based on cosine similarity
    top_n_indices = user_similarity_scores.argsort()[-top_n:][::-1]
    
    # Return the top N matching recipes
    return df.iloc[top_n_indices][['Title', 'Cleaned_Ingredients', 'Instructions']]

# Function to display the selected recipe details
def display_recipe(recipe):
    title, ingredients, instructions = recipe['Title'], recipe['Cleaned_Ingredients'], recipe['Instructions']
    print(f"\nRecipe: {title}\n")
    print("Ingredients:")
    for ingredient in ingredients:
        print(f"- {ingredient}")
    print("\nInstructions:")
    print(instructions)

# Main function to run the program
def main():
    # Step 1: User inputs available ingredients
    user_ingredients = input("Enter your available ingredients, separated by commas: ").split(", ")
    
    # Step 2: Find and display the top 5 matching recipes
    matching_recipes = find_matching_recipes(user_ingredients, df)
    print("\nMatching Recipes:")
    for i, (_, row) in enumerate(matching_recipes.iterrows()):
        print(f"{i+1}. {row['Title']} (Ingredients: {', '.join(row['Cleaned_Ingredients'])})")
    
    # Step 3: User selects a dish
    selected_index = int(input("\nSelect a recipe by number: ")) - 1
    
    # Step 4: Display the selected recipe
    display_recipe(matching_recipes.iloc[selected_index])

if __name__ == "__main__":
    main()
