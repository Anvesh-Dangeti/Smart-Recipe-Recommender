import os
from flask import Flask, request, render_template, jsonify
from ingredient_recognition import recognize_ingredients
from recipe_recommender import find_similar_recipes

app = Flask(__name__)

# Ensure the 'uploaded_images' directory exists
os.makedirs('uploaded_images', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'files' not in request.files:
        return 'No files uploaded', 400

    files = request.files.getlist('files')
    if len(files) == 0:
        return 'No files selected', 400

    all_ingredients = set()

    # Process each uploaded image file
    for file in files:
        img_path = os.path.join("uploaded_images", file.filename)
        file.save(img_path)

        # Recognize ingredients from each image
        ingredients = recognize_ingredients(img_path)
        all_ingredients.update(ingredients)  # Add detected ingredients to the set

        # Remove the temporary image file
        os.remove(img_path)

    ingredients_list = list(all_ingredients)
    recommended_recipes = find_similar_recipes(ingredients_list).to_dict(orient='records')

    return render_template('results.html', ingredients=ingredients_list, recipes=recommended_recipes)

if __name__ == '__main__':
    app.run(debug=True)
