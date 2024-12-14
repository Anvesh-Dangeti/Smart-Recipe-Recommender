# ingredient_recognition.py

import os
from google.cloud import vision
import io

# Set up Google Cloud credentials (point to your JSON key file)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/working projects/recipe recommender project/service_account.json"

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# List of known ingredients to filter results
KNOWN_INGREDIENTS = [
    "eggplant", "tomato", "garlic", "pepper", "zucchini", "cucumber", "onion", 
    "potato", "carrot", "spinach", "apple", "banana", "lettuce", "broccoli",
    "beetroot", "mushroom", "Chicken meat", "beef", "pork", "salmon", "tuna", 
    "tofu", "cheddar cheese", "mozzarella cheese", "parmesan cheese", 
    "butter", "milk", "yogurt", "cream", "rice", "wheat", "quinoa", "oats", 
    "chickpeas", "lentils", "black beans", "kidney beans", "almonds", "walnuts", 
    "cashews", "chia seeds", "flaxseeds", "sunflower seeds", "sugar", "honey", 
    "maple syrup", "stevia", "flour", "baking soda", "baking powder", "yeast", 
    "cornstarch", "cocoa powder", "vanilla extract", "soy sauce", "vinegar", 
    "mustard", "ketchup", "hot sauce", "mayonnaise", "coconut milk", "peanut butter", 
    "jam", "spinach", "lemon", "lime", "bread", "corn", "pasta", "spaghetti", "Avocado",
    "meat", "raw meat", "fish meat", "raw fish"
    # Add more items to this list as needed
]


def recognize_ingredients(image_path):
    # Load the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Use label detection to identify general labels
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # Filter results: only keep items in KNOWN_INGREDIENTS and with high confidence
    ingredients = [
        label.description.lower() for label in labels
        if label.description.lower() in KNOWN_INGREDIENTS and label.score > 0.6
    ]

    # Handle errors
    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    return ingredients
