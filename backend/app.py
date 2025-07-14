from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load and clean dataset
df = pd.read_csv("amazon_eco-friendly_products.csv")
df.dropna(subset=['name', 'description', 'category'], inplace=True)
df['material'].fillna('Unknown', inplace=True)
# Retaining duplicates for now to allow for more products,
# as a product name might not be unique if it's slightly different
# or if it exists in multiple categories.
# df.drop_duplicates(subset='name', inplace=True) # Commenting out original line

# Clean text columns
for col in ['name', 'category', 'material', 'brand', 'description']:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Eco keywords list
ECO_KEYWORDS = [
    'bamboo', 'organic', 'biodegradable', 'compostable',
    'recycled', 'eco-friendly', 'eco friendly', 'sustainable',
    'natural', 'reusable', 'plant-based', 'jute', 'cotton',
    # Added more material-focused keywords to broaden eco-score calculation
    'glass', 'stainless steel', 'silicone', 'ceramic', 'wood', 'hemp', 'upcycled'
]

# Compute eco-score for a given product row
def compute_eco_score(row):
    score = 0

    # Prioritize material for higher impact on eco-score
    # Assign higher points if eco-keyword is in material
    for kw in ECO_KEYWORDS:
        if kw in row['material']:
            score += 5 # High weight for eco-friendly materials

    # Add score from description (lower weight than material)
    for kw in ECO_KEYWORDS:
        if kw in row['description']:
            score += 1 # Lower weight for description

    # Penalize common non-eco-friendly materials
    if 'plastic' in row['material']:
        score -= 5 # Significant penalty for plastic
    if 'synthetic' in row['material']:
        score -= 3 # Penalty for synthetic materials
    if 'nylon' in row['material']:
        score -= 2 # Specific penalty for nylon

    # Ensure score does not go below zero
    score = max(0, score)
    return score

# Find the best product name match in a DataFrame using TF-IDF cosine similarity
def find_best_match(df_to_search, input_name):
    if df_to_search.empty:
        return None, 0

    names = df_to_search['name'].tolist()
    
    # Handle cases where TF-IDF might struggle with very few samples
    if len(names) == 1 and names[0].lower() == input_name.lower():
        return df_to_search.iloc[0], 1.0 # Perfect match
    elif len(names) == 0:
        return None, 0

    tfidf_vectorizer = TfidfVectorizer()
    # Fit and transform the product names and the input name
    tfidf_matrix = tfidf_vectorizer.fit_transform(names + [input_name.lower()])
    
    # Calculate cosine similarity between the input name and all product names
    sim = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    best_index = sim.argmax()
    return df_to_search.iloc[best_index], sim[best_index]

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    product_input = data.get('product')
    selected_category = data.get('category')

    if not product_input or not selected_category:
        return jsonify({"error": "Both 'product' and 'category' are required."}), 400

    selected_category = selected_category.strip().lower()
    
    # Filter the main DataFrame for the selected category
    category_df = df[df['category'] == selected_category].copy()

    if category_df.empty:
        return jsonify({"error": f"No products found in category '{selected_category}'."})

    # Find the best match for the user's input product within the selected category
    matched_row, match_score = find_best_match(category_df, product_input)
    
    # If no good match is found for the input product, provide a helpful message
    if matched_row is None or match_score < 0.2: # Adjusted threshold for a "good enough" match
        return jsonify({"message": f"Could not find a strong match for '{product_input}' in category '{selected_category}'. Please try a different product name or category."})

    original_product_name = matched_row['name']
    original_product_material = matched_row['material']
    
    # Calculate the eco-score for the original matched product
    input_score = compute_eco_score(matched_row)

    # --- REVISED LOGIC FOR FINDING ALTERNATIVES ---

    # Calculate eco-scores for all products in the category_df
    # This ensures every potential alternative has an eco-score for comparison
    category_df['eco_score'] = category_df.apply(compute_eco_score, axis=1)

    # Filter out the original product itself from potential alternatives
    potential_alternatives = category_df[
        (category_df['name'] != original_product_name) |
        (category_df['material'] != original_product_material) # Use material as well for robustness
    ].copy()

    # Strategy 1: Find alternatives that are strictly greener (higher eco_score)
    greener_alternatives = potential_alternatives[
        (potential_alternatives['eco_score'] > input_score)
    ].copy()

    # Strategy 2: If no strictly greener alternatives found, try alternatives with equal or slightly better scores,
    # or prioritize non-plastic options if the original was plastic.
    if greener_alternatives.empty:
        # Fallback 1: Look for alternatives with an equal or better score (but not necessarily strictly higher)
        # and ensure the score is positive.
        greener_alternatives = potential_alternatives[
            (potential_alternatives['eco_score'] >= input_score) & 
            (potential_alternatives['eco_score'] > 0) # Exclude zero-score generic items as 'greener'
        ].copy()
        
        # Fallback 2: If still no alternatives, and the original product contained 'plastic',
        # try to find non-plastic alternatives with a positive score.
        if greener_alternatives.empty and 'plastic' in original_product_material:
             greener_alternatives = potential_alternatives[
                (~potential_alternatives['material'].str.contains('plastic')) & # Material is NOT plastic
                (potential_alternatives['eco_score'] > 0) # Still needs a positive eco-score
             ].copy()

    # If, after all strategies, no suitable greener alternative is found
    if greener_alternatives.empty:
        return jsonify({
            "matched_input": original_product_name,
            "original_eco_score": int(input_score),
            "message": f"No strictly greener alternative found for '{original_product_name}'. Your product has an eco-score of {int(input_score)}. You might already have a good choice, or the dataset doesn't contain a better option."
        })

    # Sort the found greener alternatives by eco-score in descending order and pick the top one
    best = greener_alternatives.sort_values(by='eco_score', ascending=False).iloc[0]

    # Robustly convert price to float, handling '$' and potential NaN values
    price_value = "N/A"
    if not pd.isnull(best['price']):
        try:
            price_str = str(best['price']).replace('$', '')
            price_value = float(price_str)
        except ValueError:
            price_value = "N/A" # If conversion fails, keep as N/A

    return jsonify({
        "matched_input": original_product_name,
        "original_eco_score": int(input_score),
        "greener_alternative": best['name'],
        "eco_score": int(best['eco_score']),
        "material": best['material'],
        "brand": best['brand'],
        "description": best['description'],
        "price": price_value
    })

@app.route('/categories', methods=['GET'])
def list_categories():
    return jsonify(sorted(df['category'].unique().tolist()))

if __name__ == '__main__':
    app.run(debug=True)