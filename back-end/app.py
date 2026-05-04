"""
app.py — SkinCare.AI Flask Application
Slim entry point: configuration, data loading, and routes only.
All business logic lives in validators.py, recommender.py, and routine.py.
"""

import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from validators import validate_profile
from recommender import recommend_products, precompute_product_features
from routine import generate_routine_response

# --- CONFIGURATION & DATA LOADING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(base_dir, '../front-end')
data_dir = os.path.join(base_dir, '../data')
model_path = os.path.join(base_dir, 'trained_skincare_model.joblib')

app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)

# Load machine learning model
model = None
feature_names = []
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        feature_names = list(model.feature_names_in_)
        print("[OK] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
else:
    print(f"[WARN] Model file not found at {model_path}")

# Load products database
products_df = pd.DataFrame()
try:
    products_df = pd.read_csv(os.path.join(data_dir, 'products.csv'))
    print(f"[OK] Loaded {len(products_df)} products.")
except Exception as e:
    print(f"[ERROR] Error loading products: {e}")

# Precompute product feature matrix at startup for performance
precomputed_features = None
if model is not None and not products_df.empty:
    try:
        precomputed_features = precompute_product_features(products_df, feature_names)
        print("[OK] Product features precomputed.")
    except Exception as e:
        print(f"[WARN] Could not precompute features: {e}")


# --- ROUTES ---

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/health')
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'products_count': len(products_df),
        'features_precomputed': precomputed_features is not None
    })


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    1. Validate input
    2. Generate recommendations
    3. Build routine response
    """
    try:
        data = request.json

        # Step 1: Validate
        validation = validate_profile(data)
        if not validation['valid']:
            return jsonify({
                'error': 'Validation failed',
                'field_errors': validation['errors']
            }), 400

        sanitized = validation['sanitized']

        # Step 2: Get recommendations
        rec_out = recommend_products(
            sanitized, model, products_df,
            feature_names, precomputed_features
        )

        # Step 3: Generate response
        response = generate_routine_response(sanitized, rec_out)
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Error analyzing profile: {e}")
        return jsonify({
            'error': 'An internal error occurred. Please try again.',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)