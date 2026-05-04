"""
recommender.py — Skincare product recommendation engine.
Handles severity calculation, feature engineering, and ML-based product scoring.
Integrates water intake and gender into severity calculations.
"""

import numpy as np
import pandas as pd


# --- SEVERITY CALCULATION ---

def calculate_severities(data):
    """
    Calculate severity scores for 5 skin dimensions based on user profile.
    Each score is capped at 10.0.

    Factors: concerns, skin type, diet, climate, age, hormonal status,
             water intake, and gender-based adjustments.
    """
    acne, dry, pigm, aging, sens = 0.0, 0.0, 0.0, 0.0, 0.0

    concerns = data.get('concerns', '')
    skin_type = data.get('skinType', '')
    diet = data.get('diet', '')
    climate = data.get('climate', '')
    age_str = data.get('age', '')
    water = data.get('water', '')
    gender = data.get('gender', '')

    # --- Base logic from concerns ---
    if 'Acne' in concerns:
        acne += 8.0
    if 'Dryness' in concerns or 'flakiness' in concerns:
        dry += 8.0
    if 'Pigmentation' in concerns or 'dark spots' in concerns:
        pigm += 8.0
    if 'Anti-aging' in concerns or 'wrinkles' in concerns:
        aging += 8.0
    if 'Redness' in concerns or 'rosacea' in concerns:
        sens += 6.0
    if 'Oiliness' in concerns or 'shine' in concerns:
        acne += 2.0
    if 'Uneven texture' in concerns:
        pigm += 3.0
        dry += 2.0
    if 'Dullness' in concerns:
        pigm += 3.0
        dry += 2.0
        aging += 1.0
    if 'Sensitive' in skin_type:
        sens += 4.0

    # --- Skin type adjustments ---
    if skin_type == 'Oily':
        acne += 3.0
    elif skin_type == 'Dry':
        dry += 3.0
    elif skin_type == 'Sensitive':
        sens += 3.0
    elif skin_type == 'Combination':
        acne += 1.5
        dry += 1.5

    # --- Diet adjustments ---
    diet_lower = diet.lower()
    if 'sugar' in diet_lower or 'processed' in diet_lower:
        acne += 2.0
        aging += 1.0
    if 'dairy' in diet_lower:
        acne += 1.5
    if 'junk' in diet_lower:
        acne += 1.5
        aging += 0.5

    # --- Climate adjustments ---
    climate_lower = climate.lower()
    if 'cold' in climate_lower:
        dry += 2.0
        sens += 1.0
    if 'humid' in climate_lower:
        acne += 1.0
    if 'dry' in climate_lower and 'hot' in climate_lower:
        dry += 1.5
        pigm += 1.0  # Sun exposure
    if 'tropical' in climate_lower:
        acne += 1.0
        pigm += 0.5

    # --- Age adjustments ---
    if age_str in ['35–44']:
        aging += 3.0
    elif age_str in ['45–54']:
        aging += 5.0
    elif age_str == '55+':
        aging += 7.0
        dry += 1.5  # Mature skin tends to be drier
    elif age_str == '13–17':
        acne += 2.0  # Puberty

    # --- Hormonal adjustments ---
    horm = data.get('hormonal', '')
    if 'High Testosterone' in horm or 'PCOS' in horm or 'Hormonal-related acne' in horm:
        acne += 3.0
    if 'Low Testosterone' in horm:
        dry += 3.0
    if 'Pregnant' in horm or 'postpartum' in horm:
        pigm += 2.0
        sens += 1.0  # Pregnancy can trigger melasma and sensitivity
    if 'Menopause' in horm or 'perimenopause' in horm:
        dry += 2.5
        aging += 2.0
    if 'Stress' in horm:
        acne += 1.5
        sens += 1.0
    if 'HRT' in horm:
        sens += 1.0

    # --- Water intake adjustments ---
    if 'Less than 1L' in water:
        dry += 2.0
        aging += 1.0
    elif '1–2L' in water:
        dry += 0.5
    elif '3L+' in water:
        dry = max(0.0, dry - 1.0)  # Good hydration mitigates dryness

    # --- Gender-specific baseline adjustments ---
    if gender == 'Male':
        acne += 0.5  # Higher sebum production on average
    elif gender == 'Female' and age_str in ['45–54', '55+']:
        dry += 1.0  # Post-menopausal dryness

    # Cap all scores at 10.0
    return (
        min(10.0, acne),
        min(10.0, dry),
        min(10.0, pigm),
        min(10.0, aging),
        min(10.0, sens)
    )


# --- FEATURE ENGINEERING ---

def build_user_features(data, feature_names):
    """
    Build a feature dictionary for ML model input from user profile data.

    Args:
        data: Sanitized user profile dict
        feature_names: List of feature names from the trained model

    Returns:
        tuple: (feature_dict, severity_tuple)
    """
    # Parse age
    age_map = {
        "13–17": 15, "18–24": 21, "25–34": 30,
        "35–44": 40, "45–54": 50, "55+": 60
    }
    age = age_map.get(data.get('age'), 25)

    acne, dry, pigm, aging, sens = calculate_severities(data)

    user_feat = {
        'User_ID': -1,
        'Age': age,
        'Acne_Severity': acne,
        'Dryness_Severity': dry,
        'Pigmentation_Severity': pigm,
        'Aging_Severity': aging,
        'Sensitivity_Severity': sens
    }

    # Initialize all potential one-hot features to 0
    categorical_prefixes = [
        'Skin_Type_', 'Skin_Tone_', 'Climate_',
        'Diet_', 'Hormonal_Status_', 'Budget_Level_'
    ]
    for f in feature_names:
        if any(f.startswith(prefix) for prefix in categorical_prefixes):
            user_feat[f] = 0

    # --- Skin Type ---
    st = data.get('skinType', '')
    key = f'Skin_Type_{st}'
    if st and key in user_feat:
        user_feat[key] = 1

    # --- Skin Tone ---
    tone_map = {
        'Very fair (Type I)': 'Very_Fair',
        'Fair (Type II)': 'Fair',
        'Medium (Type III)': 'Medium',
        'Olive (Type IV)': 'Dusky',
        'Brown (Type V)': 'Dusky',
        'Deep (Type VI)': 'Deep'
    }
    tone = data.get('skinTone', '')
    mapped = tone_map.get(tone)
    if mapped:
        key = f'Skin_Tone_{mapped}'
        if key in user_feat:
            user_feat[key] = 1

    # --- Climate ---
    cli = data.get('climate', '')
    cli_lower = cli.lower()
    if 'cold' in cli_lower:
        if 'Climate_Cold' in user_feat: user_feat['Climate_Cold'] = 1
    elif 'dry' in cli_lower:
        if 'Climate_Dry' in user_feat: user_feat['Climate_Dry'] = 1
    elif 'humid' in cli_lower or 'tropical' in cli_lower:
        if 'Climate_Humid' in user_feat: user_feat['Climate_Humid'] = 1
    else:
        if 'Climate_Temperate' in user_feat: user_feat['Climate_Temperate'] = 1

    # --- Diet ---
    diet = data.get('diet', '')
    diet_lower = diet.lower()
    if 'sugar' in diet_lower or 'processed' in diet_lower:
        if 'Diet_High_Sugar' in user_feat: user_feat['Diet_High_Sugar'] = 1
    elif 'dairy' in diet_lower:
        if 'Diet_High_Dairy' in user_feat: user_feat['Diet_High_Dairy'] = 1
    elif 'vegan' in diet_lower or 'plant' in diet_lower:
        if 'Diet_Vegan' in user_feat: user_feat['Diet_Vegan'] = 1
    elif 'balanced' in diet_lower or 'healthy' in diet_lower:
        if 'Diet_Balanced' in user_feat: user_feat['Diet_Balanced'] = 1
    else:
        if 'Diet_Junk_Food' in user_feat: user_feat['Diet_Junk_Food'] = 1

    # --- Hormonal Status ---
    horm = data.get('hormonal', '')
    if any(x in horm for x in ['PCOS', 'High Testosterone', 'Hormonal-related acne']):
        if 'Hormonal_Status_PCOS' in user_feat: user_feat['Hormonal_Status_PCOS'] = 1
    elif 'Pregnant' in horm or 'postpartum' in horm:
        if 'Hormonal_Status_Pregnant' in user_feat: user_feat['Hormonal_Status_Pregnant'] = 1
    elif 'Menopause' in horm or 'perimenopause' in horm:
        if 'Hormonal_Status_Stable' in user_feat: user_feat['Hormonal_Status_Stable'] = 1
    elif age < 20 or 'HRT' in horm:
        if 'Hormonal_Status_Teen' in user_feat: user_feat['Hormonal_Status_Teen'] = 1
    else:
        if 'Hormonal_Status_Stable' in user_feat: user_feat['Hormonal_Status_Stable'] = 1

    # --- Budget Level ---
    budget = data.get('budget', '')
    if any(x in budget for x in ['Under ₹500', '₹500–₹1000', '₹1000–₹2500']):
        if 'Budget_Level_Low' in user_feat: user_feat['Budget_Level_Low'] = 1
    elif '₹2500–₹5000' in budget:
        if 'Budget_Level_Medium' in user_feat: user_feat['Budget_Level_Medium'] = 1
    else:
        if 'Budget_Level_High' in user_feat: user_feat['Budget_Level_High'] = 1

    return user_feat, (acne, dry, pigm, aging, sens)


# --- PRODUCT FEATURE PRECOMPUTATION ---

def precompute_product_features(products_df, feature_names):
    """
    Precompute the static product feature columns at startup.
    Returns a DataFrame with product-specific one-hot features filled in.
    Only needs to be called once when the server starts.
    """
    n_prods = len(products_df)
    product_features = pd.DataFrame(0, index=np.arange(n_prods), columns=feature_names)

    if 'Product_ID' in product_features.columns:
        product_features['Product_ID'] = products_df['Product_ID'].values
    if 'Price' in product_features.columns:
        product_features['Price'] = products_df['Price'].values

    for i, row in products_df.iterrows():
        for key in ['Brand', 'Ingredients', 'Category']:
            feat = f'{key}_{row[key]}'
            if feat in product_features.columns:
                product_features.loc[i, feat] = 1

    return product_features


# --- RECOMMENDATION ENGINE ---

def recommend_products(data, model, products_df, feature_names, precomputed_features=None):
    """
    Generate product recommendations using the ML model.

    Args:
        data: Sanitized user profile dict
        model: Trained sklearn model
        products_df: DataFrame of products
        feature_names: Model feature names
        precomputed_features: Optional precomputed product feature matrix

    Returns:
        dict: { 'top_items': [...], 'severities': (acne, dry, pigm, aging, sens) }
    """
    if model is None or products_df.empty:
        return {'top_items': [], 'severities': (0, 0, 0, 0, 0)}

    user_feat_base, severities = build_user_features(data, feature_names)

    # Use precomputed product features if available, otherwise compute fresh
    if precomputed_features is not None:
        X = precomputed_features.copy()
    else:
        X = precompute_product_features(products_df, feature_names)

    # Merge user features into the product matrix
    for k, v in user_feat_base.items():
        if k in X.columns:
            X[k] = v

    # Predict rating for each product
    preds = model.predict(X)

    # Store predictions
    temp_df = products_df.copy()
    temp_df['Predicted_Rating'] = preds

    # Filter by budget
    budget_txt = data.get('budget', '')
    min_price, max_price = _parse_budget(budget_txt)

    filtered = temp_df[
        (temp_df['Price'] >= min_price) & (temp_df['Price'] <= max_price)
    ].copy()

    # If no products match budget, widen range by 30% each direction
    if filtered.empty:
        expanded_min = max(0, min_price * 0.7)
        expanded_max = max_price * 1.3
        filtered = temp_df[
            (temp_df['Price'] >= expanded_min) & (temp_df['Price'] <= expanded_max)
        ].copy()

    top_items = filtered.sort_values(
        by='Predicted_Rating', ascending=False
    ).head(25).to_dict('records')

    return {
        'top_items': top_items,
        'severities': severities,
        'budget_expanded': filtered.empty  # flag if we had to widen
    }


def _parse_budget(budget_txt):
    """Parse budget text to min/max price range."""
    if 'Under ₹500' in budget_txt:
        return 0, 500
    elif '₹500–₹1000' in budget_txt:
        return 500, 1000
    elif '₹1000–₹2500' in budget_txt:
        return 1000, 2500
    elif '₹2500–₹5000' in budget_txt:
        return 2500, 5000
    elif '₹5000–₹10000' in budget_txt:
        return 5000, 10000
    elif '₹10000+' in budget_txt:
        return 10000, 999999
    return 0, 999999
