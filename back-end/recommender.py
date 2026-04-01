import pandas as pd
import numpy as np
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
data_dir = os.path.join(os.path.dirname(__file__), '../data')

if os.path.exists(model_path):
    model = joblib.load(model_path)
    feature_names = list(model.feature_names_in_)
else:
    model = None
    feature_names = []

products_df = pd.read_csv(os.path.join(data_dir, 'products.csv'))

def calculate_severities(data):
    acne, dry, pigm, aging, sens = 0.0, 0.0, 0.0, 0.0, 0.0
    
    concerns = data.get('concerns', '')
    skin_type = data.get('skinType', '')
    diet = data.get('diet', '')
    climate = data.get('climate', '')
    age_str = data.get('age', '')
    
    # Base logic from concerns
    if 'Acne' in concerns: acne += 8.0
    if 'Dryness' in concerns: dry += 8.0
    if 'Pigmentation' in concerns: pigm += 8.0
    if 'Anti-aging' in concerns or 'wrinkles' in concerns: aging += 8.0
    if 'Redness' in concerns or 'Sensitive' in skin_type: sens += 8.0
    if 'Oiliness' in concerns: acne += 2.0
    
    # Adjustments
    if skin_type == 'Oily': acne += 3.0
    if skin_type == 'Dry': dry += 3.0
    if skin_type == 'Sensitive': sens += 3.0
    
    if 'sugar' in diet.lower(): acne += 2.0; aging += 1.0
    if 'dairy' in diet.lower(): acne += 1.5
    
    if 'Cold' in climate: dry += 2.0; sens += 1.0
    if 'Hot' in climate and 'humid' in climate.lower(): acne += 1.0
    
    if age_str in ['35–44', '45–54', '55+']: aging += 4.0
    
    # cap at 10
    return min(10.0, acne), min(10.0, dry), min(10.0, pigm), min(10.0, aging), min(10.0, sens)


def build_user_features(data):
    # Parse age
    age_map = {"13–17": 15, "18–24": 21, "25–34": 30, "35–44": 40, "45–54": 50, "55+": 60}
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
    
    # Defaults
    for f in feature_names:
        if f.startswith('Skin_Type_') or f.startswith('Skin_Tone_') or \
           f.startswith('Climate_') or f.startswith('Diet_') or \
           f.startswith('Hormonal_Status_') or f.startswith('Budget_Level_'):
            user_feat[f] = 0
            
    # Assign ones
    skin_map = {'Combination': 'Combination', 'Dry': 'Dry', 'Normal': 'Normal', 'Oily': 'Oily', 'Sensitive': 'Sensitive'}
    st = data.get('skinType', '')
    if st in skin_map and f'Skin_Type_{skin_map[st]}' in user_feat:
        user_feat[f'Skin_Type_{skin_map[st]}'] = 1
        
    tone_map = {'Very fair (Type I)': 'Very_Fair', 'Fair (Type II)': 'Fair', 'Medium (Type III)': 'Medium', 
                'Olive (Type IV)': 'Dusky', 'Brown (Type V)': 'Dusky', 'Deep (Type VI)': 'Deep'}
    tone = data.get('skinTone', '')
    if tone in tone_map and f'Skin_Tone_{tone_map[tone]}' in user_feat:
        user_feat[f'Skin_Tone_{tone_map[tone]}'] = 1
        
    cli = data.get('climate', '')
    if 'Cold' in cli: user_feat['Climate_Cold'] = 1
    elif 'Dry' in cli: user_feat['Climate_Dry'] = 1
    elif 'humid' in cli.lower(): user_feat['Climate_Humid'] = 1
    else: user_feat['Climate_Temperate'] = 1
    
    diet = data.get('diet', '')
    if 'sugar' in diet.lower(): user_feat['Diet_High_Sugar'] = 1
    elif 'Dairy' in diet: user_feat['Diet_High_Dairy'] = 1
    elif 'vegan' in diet.lower(): user_feat['Diet_Vegan'] = 1
    elif 'Balanced' in diet: user_feat['Diet_Balanced'] = 1
    else: user_feat['Diet_Junk_Food'] = 1
    
    horm = data.get('hormonal', '')
    if 'PCOS' in horm: user_feat['Hormonal_Status_PCOS'] = 1
    elif 'Pregnant' in horm: user_feat['Hormonal_Status_Pregnant'] = 1
    elif age < 20: user_feat['Hormonal_Status_Teen'] = 1
    else: user_feat['Hormonal_Status_Stable'] = 1
    
    budget = data.get('budget', '')
    if 'Under ₹1500' in budget or '₹1500–₹4000' in budget: user_feat['Budget_Level_Low'] = 1
    elif '₹4000–₹8000' in budget: user_feat['Budget_Level_Medium'] = 1
    else: user_feat['Budget_Level_High'] = 1
    
    return user_feat, (acne, dry, pigm, aging, sens)


def recommend_products(data):
    if model is None:
        return []
    
    user_feat_base, severities = build_user_features(data)
    
    # Prepare dataframe for prediction
    n_prods = len(products_df)
    X = pd.DataFrame(0, index=np.arange(n_prods), columns=feature_names)
    
    for k, v in user_feat_base.items():
        if k in X.columns:
            X[k] = v
            
    # Add product features
    X['Product_ID'] = products_df['Product_ID']
    X['Price'] = products_df['Price']
    
    for i, row in products_df.iterrows():
        b = row['Brand']
        ing = row['Ingredients']
        
        if f'Brand_{b}' in X.columns:
            X.loc[i, f'Brand_{b}'] = 1
            
        if f'Ingredients_{ing}' in X.columns:
            X.loc[i, f'Ingredients_{ing}'] = 1
            
    # Predict
    preds = model.predict(X)
    
    # Store predictions back
    products_df['Predicted_Rating'] = preds
    
    # Filter by budget
    budget_txt = data.get('budget', '')
    max_price = 9999
    if 'Under ₹1500' in budget_txt: max_price = 20
    elif '₹1500–₹4000' in budget_txt: max_price = 50
    elif '₹4000–₹8000' in budget_txt: max_price = 100
    elif '₹8000–₹16000' in budget_txt: max_price = 200
    
    filtered = products_df[products_df['Price'] <= max_price].copy()
    if len(filtered) < 5: 
        # fallback if budget too strict
        filtered = products_df.copy()
        
    filtered = filtered.sort_values(by='Predicted_Rating', ascending=False)
    
    # Return top 15 recommended products and severities
    top_items = filtered.head(15).to_dict('records')
    return {'top_items': top_items, 'severities': severities}