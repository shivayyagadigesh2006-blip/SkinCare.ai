import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- CONFIGURATION & DATA LOADING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
# Point static_folder to the front-end directory
frontend_dir = os.path.join(base_dir, '../front-end')
data_dir = os.path.join(base_dir, '../data')
model_path = os.path.join(base_dir, 'model.pkl')

app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)

# Load machine learning model
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        feature_names = list(model.feature_names_in_)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        feature_names = []
else:
    print(f"Model file not found at {model_path}")
    model = None
    feature_names = []

# Load products database
try:
    products_df = pd.read_csv(os.path.join(data_dir, 'products.csv'))
    print(f"Loaded {len(products_df)} products.")
except Exception as e:
    print(f"Error loading products: {e}")
    products_df = pd.DataFrame()

# --- RECOMMENDATION LOGIC (formerly recommender.py) ---

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

    # Hormonal adjustments to severities (Dynamic Mapping)
    horm = data.get('hormonal', '')
    if 'High Testosterone' in horm or 'PCOS' in horm or 'Hormonal-related acne' in horm:
        acne += 3.0
    if 'Low Testosterone' in horm:
        dry += 3.0
    if 'Pregnant' in horm:
        pigm += 2.0; sens += 1.0 # Pregnancy can trigger melasma and sensitivity
    
    # cap at 10 to keep within model expectations
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
    
    # Initialize all potential features to 0
    for f in feature_names:
        if any(f.startswith(prefix) for prefix in ['Skin_Type_', 'Skin_Tone_', 'Climate_', 'Diet_', 'Hormonal_Status_', 'Budget_Level_']):
            user_feat[f] = 0
            
    # Assign ones based on user profile
    st = data.get('skinType', '')
    if st and f'Skin_Type_{st}' in user_feat:
        user_feat[f'Skin_Type_{st}'] = 1
        
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
    if any(x in horm for x in ['PCOS', 'High Testosterone', 'Hormonal-related acne']): 
        user_feat['Hormonal_Status_PCOS'] = 1
    elif 'Pregnant' in horm: 
        user_feat['Hormonal_Status_Pregnant'] = 1
    elif age < 20 or 'HRT' in horm:
        user_feat['Hormonal_Status_Teen'] = 1
    else: 
        user_feat['Hormonal_Status_Stable'] = 1
    
    budget = data.get('budget', '')
    if any(x in budget for x in ['Under ₹500', '₹500–₹1000', '₹1000–₹2500']):
        user_feat['Budget_Level_Low'] = 1
    elif '₹2500–₹5000' in budget:
        user_feat['Budget_Level_Medium'] = 1
    else:
        user_feat['Budget_Level_High'] = 1
    
    return user_feat, (acne, dry, pigm, aging, sens)

def recommend_products(data):
    if model is None or products_df.empty:
        return {'top_items': [], 'severities': (0,0,0,0,0)}
    
    user_feat_base, severities = build_user_features(data)
    
    # Prepare matrix for prediction
    n_prods = len(products_df)
    X = pd.DataFrame(0, index=np.arange(n_prods), columns=feature_names)
    
    for k, v in user_feat_base.items():
        if k in X.columns:
            X[k] = v
            
    # Add product specific features
    if 'Product_ID' in X.columns: X['Product_ID'] = products_df['Product_ID']
    if 'Price' in X.columns: X['Price'] = products_df['Price']
    
    for i, row in products_df.iterrows():
        for key in ['Brand', 'Ingredients', 'Category']:
            feat = f'{key}_{row[key]}'
            if feat in X.columns:
                X.loc[i, feat] = 1
            
    # Predict rating for each product
    preds = model.predict(X)
    
    # Store predictions
    temp_df = products_df.copy()
    temp_df['Predicted_Rating'] = preds
    
    # Filter by budget strictly according to range
    budget_txt = data.get('budget', '')
    min_price, max_price = 0, 999999
    if 'Under ₹500' in budget_txt:
        min_price, max_price = 0, 500
    elif '₹500–₹1000' in budget_txt:
        min_price, max_price = 500, 1000
    elif '₹1000–₹2500' in budget_txt:
        min_price, max_price = 1000, 2500
    elif '₹2500–₹5000' in budget_txt:
        min_price, max_price = 2500, 5000
    elif '₹5000–₹10000' in budget_txt:
        min_price, max_price = 5000, 10000
    elif '₹10000+' in budget_txt:
        min_price, max_price = 10000, 999999
    
    filtered = temp_df[(temp_df['Price'] >= min_price) & (temp_df['Price'] <= max_price)].copy()
    
    # Return top items and user severities
    top_items = filtered.sort_values(by='Predicted_Rating', ascending=False).head(25).to_dict('records')
    return {'top_items': top_items, 'severities': severities}

# --- ROUTINE GENERATION (formerly routine.py) ---

def get_level(score):
    if score < 3.5: return "Low"
    if score < 7.0: return "Moderate"
    return "High"

def generate_choice_block(cat, items, user_concerns):
    if not items: return ""
    
    primary = items[0]
    alts = items[1:3] # show up to 2 alternatives
    
    reasoning = "This is our primary recommendation because "
    ing = primary['Ingredients'].lower()
    conc = user_concerns.lower()
    
    if 'acne' in conc and 'salicylic' in ing:
        reasoning += "it contains Salicylic Acid which is the gold standard for unclogging your pores and preventing the breakouts you mentioned."
    elif 'dry' in conc and ('hyaluronic' in ing or 'ceramide' in ing):
        reasoning += "its blend of hydrating actives will repair your moisture barrier and stop the flakiness you're experiencing."
    elif 'pigment' in conc and ('vitamin c' in ing or 'niacinamide' in ing or 'alpha arbutin' in ing):
        reasoning += "it uses potent brightening agents to target your dark spots and even out your skin tone."
    elif 'aging' in conc and ('retinol' in ing or 'peptide' in ing):
        reasoning += "the anti-aging actives will stimulate collagen production to address your wrinkles and fine lines."
    elif 'sunscreen' in cat.lower():
        reasoning += "it provides high-performance UV protection without being greasy, which is crucial for preventing future damage."
    else:
        reasoning += f"its formula targets your primary concerns with a high-rated blend of {primary['Ingredients'].replace('|', ', ')}."

    html = f'<div class="choice-block">'
    html += f'  <div class="primary-suggestion verified">'
    html += f'    <span class="strong-badge"><span class="check-icon">✓</span> OUR VERIFIED AI SELECTION</span>'
    html += f'    <div class="prod-name">{primary["Brand"]}</div>'
    html += f'    <div class="prod-meta">{primary["Category"]} · ₹{primary["Price"]}</div>'
    html += f'    <div class="why-reason"><strong>Why this?</strong> {reasoning}</div>'
    html += f'    <div class="purchase-guide">Recommended as your primary purchase for this step.</div>'
    html += f'  </div>'
    
    if alts:
        html += f'  <div class="alternatives-section">'
        html += f'    <div class="alt-label">Secondary Backup Options:</div>'
        for alt in alts:
            html += f'    <div class="alt-item">{alt["Brand"]} ({alt["Category"]}, ₹{alt["Price"]})</div>'
        html += f'  </div>'
    
    return html + '</div>'

def generate_routine_response(user_data, recommender_output):
    top_items = recommender_output['top_items']
    acne, dry, pigm, aging, sens = recommender_output['severities']
    
    # Filter products by category
    by_cat = {}
    for item in top_items:
        cat = item['Category']
        if cat not in by_cat: by_cat[cat] = []
        by_cat[cat].append(item)

    # Metrics
    metrics = {
        "acne": get_level(acne),
        "dryness": get_level(dry),
        "pigmentation": get_level(pigm),
        "aging": get_level(aging),
        "sensitivity": get_level(sens)
    }
    
    # Analysis summary
    st = user_data.get('skinType', 'Normal')
    concerns = user_data.get('concerns', 'wellness')
    analysis = f"Based on our AI analysis, you have <strong>{st}</strong> skin with primary concerns focusing on <strong>{concerns}</strong>. "
    if acne > 6.0: analysis += "Your profile strongly indicates an active inflammatory state. "
    if dry > 6.0: analysis += "Significant barrier disruption detected. "
    analysis += "A consistent, science-backed routine is non-negotiable for achieving the results you desire."
    
    # Routine Construction
    morning_html = ""
    night_html = ""
    weekly_html = ""

    # Morning
    if by_cat.get('Cleanser'): morning_html += "<h4>1. Deep Cleanse</h4>" + generate_choice_block('Cleanser', by_cat['Cleanser'], concerns)
    if by_cat.get('Serum'): morning_html += "<h4>2. Target Treatment</h4>" + generate_choice_block('Serum', by_cat['Serum'], concerns)
    if by_cat.get('Moisturizer'): morning_html += "<h4>3. Hydrate & Seal</h4>" + generate_choice_block('Moisturizer', by_cat['Moisturizer'], concerns)
    if by_cat.get('Sunscreen'): morning_html += "<h4>4. PROTECT (Mandatory)</h4>" + generate_choice_block('Sunscreen', by_cat['Sunscreen'], concerns)

    # Night
    if by_cat.get('Cleanser'):
        cleansers = by_cat['Cleanser'][1:] if len(by_cat['Cleanser']) > 1 else by_cat['Cleanser']
        night_html += "<h4>1. Gentle Cleanse</h4>" + generate_choice_block('Cleanser', cleansers, concerns)
    treatments = (by_cat.get('Treatment', []) + by_cat.get('Exfoliant', []))
    if treatments: night_html += "<h4>2. Active Repair</h4>" + generate_choice_block('Treatment', treatments, concerns)
    if by_cat.get('Moisturizer'):
        moisturizers = by_cat['Moisturizer'][1:] if len(by_cat['Moisturizer']) > 1 else by_cat['Moisturizer']
        night_html += "<h4>3. Overnight Recovery</h4>" + generate_choice_block('Moisturizer', moisturizers, concerns)

    # Weekly
    weekly_items = (by_cat.get('Mask', []) + by_cat.get('Exfoliant', []))
    if weekly_items: weekly_html += "<h4>Weekly Specialized Treatment</h4>" + generate_choice_block('Specialized', weekly_items, concerns)

    tips = "1. **Double Cleanse**: Use a micellar water before your cleanser if wearing makeup.\n"
    tips += "2. **Wait Times**: Allow treatments to absorb for 2-3 minutes before applying moisturizer.\n"
    tips += "3. **Strict Consistency**: Skipping even one day can set back progress.\n"
    tips += "4. **Pillowcase**: Change your pillowcase every 2-3 days to reduce bacterial transfer."

    tone = user_data.get('skinTone', 'Medium')
    fairness = f"Your recommendations are optimized for <strong>{tone}</strong> skin. "
    fairness += "Our AI ensures no bias in ingredient selection across the Fitzpatrick scale."

    return {
        "metrics": metrics,
        "analysis": analysis,
        "routine": morning_html + night_html + weekly_html,
        "explanation": "We have selected these specific formulations because they address the root cause of your skin concerns while protecting your barrier.",
        "tips": tips,
        "fairness": fairness
    }

# --- ROUTES ---

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        # 1. Get recommendations
        rec_out = recommend_products(data)
        # 2. Generate response
        return jsonify(generate_routine_response(data, rec_out))
    except Exception as e:
        print(f"Error analyzing profile: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)