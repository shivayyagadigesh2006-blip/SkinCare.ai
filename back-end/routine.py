def get_level(score):
    if score < 3.5: return "Low"
    if score < 7.0: return "Moderate"
    return "High"

def generate_response(user_data, recommender_output):
    top_items = recommender_output['top_items']
    acne, dry, pigm, aging, sens = recommender_output['severities']
    
    # 1. Metrics
    metrics = {
        "acne": get_level(acne),
        "dryness": get_level(dry),
        "pigmentation": get_level(pigm),
        "aging": get_level(aging),
        "sensitivity": get_level(sens)
    }
    
    # 2. Analysis
    st = user_data.get('skinType', 'Normal').lower()
    concerns = user_data.get('concerns', 'general wellness')
    analysis = f"Based on your profile, you have {st} skin with primary concerns around {concerns}. "
    if acne > 6.0:
        analysis += "Your profile indicates a higher likelihood of breakouts, likely influenced by your environment or dietary habits. "
    if dry > 6.0:
        analysis += "Your skin appears prone to dryness and flakiness, which can be exacerbated by cold or dry climates. "
    if sens > 6.0:
        analysis += "You have a high sensitivity score, meaning we need to be careful with harsh active ingredients. "
    analysis += "Overall, a consistent, gentle routine prioritizing hydration and skin-barrier repair will set a great foundation."
    
    # 3. Routine
    # Try to pick 2 items for morning and 2 for night from top 15
    morning_items = []
    night_items = []
    
    for i, item in enumerate(top_items):
        brand = item['Brand']
        ing = item['Ingredients'].replace('|', ', ')
        price = item['Price']
        inr_price = int(price * 83)
        
        step_str = f"{brand} product (Price: ₹{inr_price}, Contains: {ing})"
        if i % 2 == 0 and len(morning_items) < 2:
            morning_items.append(step_str)
        elif len(night_items) < 2:
            night_items.append(step_str)
            
    routine = "Morning:\n"
    routine += f"Step 1: Gentle Cleanser & Hydration - Apply your {morning_items[0] if morning_items else 'Gentle Cleansing Wash'}.\n"
    routine += f"Step 2: Protection - Always apply a broad-spectrum SPF 30 or higher.\n\n"
    
    routine += "Night:\n"
    routine += f"Step 1: Cleanse - Remove impurities from the day.\n"
    routine += f"Step 2: Treatment - Apply your {night_items[0] if night_items else 'Targeted Treatment'}.\n"
    if len(night_items) > 1:
        routine += f"Step 3: Moisturize - Lock in hydration with {night_items[1]}.\n"
        
    # 4. Explanation
    diet = user_data.get('diet', '').lower()
    climate = user_data.get('climate', '').lower()
    exp = f"Your {climate} climate impacts how your skin retains moisture, which is why your recommended products contain ingredients geared towards balancing this. "
    if 'sugar' in diet or 'dairy' in diet:
        exp += "Additionally, diets high in sugar or dairy can trigger inflammatory responses leading to breakouts. "
    exp += "We have selected products with specific active ingredients to target your primary concerns while respecting your budget constraints."
    
    # 5. Tips
    tips = "1. Hydration is key: Drink plenty of water throughout the day.\n"
    tips += "2. Consistency: Stick to your routine for at least 4-6 weeks to see results.\n"
    tips += "3. Sleep: Aim for 7-9 hours of quality sleep to aid skin repair.\n"
    tips += "4. Patch Test: Always patch test new products, especially with sensitive skin.\n"
    tips += "5. Sun Protection: Reapply SPF every 2 hours when outdoors."
    
    # 6. Fairness
    tone = user_data.get('skinTone', 'Medium')
    fairness = f"Your recommendations have been evaluated for {tone} skin tones. "
    if 'Deep' in tone or 'Brown' in tone or 'Dusky' in tone:
        fairness += "We've ensured that the products avoid ingredients known to cause ashiness (like certain physical SPFs) and we're careful with strong acids to prevent post-inflammatory hyperpigmentation. "
    else:
        fairness += "The selected active ingredients are generally well-tolerated and equitable for your skin tone. "
    fairness += "Please note that all skin reacts uniquely, regardless of tone."
    
    return {
        "metrics": metrics,
        "analysis": analysis,
        "routine": routine,
        "explanation": exp,
        "tips": tips,
        "fairness": fairness
    }