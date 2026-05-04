"""
routine.py — Skincare routine generation engine.
Builds morning/night/weekly routines with product recommendations,
AI reasoning, and personalized tips.
"""


def get_level(score):
    """Convert a severity score (0-10) to a human-readable level."""
    if score < 3.5:
        return "Low"
    if score < 7.0:
        return "Moderate"
    return "High"


def get_level_numeric(score):
    """Return severity as a percentage (0-100) for animated bars."""
    return round(min(100, max(0, score * 10)), 1)


# --- CONCERN-TO-INGREDIENT REASONING MAP ---
REASONING_MAP = [
    {
        'concerns': ['acne', 'breakout'],
        'ingredients': ['salicylic', 'benzoyl', 'tea tree', 'niacinamide', 'zinc'],
        'reason': "it contains acne-fighting actives that unclog pores, reduce inflammation, and prevent future breakouts you're dealing with."
    },
    {
        'concerns': ['dry', 'flak'],
        'ingredients': ['hyaluronic', 'ceramide', 'glycerin', 'squalane', 'panthenol', 'shea'],
        'reason': "its blend of deep hydrating actives will repair your moisture barrier and stop the flakiness and tightness you're experiencing."
    },
    {
        'concerns': ['pigment', 'dark spot', 'uneven'],
        'ingredients': ['vitamin c', 'niacinamide', 'alpha arbutin', 'kojic', 'tranexamic', 'licorice'],
        'reason': "it uses clinically-proven brightening agents to target your dark spots and even out your skin tone over 6-8 weeks."
    },
    {
        'concerns': ['aging', 'wrinkle', 'fine line'],
        'ingredients': ['retinol', 'peptide', 'bakuchiol', 'collagen', 'coenzyme'],
        'reason': "the anti-aging actives will boost collagen production and cellular renewal to visibly reduce your wrinkles and fine lines."
    },
    {
        'concerns': ['redness', 'rosacea', 'sensitive'],
        'ingredients': ['centella', 'aloe', 'chamomile', 'bisabolol', 'oat', 'allantoin'],
        'reason': "it contains soothing botanical actives that calm redness and strengthen your sensitive skin barrier."
    },
    {
        'concerns': ['dull'],
        'ingredients': ['vitamin c', 'aha', 'glycolic', 'lactic', 'niacinamide'],
        'reason': "its brightening formula revives dull skin by promoting gentle exfoliation and boosting radiance."
    },
    {
        'concerns': ['oil', 'shine'],
        'ingredients': ['niacinamide', 'salicylic', 'zinc', 'clay', 'charcoal'],
        'reason': "it helps regulate sebum production and minimize pores to control the excess oiliness you're experiencing."
    },
]


def _generate_reasoning(concerns_str, ingredients_str, category):
    """Generate intelligent reasoning for why a product was selected."""
    conc_lower = concerns_str.lower()
    ing_lower = ingredients_str.lower()

    for entry in REASONING_MAP:
        concern_match = any(c in conc_lower for c in entry['concerns'])
        ingredient_match = any(i in ing_lower for i in entry['ingredients'])
        if concern_match and ingredient_match:
            return entry['reason']

    # Fallback for sunscreen
    if 'sunscreen' in category.lower() or 'spf' in ing_lower:
        return "it provides critical UV protection that prevents further damage, pigmentation, and premature aging — the single most impactful step in any routine."

    # Generic fallback
    formatted_ing = ingredients_str.replace('|', ', ')
    return f"its high-rated formula featuring {formatted_ing} is optimally matched to your skin profile by our AI."


import urllib.parse

def _amazon_url(brand, category):
    query = f"{brand} {category}"
    return f"https://www.amazon.in/s?k={urllib.parse.quote_plus(query)}"

def generate_choice_block(cat, items, user_concerns):
    """Generate HTML for a product choice block with primary + alternatives."""
    if not items:
        return ""

    primary = items[0]
    alts = items[1:3]  # up to 2 alternatives

    reasoning = "This is our primary recommendation because "
    reasoning += _generate_reasoning(
        user_concerns,
        primary['Ingredients'],
        cat
    )

    primary_url = _amazon_url(primary['Brand'], primary['Category'])

    html = '<div class="choice-block">'
    html += '  <div class="primary-suggestion verified">'
    html += '    <span class="strong-badge"><span class="check-icon">✓</span> OUR VERIFIED AI SELECTION</span>'
    html += f'    <div class="prod-name"><a href="{primary_url}" target="_blank" style="color:var(--color-text); text-decoration:none;" class="amazon-link">{primary["Brand"]} ↗</a></div>'
    html += f'    <div class="prod-meta">{primary["Category"]} · ₹{primary["Price"]}</div>'
    html += f'    <div class="prod-ingredients"><strong>Key Actives:</strong> {primary["Ingredients"].replace("|", " · ")}</div>'
    html += f'    <div class="why-reason"><strong>Why this?</strong> {reasoning}</div>'
    html += '    <div class="purchase-guide">Recommended as your primary purchase for this step.</div>'
    html += '  </div>'

    if alts:
        html += '  <div class="alternatives-section">'
        html += '    <div class="alt-label">Secondary Backup Options:</div>'
        for alt in alts:
            alt_url = _amazon_url(alt['Brand'], alt['Category'])
            alt_ings = alt['Ingredients'].replace('|', ' · ')
            html += f'    <div class="alt-item"><a href="{alt_url}" target="_blank" class="alt-name amazon-link" style="text-decoration:none; color:inherit;">{alt["Brand"]} ↗</a> <span class="alt-meta">{alt["Category"]} · ₹{alt["Price"]}</span><span class="alt-ings">{alt_ings}</span></div>'
        html += '  </div>'

    html += '</div>'
    return html


def generate_routine_response(user_data, recommender_output):
    """
    Generate the full routine response from user data and recommendation output.

    Returns a dict with: metrics, analysis, routine, explanation, tips, fairness
    """
    top_items = recommender_output['top_items']
    acne, dry, pigm, aging, sens = recommender_output['severities']

    # Filter products by category
    by_cat = {}
    for item in top_items:
        cat = item['Category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(item)

    # --- Metrics (with numeric values for animated bars) ---
    metrics = {
        "acne": get_level(acne),
        "dryness": get_level(dry),
        "pigmentation": get_level(pigm),
        "aging": get_level(aging),
        "sensitivity": get_level(sens),
        "acne_pct": get_level_numeric(acne),
        "dryness_pct": get_level_numeric(dry),
        "pigmentation_pct": get_level_numeric(pigm),
        "aging_pct": get_level_numeric(aging),
        "sensitivity_pct": get_level_numeric(sens),
    }

    # --- Analysis summary ---
    st = user_data.get('skinType', 'Normal')
    concerns = user_data.get('concerns', 'general wellness')
    water = user_data.get('water', '')
    climate = user_data.get('climate', '')

    analysis = f"Based on our AI analysis, you have <strong>{st}</strong> skin with primary concerns focusing on <strong>{concerns}</strong>. "

    # Severity-specific insights
    if acne > 6.0:
        analysis += "Your profile strongly indicates an active inflammatory state — targeted anti-acne actives are critical. "
    if dry > 6.0:
        analysis += "Significant barrier disruption detected — intensive hydration is your top priority. "
    if pigm > 6.0:
        analysis += "Elevated pigmentation markers detected — consistent use of brightening actives is essential. "
    if aging > 6.0:
        analysis += "Accelerated aging indicators present — collagen-boosting and cellular renewal actives are recommended. "
    if sens > 6.0:
        analysis += "High sensitivity flagged — all products are selected to minimize irritation risk. "

    # Water intake insight
    if 'Less than 1L' in water:
        analysis += "<br><br>⚠️ <strong>Hydration Alert:</strong> Your water intake is below optimal. Increasing to 2-3L/day will significantly improve skin hydration from within."
    elif '3L+' in water:
        analysis += "<br><br>✅ <strong>Excellent hydration</strong> — your water intake supports skin barrier function."

    # Climate insight
    if 'Cold' in climate or 'dry' in climate.lower():
        analysis += " Your cold/dry climate demands extra barrier protection and heavier moisturizers."
    elif 'humid' in climate.lower() or 'Tropical' in climate:
        analysis += " Your humid climate means lighter, gel-based formulations will absorb better."

    analysis += "<br><br>A consistent, science-backed routine is non-negotiable for achieving the results you desire."

    # --- Routine Construction ---
    morning_html = ""
    night_html = ""
    weekly_html = ""

    # Morning
    if by_cat.get('Cleanser'):
        morning_html += "<h4>1. Deep Cleanse</h4>" + generate_choice_block('Cleanser', by_cat['Cleanser'], concerns)
    if by_cat.get('Serum'):
        morning_html += "<h4>2. Target Treatment</h4>" + generate_choice_block('Serum', by_cat['Serum'], concerns)
    if by_cat.get('Moisturizer'):
        morning_html += "<h4>3. Hydrate & Seal</h4>" + generate_choice_block('Moisturizer', by_cat['Moisturizer'], concerns)
    if by_cat.get('Sunscreen'):
        morning_html += "<h4>4. PROTECT (Mandatory)</h4>" + generate_choice_block('Sunscreen', by_cat['Sunscreen'], concerns)

    # Night
    if by_cat.get('Cleanser'):
        cleansers = by_cat['Cleanser'][1:] if len(by_cat['Cleanser']) > 1 else by_cat['Cleanser']
        night_html += "<h4>1. Gentle Cleanse</h4>" + generate_choice_block('Cleanser', cleansers, concerns)
    treatments = by_cat.get('Treatment', []) + by_cat.get('Exfoliant', [])
    if treatments:
        night_html += "<h4>2. Active Repair</h4>" + generate_choice_block('Treatment', treatments, concerns)
    if by_cat.get('Moisturizer'):
        moisturizers = by_cat['Moisturizer'][1:] if len(by_cat['Moisturizer']) > 1 else by_cat['Moisturizer']
        night_html += "<h4>3. Overnight Recovery</h4>" + generate_choice_block('Moisturizer', moisturizers, concerns)

    # Weekly
    weekly_items = by_cat.get('Mask', []) + by_cat.get('Exfoliant', [])
    if weekly_items:
        weekly_html += "<h4>Weekly Specialized Treatment</h4>" + generate_choice_block('Specialized', weekly_items, concerns)

    # Wrap sections
    routine_html = ""
    if morning_html:
        routine_html += '<div class="routine-phase"><div class="phase-header"><span class="phase-icon">☀️</span><span class="phase-title">Morning Protocol</span></div>' + morning_html + '</div>'
    if night_html:
        routine_html += '<div class="routine-phase"><div class="phase-header"><span class="phase-icon">🌙</span><span class="phase-title">Night Protocol</span></div>' + night_html + '</div>'
    if weekly_html:
        routine_html += '<div class="routine-phase"><div class="phase-header"><span class="phase-icon">📅</span><span class="phase-title">Weekly Protocol</span></div>' + weekly_html + '</div>'

    # --- Tips (personalized) ---
    tips = []
    tips.append("1. **Double Cleanse**: Use micellar water before your cleanser if wearing makeup or sunscreen.")
    tips.append("2. **Wait Times**: Allow serums and treatments to absorb for 2-3 minutes before moisturizer.")
    tips.append("3. **Strict Consistency**: Skipping even one day can set back progress. Results take 6-8 weeks.")
    tips.append("4. **Pillowcase**: Change your pillowcase every 2-3 days to reduce bacterial transfer.")

    # Personalized tips based on profile
    if acne > 5.0:
        tips.append("5. **Hands Off**: Avoid touching your face — bacterial transfer is a major acne trigger.")
        tips.append("6. **Phone Hygiene**: Clean your phone screen daily — it touches your face more than you think.")
    if dry > 5.0:
        tips.append("5. **Lukewarm Water**: Hot water strips your skin's natural oils. Always wash with lukewarm water.")
        tips.append("6. **Humidifier**: Consider using a humidifier, especially during winter months.")
    if pigm > 5.0:
        tips.append("5. **Reapply SPF**: Reapply sunscreen every 2-3 hours when outdoors — UV is the #1 pigmentation trigger.")
    if aging > 5.0:
        tips.append("5. **Sleep Position**: Sleeping on your back prevents sleep wrinkles from forming.")
        tips.append("6. **Antioxidant Diet**: Berries, green tea, and leafy greens boost skin's anti-aging defense.")

    tips_text = "\n".join(tips)

    # --- Fairness statement ---
    tone = user_data.get('skinTone', 'Medium')
    fairness = f"Your recommendations are optimized for <strong>{tone}</strong> skin. "
    fairness += "Our AI ensures no bias in ingredient selection across the Fitzpatrick scale. "
    fairness += "Products are ranked purely on clinical efficacy for your specific concern profile, regardless of skin tone."

    # --- Explanation ---
    explanation = "We have selected these specific formulations because they address the root cause of your skin concerns "
    explanation += "while protecting your barrier function. Each product's active ingredients are cross-referenced against "
    explanation += "clinical research databases, and our ML model predicts personalized efficacy based on profiles similar to yours."

    if not top_items:
        explanation += "<br><br>⚠️ <strong>Note:</strong> We found limited products in your exact budget range. "
        explanation += "Consider adjusting your budget for more options, or we've included the closest matches."

    return {
        "metrics": metrics,
        "analysis": analysis,
        "routine": routine_html,
        "explanation": explanation,
        "tips": tips_text,
        "fairness": fairness
    }
