"""
validators.py — Input validation for SkinCare.AI user profiles.
Enforces whitelist values for all fields, returns structured error dicts.
"""

# --- Allowed values for each field (whitelist) ---
ALLOWED_AGE = ['13–17', '18–24', '25–34', '35–44', '45–54', '55+']
ALLOWED_GENDER = ['Female', 'Male', 'Non-binary', 'Prefer not to say']
ALLOWED_SKIN_TYPE = ['Oily', 'Dry', 'Combination', 'Normal', 'Sensitive']
ALLOWED_SKIN_TONE = [
    'Very fair (Type I)', 'Fair (Type II)', 'Medium (Type III)',
    'Olive (Type IV)', 'Brown (Type V)', 'Deep (Type VI)'
]
ALLOWED_CONCERNS = [
    'Acne / breakouts', 'Dryness / flakiness', 'Oiliness / shine',
    'Pigmentation / dark spots', 'Anti-aging / wrinkles', 'Redness / rosacea',
    'Uneven texture', 'Dullness'
]
ALLOWED_DIET = [
    'Balanced / healthy', 'High sugar / processed', 'Dairy-heavy',
    'Plant-based / vegan', 'High protein / gym-focused'
]
ALLOWED_CLIMATE = [
    'Hot & humid', 'Hot & dry', 'Temperate / mild', 'Cold & dry', 'Tropical'
]
ALLOWED_HORMONAL_FEMALE = [
    'Regular cycles, no issues', 'PCOS / hormonal acne',
    'Pregnant / postpartum', 'Menopause / perimenopause',
    'On hormonal contraception'
]
ALLOWED_HORMONAL_MALE = [
    'Balanced / Healthy', 'High Testosterone (potential oiliness)',
    'Low Testosterone (potential dryness)', 'Stress-related hormonal spikes',
    'Not applicable'
]
ALLOWED_HORMONAL_NB = [
    'Hormonal balance / Regular', 'Hormonal-related acne',
    'Hormone Replacement Therapy (HRT)', 'Other / Fluctuating',
    'Not applicable'
]
ALLOWED_HORMONAL_PNTS = [
    'Hormonal balance / Regular', 'Hormonal-related acne',
    'Other / Fluctuating', 'Not applicable'
]
ALLOWED_WATER = ['Less than 1L/day', '1–2L/day', '2–3L/day', '3L+ per day']
ALLOWED_BUDGET = [
    'Under ₹500 (Value)', '₹500–₹1000 (Budget Friendly)',
    '₹1000–₹2500 (Mid-range)', '₹2500–₹5000 (Premium)',
    '₹5000–₹10000 (High-end)', '₹10000+ (Luxury)'
]

# Map gender → allowed hormonal options
HORMONAL_BY_GENDER = {
    'Female': ALLOWED_HORMONAL_FEMALE,
    'Male': ALLOWED_HORMONAL_MALE,
    'Non-binary': ALLOWED_HORMONAL_NB,
    'Prefer not to say': ALLOWED_HORMONAL_PNTS,
}

# Required fields
REQUIRED_FIELDS = [
    'age', 'gender', 'skinType', 'skinTone', 'concerns',
    'diet', 'climate', 'hormonal', 'water', 'budget'
]


def _sanitize_string(value):
    """Strip and return a string; return empty string for non-strings."""
    if not isinstance(value, str):
        return ''
    return value.strip()


def validate_profile(data):
    """
    Validate a user profile dict.

    Returns:
        dict: { 'valid': bool, 'errors': { field: message }, 'sanitized': { cleaned data } }
    """
    if not isinstance(data, dict):
        return {
            'valid': False,
            'errors': {'_general': 'Invalid request format. Expected a JSON object.'},
            'sanitized': {}
        }

    errors = {}
    sanitized = {}

    # --- Check required fields ---
    for field in REQUIRED_FIELDS:
        raw = data.get(field)
        if raw is None or (isinstance(raw, str) and raw.strip() == ''):
            errors[field] = f'{field} is required.'

    # --- Age ---
    age = _sanitize_string(data.get('age', ''))
    if age and age not in ALLOWED_AGE:
        errors['age'] = f'Invalid age range. Allowed: {", ".join(ALLOWED_AGE)}'
    sanitized['age'] = age

    # --- Gender ---
    gender = _sanitize_string(data.get('gender', ''))
    if gender and gender not in ALLOWED_GENDER:
        errors['gender'] = f'Invalid gender. Allowed: {", ".join(ALLOWED_GENDER)}'
    sanitized['gender'] = gender

    # --- Skin Type ---
    skin_type = _sanitize_string(data.get('skinType', ''))
    if skin_type and skin_type not in ALLOWED_SKIN_TYPE:
        errors['skinType'] = f'Invalid skin type. Allowed: {", ".join(ALLOWED_SKIN_TYPE)}'
    sanitized['skinType'] = skin_type

    # --- Skin Tone ---
    skin_tone = _sanitize_string(data.get('skinTone', ''))
    if skin_tone and skin_tone not in ALLOWED_SKIN_TONE:
        errors['skinTone'] = f'Invalid skin tone.'
    sanitized['skinTone'] = skin_tone

    # --- Concerns (comma-separated string or single string) ---
    raw_concerns = data.get('concerns', '')
    if isinstance(raw_concerns, list):
        concerns_str = ', '.join([c.strip() for c in raw_concerns if isinstance(c, str)])
    else:
        concerns_str = _sanitize_string(raw_concerns)

    if concerns_str:
        concern_list = [c.strip() for c in concerns_str.split(',') if c.strip()]
        invalid_concerns = [c for c in concern_list if c not in ALLOWED_CONCERNS]
        if invalid_concerns:
            errors['concerns'] = f'Invalid concern(s): {", ".join(invalid_concerns)}'
    sanitized['concerns'] = concerns_str

    # --- Diet ---
    diet = _sanitize_string(data.get('diet', ''))
    if diet and diet not in ALLOWED_DIET:
        errors['diet'] = f'Invalid diet type.'
    sanitized['diet'] = diet

    # --- Climate ---
    climate = _sanitize_string(data.get('climate', ''))
    if climate and climate not in ALLOWED_CLIMATE:
        errors['climate'] = f'Invalid climate type.'
    sanitized['climate'] = climate

    # --- Hormonal status (depends on gender) ---
    hormonal = _sanitize_string(data.get('hormonal', ''))
    if hormonal and gender:
        allowed_hormonal = HORMONAL_BY_GENDER.get(gender, [])
        if hormonal not in allowed_hormonal:
            errors['hormonal'] = f'Invalid hormonal status for selected gender.'
    sanitized['hormonal'] = hormonal

    # --- Water intake ---
    water = _sanitize_string(data.get('water', ''))
    if water and water not in ALLOWED_WATER:
        errors['water'] = f'Invalid water intake level.'
    sanitized['water'] = water

    # --- Budget ---
    budget = _sanitize_string(data.get('budget', ''))
    if budget and budget not in ALLOWED_BUDGET:
        errors['budget'] = f'Invalid budget range.'
    sanitized['budget'] = budget

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'sanitized': sanitized
    }
