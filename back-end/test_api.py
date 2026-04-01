import requests
import json

payload = {
    "age": "25–34",
    "gender": "Female",
    "skinType": "Oily",
    "skinTone": "Medium (Type III)",
    "concerns": "Acne / breakouts",
    "diet": "High sugar / processed",
    "climate": "Hot & humid",
    "hormonal": "PCOS / hormonal acne",
    "water": "1–2L/day",
    "budget": "$20–$50 (mid-range)"
}

try:
    res = requests.post('http://localhost:5000/analyze', json=payload)
    print("Status Code:", res.status_code)
    print(json.dumps(res.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
