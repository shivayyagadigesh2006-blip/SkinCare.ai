from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import recommend_products
from routine import generate_response

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        print("Received user data:", data)
        
        # 1. Get recommended products based on the user's profile
        recommended_items = recommend_products(data)
        
        # 2. Generate the full JSON response using the recommended products and user profile
        response_json = generate_response(data, recommended_items)
        
        return jsonify(response_json)
        
    except Exception as e:
        print(f"Error analyzing profile: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)