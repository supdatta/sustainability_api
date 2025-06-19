from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model safely with error handling
MODEL_PATH = 'sustainability_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print(f"[ERROR] Model file not found at {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[ERROR] Failed to load model: {e}")

@app.route('/')
def home():
    return "Sustainability Prediction API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json.get('features')
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid input, expected JSON with list of features.'}), 400

        prediction = model.predict([data])
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

