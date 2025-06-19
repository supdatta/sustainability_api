from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "API is live"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # example: {"features": [1, 2, 3, 4]}
    prediction = model.predict([data])
    return jsonify({'prediction': prediction.tolist()})

