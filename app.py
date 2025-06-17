from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)
model = joblib.load("sustainability_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

    pred = model.predict(img)[0]
    label = "Sustainable" if pred == 0 else "Non-Sustainable"
    return jsonify({"prediction": label})
