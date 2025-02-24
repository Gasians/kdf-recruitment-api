from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("recruitment_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_new = np.array([[data['age'], data['physical_fitness_score'], data['height'], data['weight'], data['experience_years']]])
    prediction = model.predict(X_new)[0]
    return jsonify({'selected': int(prediction)})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to port 10000 for Render
    app.run(host="0.0.0.0", port=port, debug=True)

