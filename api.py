from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("recruitment_model.pkl")  # Ensure this file exists

@app.route('/predict', methods=['POST'])  # ✅ Make sure 'POST' is included
def predict():
    try:
        data = request.json
        X_new = np.array([[data['age'], data['physical_fitness_score'], data['height'], data['weight'], data['experience_years']]])
        prediction = model.predict(X_new)[0]
        return jsonify({'selected': int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))  # ✅ Ensure correct port for Render
    app.run(host="0.0.0.0", port=port, debug=True)

