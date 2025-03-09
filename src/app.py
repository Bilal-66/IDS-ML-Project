import joblib
from flask import Flask, request, jsonify
import numpy as np
import os
app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest.joblib")
model = joblib.load(model_path)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "This endpoint expects a POST request with JSON features."

    if request.method == "POST":
        try:
            data = request.get_json()
            features = data["features"]
            X = np.array(features).reshape(1, -1)

            y_pred = model.predict(X)[0]

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[0]  
                confidence = float(np.max(y_proba))  
            else:
                y_proba = None
                confidence = "N/A"

            threshold = 0.4  
            if y_proba is not None and confidence > threshold and y_pred != "normal":
                prediction = "attack"
            else:
                prediction = "normal"

            return jsonify({
                "prediction": prediction,
                "confidence": confidence
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
