from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app) 

# Load models
model_dir = os.path.join(os.path.dirname(__file__), "../public/model")
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

print("Model Directory Resolved To:", os.path.abspath(model_dir))
print("Files in Model Directory:", os.listdir(os.path.abspath(model_dir)))

models = {
    "linreg": joblib.load(os.path.join(model_dir, "linreg_model.pkl")),
    "rf": joblib.load(os.path.join(model_dir, "rfr_model.pkl")),
    "gbr": joblib.load(os.path.join(model_dir, "gbr_model.pkl"))
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = data.get("input")
        model_name = data.get("model")

        if not input_data or model_name not in models:
            return jsonify({"error": "Invalid input or model name"}), 400

        X = np.array(input_data).reshape(1, -1)

        if X.shape[1] != 25:
            return jsonify({"error": "Input must contain 25 values."}), 400

        X = scaler.transform(X)

        model = models[model_name]
        raw_output = model.predict(X)[0]
        raw_output = np.maximum(raw_output, 0)

        total = np.sum(raw_output)
        if total == 0:
            normalized = [0] * len(raw_output)
        else:
            normalized = [(x / total) * 100 for x in raw_output]

        return jsonify({"portfolio": normalized})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

'''
pip install scikit-learn flask flask-cors joblib numpy
cd src
python app.py
'''