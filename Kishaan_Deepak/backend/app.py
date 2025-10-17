import os
import joblib
from flask import Flask, render_template, request, jsonify
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
DATA_CSV = os.path.abspath(os.path.join(BASE_DIR, '..', 'mnt', 'data', 'indian_crop_climate_data.csv'))

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        model = joblib.load(MODEL_PATH)
    return model

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/options')
def options():
    # Return unique values for crop_type, region, soil_type from CSV to populate front-end menus
    cols = {'crop_type': [], 'region': [], 'soil_type': []}
    try:
        df = pd.read_csv(DATA_CSV)
        for c in cols.keys():
            if c in df.columns:
                vals = df[c].dropna().unique().tolist()
                cols[c] = sorted([str(v).strip() for v in vals if str(v).strip() != ''], key=lambda s: s.lower())
    except Exception:
        # If CSV missing or unreadable, return empty lists (frontend will fall back to defaults)
        pass
    return jsonify(cols)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or request.form.to_dict()
    # required fields for the web form
    required = ['crop_type', 'region', 'temperature_c', 'rainfall_mm', 'humidity_percent', 'soil_type']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': 'Missing fields', 'missing': missing}), 400
    try:
        # Build a row dict for prediction
        row = {}
        row['crop_type'] = data['crop_type']
        row['region'] = data['region']
        # numeric conversions
        row['temperature_c'] = float(data['temperature_c'])
        row['rainfall_mm'] = float(data['rainfall_mm'])
        # accept humidity_percent from the form and also provide 'humidity' if model expects it
        humidity_percent = float(data['humidity_percent'])
        row['humidity_percent'] = humidity_percent
        row['soil_type'] = data['soil_type']

        # Prepare DataFrame for model: some models expect 'humidity' column name, so map if needed
        X = pd.DataFrame([row])
        if 'humidity' not in X.columns and 'humidity_percent' in X.columns:
            X = X.rename(columns={'humidity_percent': 'humidity'})
        # provide a 'percent' column if model expects it
        if 'percent' not in X.columns:
            X['percent'] = 0.0

        pipe = load_model()
        pred = pipe.predict(X)[0]
        return jsonify({'prediction_tonnes_per_hectare': round(float(pred), 4)})
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
