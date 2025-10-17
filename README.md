# Kishaan Deepak — Crop Yield Prediction (Full-stack)

## Overview
This repository is a full-stack crop yield prediction app. It provides:

- A Flask backend (`backend/app.py`) that serves a web UI and exposes endpoints for options and prediction.
- A simple machine-learning training script (`backend/train_model.py`) that trains a Linear Regression model and saves it to `backend/model.joblib`.
- A responsive frontend (`frontend/templates/index.html`, `frontend/static/*`) that allows users to select crop, region, soil and enter climate values, then request a prediction.
- A CLI helper script (`backend/collect_input.py`) to collect inputs via terminal and store them in `mnt/data/inputs_collected.csv`.

## Project layout

```
.
├── backend/
│   ├── app.py                  # Flask app (serves UI, /options, /predict)
│   ├── train_model.py          # Train model and save to model.joblib
│   ├── collect_input.py        # CLI input collector (menu-driven)
│   ├── model.joblib            # trained model (created after training)
│   └── requirements.txt        # Python dependencies (Flask, pandas, scikit-learn...)
├── frontend/
│   ├── templates/
│   │   └── index.html          # Web UI (form)
│   └── static/
│       ├── css/style.css       # styles
│       └── js/main.js          # client-side logic (fetch /options, submit /predict)
└── mnt/
    └── data/
        ├── indian_crop_climate_data.csv   # training dataset (required)
        └── inputs_collected.csv           # appended inputs from UI/CLI

```

## Setup (development)

1. Create and activate a virtual environment

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1    # Windows PowerShell
# or for cmd.exe: venv\Scripts\activate.bat
# or for Unix/macOS: source venv/bin/activate
```

2. Install dependencies

```powershell
pip install -r backend/requirements.txt
# if you don't have requirements.txt, at minimum install:
pip install flask pandas scikit-learn joblib
```

3. Place your dataset at:

```
mnt/data/indian_crop_climate_data.csv
```

The UI reads unique values for `crop_type`, `region`, and `soil_type` from this CSV to populate menus. Ensure the CSV contains at least these columns (column names are case-sensitive):

- crop_type
- region
- temperature_c
- rainfall_mm
- humidity_percent  (or `humidity` — training code handles common variants)
- soil_type
- production_tonnes_per_hectare

If your CSV uses `humidity` instead of `humidity_percent`, the backend normalizes names when training and the frontend maps `humidity_percent` → `humidity` before predicting.

## Training the model

Train and save the model (this creates `backend/model.joblib`):

```powershell
python backend/train_model.py
```

After training you should see `backend/model.joblib` created.

## Run the web app

```powershell
python backend/app.py
```

Visit `http://127.0.0.1:5000/` in your browser. The form will load menu options from the dataset (crop, region, soil). Enter numeric climate values and click `Predict Yield`.

The frontend calls `/predict` and displays the result returned by the model.

## CLI collector

You can also collect inputs from a terminal using the menu-driven CLI:

```powershell
python backend/collect_input.py
```

This appends entries to `mnt/data/inputs_collected.csv` and will attempt to run the model for a prediction if `backend/model.joblib` exists.

## API

### GET /options
Returns JSON lists of unique values from the dataset for `crop_type`, `region`, and `soil_type` (used to populate the frontend selects).

Example response:

```json
{
  "crop_type": ["Rice","Wheat","Maize"],
  "region": ["West Bengal","Punjab"],
  "soil_type": ["loamy","sandy"]
}
```

### POST /predict
Expects JSON with these fields (sent by the frontend form):

- crop_type
- region
- temperature_c (number)
- rainfall_mm (number)
- humidity_percent (number)
- soil_type

The backend maps `humidity_percent` to `humidity` if the model expects it. It also fills a default `percent` column (0.0) if your model expects it.

Response example:

```json
{ "prediction_tonnes_per_hectare": 5.2345 }
```

## Troubleshooting

- If `/options` returns empty lists, confirm `mnt/data/indian_crop_climate_data.csv` exists and is readable.
- If `/predict` returns model errors, verify `backend/model.joblib` exists and that the model's input columns match the form fields. I can add a schema extractor to save exact feature names after training.

## Next improvements (suggested)

- Add form validation ranges (e.g., humidity 0–100).
- Make selects searchable (use a small JS plugin) if region list is long.
- Add a schema file after training that records exact model input feature names and types; frontend can use that for validation.
- Containerize with Docker for easy deployment.

## License
MIT
