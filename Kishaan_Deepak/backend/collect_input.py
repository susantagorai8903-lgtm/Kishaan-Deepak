import os
import csv
import joblib
import pandas as pd

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'mnt', 'data')
INPUTS_FILE = os.path.join(DATA_DIR, 'inputs_collected.csv')
MODEL_FILE = os.path.join(THIS_DIR, 'model.joblib')

# Column names we'll collect
CROP_COL = 'crop_type'
REGION_COL = 'region'
TEMP_COL = 'temperature_c'
RAIN_COL = 'rainfall_mm'
HUMID_COL = 'humidity_percent'
SOIL_COL = 'soil_type'

def load_unique_values(csv_path, col):
    """Return sorted unique non-null values for column `col` in csv_path. If file/column missing, return empty list."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    # convert to str and sort nicely
    vals = [str(v).strip() for v in vals if str(v).strip() != '']
    vals = sorted(list(set(vals)), key=lambda s: s.lower())
    return vals

def choose_from(menu, prompt):
    print(f"\n{prompt}")
    for i, opt in enumerate(menu, 1):
        print(f"  {i}) {opt}")
    while True:
        choice = input('Select number (or type new value): ').strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(menu):
                return menu[idx]
            else:
                print('Invalid number, try again.')
        elif choice:
            return choice

def get_numeric(prompt, typ=float):
    while True:
        val = input(prompt).strip()
        try:
            return typ(val)
        except ValueError:
            print('Invalid number, try again.')

def load_model_if_exists():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            print('Warning: failed to load model; continuing without predictions.')
    return None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, 'indian_crop_climate_data.csv')

    # Load menu options from CSV where possible
    crop_opts = load_unique_values(csv_path, CROP_COL) or ['wheat', 'rice', 'maize']
    region_opts = load_unique_values(csv_path, REGION_COL) or ['State1', 'State2']
    soil_opts = load_unique_values(csv_path, SOIL_COL) or ['sandy', 'loamy', 'clay']

    crop = choose_from(crop_opts, 'Choose crop_type:')
    region = choose_from(region_opts, 'Choose region (state):')
    temp = get_numeric(f'Enter {TEMP_COL}: ', float)
    rain = get_numeric(f'Enter {RAIN_COL}: ', float)
    humidity = get_numeric(f'Enter {HUMID_COL}: ', float)
    soil = choose_from(soil_opts, 'Choose soil_type:')

    # Construct row using the CSV column names requested
    row = {
        CROP_COL: crop,
        REGION_COL: region,
        TEMP_COL: temp,
        RAIN_COL: rain,
        HUMID_COL: humidity,
        SOIL_COL: soil,
    }

    write_header = not os.path.exists(INPUTS_FILE)
    with open(INPUTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nSaved input row to: {INPUTS_FILE}")

    # Attempt to load model and predict. The training code normalized 'humidity_percent' to 'humidity',
    # so we rename before predicting if needed.
    model = load_model_if_exists()
    if model is not None:
        df = pd.DataFrame([row])
        # rename humidity_percent -> humidity for compatibility with some models
        if HUMID_COL in df.columns and 'humidity' not in df.columns:
            df = df.rename(columns={HUMID_COL: 'humidity'})
        try:
            pred = model.predict(df)
            print(f"Model prediction: {pred[0]}")
        except Exception as e:
            print('Could not run prediction with loaded model:', e)

if __name__ == '__main__':
    main()
