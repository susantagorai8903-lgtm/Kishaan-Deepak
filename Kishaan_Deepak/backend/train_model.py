import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Compute paths relative to the project root (two levels up from this file)
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
# data is stored in the `mnt/data` folder at the project root
CSV_PATH = os.path.join(PROJECT_ROOT, 'mnt', 'data', 'indian_crop_climate_data.csv')
MODEL_PATH = os.path.join(THIS_DIR, 'model.joblib')

CATEGORICAL_FEATURES = ['crop_type', 'region', 'soil_type']
# Expected numeric features; we'll normalize alternate column names after loading
NUMERIC_FEATURES = ['temperature_c', 'rainfall_mm', 'humidity', 'percent']
TARGET = 'production_tonnes_per_hectare'

def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    # Normalize some alternate column names commonly seen in the dataset
    rename_map = {}
    if 'humidity_percent' in df.columns and 'humidity' not in df.columns:
        rename_map['humidity_percent'] = 'humidity'
    # some files may have slightly different target column naming
    if 'production_tonnes_per_hectare' in df.columns and TARGET not in df.columns:
        rename_map['production_tonnes_per_hectare'] = TARGET
    if 'production_tonnes_per_hectare' in df.columns and TARGET not in df.columns:
        rename_map['production_tonnes_per_hectare'] = TARGET
    if 'production_tonnes_per_hectare' in df.columns and TARGET not in df.columns:
        rename_map['production_tonnes_per_hectare'] = TARGET
    # also map a 'percent' style column if present
    if 'humidity_percent' in df.columns and 'percent' not in df.columns:
        # don't overwrite humidity mapping; only add if percent missing
        if 'percent' not in df.columns:
            # guess that 'humidity_percent' might correspond to 'percent' in some exports
            pass
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def validate_columns(df):
    # Ensure target column exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in CSV. Available columns: {list(df.columns)}")
    # Determine which expected features are present
    present_cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    present_nums = [c for c in NUMERIC_FEATURES if c in df.columns]
    if not present_cats and not present_nums:
        raise ValueError(f"No expected feature columns found in CSV. Expected some of: {CATEGORICAL_FEATURES + NUMERIC_FEATURES}. Available: {list(df.columns)}")
    return present_cats, present_nums

def build_pipeline():
    # Return a factory that builds a pipeline for the actual feature lists
    def factory(numeric_features, categorical_features):
        transformers = []
        if numeric_features:
            numeric_pipe = Pipeline([('scaler', StandardScaler())])
            transformers.append(('num', numeric_pipe, numeric_features))
        if categorical_features:
            # create OneHotEncoder compatible with different sklearn versions
            try:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            categorical_pipe = Pipeline([('ohe', ohe)])
            transformers.append(('cat', categorical_pipe, categorical_features))
        preprocessor = ColumnTransformer(transformers)
        pipeline = Pipeline([('preprocessor', preprocessor), ('reg', LinearRegression())])
        return pipeline

    return factory

def train_and_save(csv_path=CSV_PATH, model_path=MODEL_PATH):
    print('Loading data from', csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Make sure you run this from the project root or pass the correct --csv path.")
    df = load_data(csv_path)
    present_cats, present_nums = validate_columns(df)
    # Warn if some expected columns are missing and will be ignored
    missing = [c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c not in df.columns]
    if missing:
        print(f"Warning: some expected columns are missing and will be ignored: {missing}")
    feature_cols = present_cats + present_nums
    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline_factory = build_pipeline()
    pipeline = pipeline_factory(present_nums, present_cats)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'Training complete. Test MSE: {mse:.4f}, R2: {r2:.4f}')
    joblib.dump(pipeline, model_path)
    print(f'Model saved to: {model_path}')

if __name__ == '__main__':
    train_and_save()
