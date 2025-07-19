# model.py


import pandas as pd
from sklearn.linear_model import SGDClassifier
import joblib
import os

MODEL_PATH = "incremental_model.pkl"

def get_feature_data(df):
    feature_cols = [
        'rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50',
        'sma_10', 'sma_100', 'ema_20', 'volatility',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'adx', 'volume'
    ]
    # Fill missing columns with 0 if not present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0)
    return X

def train_incremental_model(df):
    df = df.dropna().copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = get_feature_data(df)
    y = df['target']
    feature_signature_path = MODEL_PATH + '.features'
    feature_cols = list(X.columns)
    # Check if model exists and if feature set matches
    if os.path.exists(MODEL_PATH) and os.path.exists(feature_signature_path):
        try:
            with open(feature_signature_path, 'r') as f:
                saved_features = f.read().split(',')
            if saved_features != feature_cols:
                # Feature set changed, delete old model
                os.remove(MODEL_PATH)
                os.remove(feature_signature_path)
        except Exception:
            # If any error, remove model to avoid mismatch
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            if os.path.exists(feature_signature_path):
                os.remove(feature_signature_path)
    # Double check: if model still exists, check feature names
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            # Check model's feature names if possible (for sklearn >=1.0)
            if hasattr(model, 'feature_names_in_'):
                if list(model.feature_names_in_) != feature_cols:
                    os.remove(MODEL_PATH)
                    if os.path.exists(feature_signature_path):
                        os.remove(feature_signature_path)
                    model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
                else:
                    pass
            else:
                pass
        except Exception:
            model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    if not os.path.exists(MODEL_PATH):
        model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    model.partial_fit(X, y, classes=[0, 1])
    joblib.dump(model, MODEL_PATH)
    # Save current feature set
    with open(feature_signature_path, 'w') as f:
        f.write(','.join(feature_cols))
    return model

# --- Hyperparameter tuning with GridSearchCV ---
from sklearn.model_selection import GridSearchCV
def tune_hyperparameters(X, y):
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['log_loss', 'hinge'],
        'max_iter': [1000, 2000]
    }
    base_model = SGDClassifier(random_state=42)
    grid = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_

def predict_incremental(df):
    if not os.path.exists(MODEL_PATH):
        return None
    model = joblib.load(MODEL_PATH)
    X = get_feature_data(df)
    return model.predict(X)
