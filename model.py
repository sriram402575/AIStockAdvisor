# model.py

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_model(df):
    df = df.dropna().copy()

    # Create binary target: 1 if price will go up next day, else 0
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Feature set
    feature_cols = ['rsi', 'macd', 'sma_20', 'sma_50']
    X = df[feature_cols]
    y = df['target']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model
