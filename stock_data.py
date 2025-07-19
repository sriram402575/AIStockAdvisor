import yfinance as yf
import pandas as pd
import ta

def fetch_stock_data(symbol, period='6mo', interval='1d'):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        if 'Close' not in df.columns:
            raise ValueError(f"'Close' column missing for {symbol}")

        df.reset_index(inplace=True)
        close = df['Close'].squeeze()
        print("[DEBUG] close type:", type(close), "shape:", getattr(close, 'shape', None))

        # Indicator calculations (as before)
        try:
            df['rsi'] = ta.momentum.RSIIndicator(close=close, window=14, fillna=True).rsi()
            print("[DEBUG] RSI calculated, columns now:", df.columns.tolist())
        except Exception as e:
            print("[ERROR] RSI calculation failed:", e)
            df['rsi'] = pd.NA
        try:
            macd = ta.trend.MACD(close=close, fillna=True)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            print("[DEBUG] MACD calculated, columns now:", df.columns.tolist())
        except Exception as e:
            print("[ERROR] MACD calculation failed:", e)
            df['macd'] = pd.NA
            df['macd_signal'] = pd.NA
        try:
            df['sma_20'] = close.rolling(window=20, min_periods=1).mean()
            print("[DEBUG] SMA 20 calculated, columns now:", df.columns.tolist())
        except Exception as e:
            print("[ERROR] SMA 20 calculation failed:", e)
            df['sma_20'] = pd.NA
        try:
            df['sma_50'] = close.rolling(window=50, min_periods=1).mean()
            print("[DEBUG] SMA 50 calculated, columns now:", df.columns.tolist())
        except Exception as e:
            print("[ERROR] SMA 50 calculation failed:", e)
            df['sma_50'] = pd.NA

        # Ensure all required columns exist
        required = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50']
        for col in required:
            if col not in df.columns:
                df[col] = pd.NA

        print("[DEBUG] DataFrame columns before dropna:", df.columns.tolist())
        print("[DEBUG] DataFrame head before dropna:\n", df.head())

        # Drop initial NaNs (warm-up period)
        df = df.dropna(subset=required)
        if df.empty:
            raise ValueError(f"Not enough data to compute indicators for {symbol}")

        return df
    except Exception as e:
        print(f"[ERROR] {e}")
        raise RuntimeError(f"Error fetching or processing stock data: {e}")