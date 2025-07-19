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


        # --- Add more features ---
        try:
            df['rsi'] = ta.momentum.RSIIndicator(close=close, window=14, fillna=True).rsi()
        except Exception as e:
            df['rsi'] = pd.NA
        try:
            macd = ta.trend.MACD(close=close, fillna=True)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
        except Exception as e:
            df['macd'] = pd.NA
            df['macd_signal'] = pd.NA
        try:
            df['sma_20'] = close.rolling(window=20, min_periods=1).mean()
        except Exception as e:
            df['sma_20'] = pd.NA
        try:
            df['sma_50'] = close.rolling(window=50, min_periods=1).mean()
        except Exception as e:
            df['sma_50'] = pd.NA
        # New features
        try:
            df['sma_10'] = close.rolling(window=10, min_periods=1).mean()
        except Exception:
            df['sma_10'] = pd.NA
        try:
            df['sma_100'] = close.rolling(window=100, min_periods=1).mean()
        except Exception:
            df['sma_100'] = pd.NA
        try:
            df['ema_20'] = close.ewm(span=20, adjust=False).mean()
        except Exception:
            df['ema_20'] = pd.NA
        try:
            df['volatility'] = df['Close'].rolling(window=20, min_periods=1).std()
        except Exception:
            df['volatility'] = pd.NA
        try:
            bb = ta.volatility.BollingerBands(close=close, window=20, fillna=True)
            df['bb_bbm'] = bb.bollinger_mavg()
            df['bb_bbh'] = bb.bollinger_hband()
            df['bb_bbl'] = bb.bollinger_lband()
        except Exception:
            df['bb_bbm'] = pd.NA
            df['bb_bbh'] = pd.NA
            df['bb_bbl'] = pd.NA
        try:
            df['adx'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=close, window=14, fillna=True).adx()
        except Exception:
            df['adx'] = pd.NA
        # Volume
        if 'Volume' in df.columns:
            df['volume'] = df['Volume']
        else:
            df['volume'] = pd.NA
        # Placeholder for sector/market index/macro data
        # You can add more here as needed

        # Ensure all required columns exist
        required = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_10', 'sma_100', 'ema_20', 'volatility', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'adx', 'volume']
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