import pandas as pd

def get_advice(df):
    if df.shape[0] == 0:
        return "No Data", None, None

    # Analyze all previous data for buy/sell signals
    buy_signals = df[(df['rsi'] < 35) & (df['macd'] > df['macd_signal']) & (df['sma_20'] > df['sma_50'])]
    sell_signals = df[df['rsi'] > 70]

    # Latest recommendation
    latest = df.iloc[-1]
    price = float(latest['Close'])
    if latest['rsi'] < 35 and latest['macd'] > latest['macd_signal'] and latest['sma_20'] > latest['sma_50']:
        target = round(price * 1.03, 2)
        stoploss = round(price * 0.98, 2)
        return "Buy", target, stoploss
    elif latest['rsi'] > 70:
        return "Sell", None, None
    else:
        # If no current signal, but there were previous buy/sell signals, return info about them
        if not buy_signals.empty:
            last_buy = buy_signals.iloc[-1]
            target = round(float(last_buy['Close']) * 1.03, 2)
            stoploss = round(float(last_buy['Close']) * 0.98, 2)
            return "Previous Buy", target, stoploss
        elif not sell_signals.empty:
            return "Previous Sell", None, None
        else:
            return "Hold", None, None