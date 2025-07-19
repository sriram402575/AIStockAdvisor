


import plotly.graph_objs as go
import pandas as pd
import streamlit as st
from stock_data import fetch_stock_data
from advisor import get_advice
from model import train_incremental_model, predict_incremental

# --- Caching for speed ---
@st.cache_data(show_spinner=False, max_entries=50)
def cached_fetch_stock_data(symbol):
    return fetch_stock_data(symbol)

@st.cache_data(show_spinner=False, max_entries=50)
def cached_get_advice(symbol):
    df = cached_fetch_stock_data(symbol)
    return get_advice(df)

st.set_page_config(page_title="AI Stock Advisor", page_icon="📈", layout="wide")



# --- Enhanced CSS for Modern, Beautiful Black Theme ---
st.markdown("""
<style>
body, .stApp { background: linear-gradient(135deg, #181c24 0%, #232a36 100%) !important; }
.big-title { font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 2.8em; font-weight: bold; color: #e3e6f0; letter-spacing: 1.5px; margin-bottom: 0.1em; text-shadow: 0 2px 8px #232a36; }
.subtitle { font-size: 1.18em; color: #b2b7d1; margin-bottom: 1.2em; font-family: 'Segoe UI', 'Roboto', sans-serif; }
.stock-btn { margin: 2px 4px 2px 0; font-weight: 600; border-radius: 10px; background: linear-gradient(90deg,#232a36 60%,#181c24 100%); color: #e3e6f0; border: 1.5px solid #3a4052; box-shadow: 0 2px 8px #232a36; transition: 0.2s; }
.stock-btn:hover { background: #283593; color: #fff; border: 1.5px solid #1976d2; }
.highlight-buy { background: linear-gradient(90deg,#1b3c1b 60%,#388e3c 100%); border-radius: 12px; padding: 16px; font-size: 1.15em; border: 1.5px solid #81c784; box-shadow: 0 2px 12px #1b3c1b; color: #e3e6f0; }
.highlight-sell { background: linear-gradient(90deg,#3c1b1b 60%,#d32f2f 100%); border-radius: 12px; padding: 16px; font-size: 1.15em; border: 1.5px solid #e57373; box-shadow: 0 2px 12px #3c1b1b; color: #e3e6f0; }
.section-title { font-size: 1.35em; font-weight: 700; color: #90caf9; margin-top: 1.7em; margin-bottom: 0.7em; letter-spacing: 0.5px; border-left: 5px solid #1976d2; padding-left: 10px; background: #232a36; border-radius: 6px; }
.indicator-box { background: linear-gradient(90deg,#232a36 60%,#181c24 100%); border-radius: 10px; padding: 14px 20px; margin-bottom: 12px; border: 1.5px solid #3a4052; font-size: 1.08em; box-shadow: 0 2px 8px #232a36; color: #e3e6f0; }
.stButton>button { font-weight: 600; border-radius: 10px; background: linear-gradient(90deg,#232a36 60%,#181c24 100%); color: #e3e6f0; border: 1.5px solid #3a4052; box-shadow: 0 2px 8px #232a36; transition: 0.2s; }
.stButton>button:hover { background: #283593; color: #fff; border: 1.5px solid #1976d2; }
.stDataFrame, .stTable { border-radius: 10px; background: #232a36; border: 1.5px solid #3a4052; box-shadow: 0 2px 8px #232a36; color: #e3e6f0; }
.stAlert, .stInfo, .stSuccess, .stWarning, .stError { border-radius: 10px !important; color: #e3e6f0 !important; background: #232a36 !important; }
.metric-card { background: #181c24; border-radius: 12px; box-shadow: 0 2px 12px #232a36; padding: 18px 20px; margin-bottom: 18px; border: 1.5px solid #3a4052; color: #e3e6f0; }
.divider { height: 2px; background: linear-gradient(90deg,#1976d2 0,#3a4052 100%); border-radius: 2px; margin: 18px 0; }
.heatmap-title { font-size: 1.1em; font-weight: 600; color: #90caf9; margin-bottom: 0.5em; }
.buy-signal { color: #81c784; font-weight: bold; }
.sell-signal { color: #e57373; font-weight: bold; }
.hold-signal { color: #fbc02d; font-weight: bold; }
.stTextInput>div>input { border-radius: 8px; border: 1.5px solid #3a4052; background: #232a36; color: #e3e6f0; }
.stTextInput>div>input:focus { border: 1.5px solid #1976d2; background: #181c24; color: #e3e6f0; }
.stTable th { background: #232a36; color: #90caf9; font-weight: 700; }
.stTable td { background: #181c24; color: #e3e6f0; }
.stDataFrame th { background: #232a36; color: #90caf9; font-weight: 700; }
.stDataFrame td { background: #181c24; color: #e3e6f0; }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="big-title">📈 AI Stock Advisor <span style="font-size:0.7em; color:#283593;">(India)</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get actionable stock recommendations, news, fundamentals, ML predictions, and more!</div>', unsafe_allow_html=True)

# --- Search box under main title ---
st.markdown("<div style='margin-bottom: 1.5em'></div>", unsafe_allow_html=True)
search_stock = st.text_input("Enter NSE Symbol (e.g. RELIANCE)", value="RELIANCE", key="main_search")

# Remove duplicate set_page_config and use a more stylish title
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# --- Sidebar: Stock Search and Top Buy Suggestions ---
suggested_stocks = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT",
    "ITC", "ASIANPAINT", "AXISBANK", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO"
]

import time
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

def get_mock_sentiment_trend(symbol, days=7):
    # Simulate fetching sentiment scores for Twitter and Reddit for the last N days
    np.random.seed(abs(hash(symbol)) % 2**32)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    twitter_scores = np.random.uniform(-1, 1, days)
    reddit_scores = np.random.uniform(-1, 1, days)
    return pd.DataFrame({
        'Date': dates,
        'Twitter': twitter_scores,
        'Reddit': reddit_scores
    })


with st.sidebar:
    st.markdown("### 💡 Top Buy Suggestions")
    # Precompute buy suggestions for sidebar, show loading spinner
    with st.spinner("Screening top stocks..."):
        sidebar_buy_suggestions = []
        sidebar_errors = []
        for stock in suggested_stocks:
            try:
                dec, t, sl = cached_get_advice(stock + ".NS")
                if dec == "Buy":
                    sidebar_buy_suggestions.append(f"{stock}: Target ₹{t} | Stop-loss ₹{sl}")
            except Exception as e:
                sidebar_errors.append(f"{stock}")
        time.sleep(0.2)  # Small delay for better UX
    if sidebar_buy_suggestions:
        for s in sidebar_buy_suggestions:
            st.success(s)
    else:
        st.warning("No current Buy signals among suggested stocks.\n\nTip: Try clicking 'Analyze Stock' for a specific stock, or check back later as signals update daily.")
        if sidebar_errors:
            st.caption(f"Note: Data unavailable for: {', '.join(sidebar_errors)}")

# --- Main area: Stock selection and analysis ---


# --- Stock Search Section ---
st.markdown("<div class='section-title'>🔍 Stock Search</div>", unsafe_allow_html=True)
user_input = search_stock
symbol = user_input.strip().upper() + ".NS"

# Search button (replaces quick picker)
search_clicked = st.button("🔎 Search & Analyze Stock", use_container_width=True)



if search_clicked:
    with st.spinner("Fetching data and analyzing..."):
        try:
            import yfinance as yf
            from datetime import datetime
            import numpy as np
            # --- Main Analysis ---
            df = cached_fetch_stock_data(symbol)
            decision, target, stoploss = get_advice(df)  # Not cached here, as it uses the latest df
            latest = df.iloc[-1]

            st.markdown(f"<div class='section-title'>📊 Recommendation for <span style='color:#1a237e'>{user_input.upper()}</span></div>", unsafe_allow_html=True)
            ind_cols = st.columns(3)
            with ind_cols[0]:
                st.markdown(f"<div class='metric-card'><span style='font-size:1.1em;'>📊 RSI</span><br><b>{latest['rsi']:.2f}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><span style='font-size:1.1em;'>📈 MACD</span><br><b>{latest['macd']:.2f}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><span style='font-size:1.1em;'>📉 MACD Signal</span><br><b>{latest['macd_signal']:.2f}</b></div>", unsafe_allow_html=True)
            with ind_cols[1]:
                st.markdown(f"<div class='metric-card'><span style='font-size:1.1em;'>🟦 SMA 20</span><br><b>{latest['sma_20']:.2f}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><span style='font-size:1.1em;'>🟪 SMA 50</span><br><b>{latest['sma_50']:.2f}</b></div>", unsafe_allow_html=True)
            with ind_cols[2]:
                st.markdown(f"<div class='metric-card'><span style='font-size:1.1em;'>💰 Close Price</span><br><b>{latest['Close']:.2f}</b></div>", unsafe_allow_html=True)

            # --- Incremental ML Classification ---
            st.markdown("<div class='section-title'>🤖 Incremental ML Buy/Sell Classification (SGDClassifier)</div>", unsafe_allow_html=True)
            # Train or update the incremental model with the latest data
            model = train_incremental_model(df)
            # Predict for the latest row
            pred = predict_incremental(df.tail(1))
            label_map = {1: "Buy", 0: "Sell/Hold"}
            if pred is not None:
                st.info(f"Incremental ML Model Suggestion: {label_map.get(pred[0], 'Hold')}")
            else:
                st.info("No incremental model found yet. It will be created now.")

            # --- Fundamental Data (with advanced metrics and sector comparison) ---
            st.markdown("<div class='section-title'>📑 Key Fundamentals & Sector Comparison</div>", unsafe_allow_html=True)
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                # Basic metrics
                fundamentals = {
                    "P/E Ratio": info.get("trailingPE", "-"),
                    "EPS": info.get("trailingEps", "-"),
                    "Market Cap": info.get("marketCap", "-"),
                    "Book Value": info.get("bookValue", "-"),
                    "Dividend Yield": info.get("dividendYield", "-"),
                    "52W High": info.get("fiftyTwoWeekHigh", "-"),
                    "52W Low": info.get("fiftyTwoWeekLow", "-"),
                    "Sector": info.get("sector", "-"),
                    "Industry": info.get("industry", "-"),
                }
                # Advanced metrics
                fundamentals["ROE (Return on Equity)"] = info.get("returnOnEquity", "-")
                fundamentals["Debt/Equity"] = info.get("debtToEquity", "-")
                fundamentals["Revenue Growth (YoY)"] = info.get("revenueGrowth", "-")
                fundamentals["Earnings Growth (YoY)"] = info.get("earningsGrowth", "-")
                fundamentals["Profit Margin"] = info.get("profitMargins", "-")
                fundamentals["Operating Margin"] = info.get("operatingMargins", "-")
                fundamentals["PEG Ratio"] = info.get("pegRatio", "-")
                fundamentals["Beta"] = info.get("beta", "-")
                st.table(pd.DataFrame(list(fundamentals.items()), columns=["Metric", "Value"]))

                # Sector comparison (simple: average of sector peers in suggested_stocks)
                sector = info.get("sector", None)
                if sector:
                    sector_peers = []
                    for peer in suggested_stocks:
                        try:
                            peer_info = yf.Ticker(peer + ".NS").info
                            if peer_info.get("sector", None) == sector:
                                sector_peers.append(peer_info)
                        except Exception:
                            continue
                    if len(sector_peers) > 2:
                        avg_pe = sum([p.get("trailingPE", 0) or 0 for p in sector_peers if p.get("trailingPE")]) / len([p for p in sector_peers if p.get("trailingPE")])
                        avg_roe = sum([p.get("returnOnEquity", 0) or 0 for p in sector_peers if p.get("returnOnEquity")]) / len([p for p in sector_peers if p.get("returnOnEquity")])
                        avg_de = sum([p.get("debtToEquity", 0) or 0 for p in sector_peers if p.get("debtToEquity")]) / len([p for p in sector_peers if p.get("debtToEquity")])
                        st.info(f"**Sector Averages ({sector}):**  P/E: {avg_pe:.2f} | ROE: {avg_roe:.2f} | Debt/Equity: {avg_de:.2f}")
            except Exception as e:
                st.info(f"Could not fetch fundamentals: {e}")

            # --- News & Social Sentiment Analysis ---
            st.markdown("<div class='section-title'>📰 Latest News & Social Sentiment</div>", unsafe_allow_html=True)
            try:
                import requests
                news = ticker.get_news()
                if news:
                    def finbert_sentiment(text):
                        api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
                        headers = {"Authorization": "Bearer hf_..."}  # <-- Insert your HuggingFace token if you have one, else remove this line for public
                        payload = {"inputs": text}
                        try:
                            response = requests.post(api_url, json=payload, timeout=8)
                            if response.status_code == 200:
                                result = response.json()
                                if isinstance(result, list) and result and isinstance(result[0], list):
                                    result = result[0]
                                if isinstance(result, list) and result and 'label' in result[0]:
                                    label = result[0]['label']
                                    score = result[0]['score']
                                    if label == 'positive':
                                        return f"🟢 Positive ({score:.2f})"
                                    elif label == 'negative':
                                        return f"🔴 Negative ({score:.2f})"
                                    else:
                                        return f"⚪ Neutral ({score:.2f})"
                        except Exception:
                            pass
                        return "⚪ Neutral"
                    count = 0
                    for n in news:
                        # Defensive: skip if 'title' or 'link' missing
                        title = n.get('title')
                        link = n.get('link')
                        provider = n.get('provider', 'Unknown')
                        pubtime = n.get('providerPublishTime', '')
                        summary = n.get('summary', '')
                        if not title or not link:
                            continue
                        text = f"{title} {summary}"
                        sentiment = finbert_sentiment(text)
                        # Defensive: format date if possible
                        try:
                            if isinstance(pubtime, (int, float)):
                                from datetime import datetime
                                pubtime_fmt = datetime.fromtimestamp(pubtime).strftime('%Y-%m-%d')
                            elif hasattr(pubtime, 'strftime'):
                                pubtime_fmt = pubtime.strftime('%Y-%m-%d')
                            else:
                                pubtime_fmt = str(pubtime)
                        except Exception:
                            pubtime_fmt = ''
                        st.markdown(f"- [{title}]({link}) ({provider}, {pubtime_fmt})")
                        st.caption(f"Sentiment: {sentiment}")
                        count += 1
                        if count >= 5:
                            break
                else:
                    st.info("No recent news found.")
            except Exception as e:
                st.info(f"Could not fetch news or sentiment: {e}")

            # --- Twitter & Reddit Sentiment Trend ---
            st.markdown("<div class='section-title'>🐦 Twitter & Reddit Sentiment Trend</div>", unsafe_allow_html=True)
            sentiment_df = get_mock_sentiment_trend(symbol)
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Scatter(x=sentiment_df['Date'], y=sentiment_df['Twitter'], mode='lines+markers', name='Twitter', line=dict(color='#1da1f2')))
            fig_sent.add_trace(go.Scatter(x=sentiment_df['Date'], y=sentiment_df['Reddit'], mode='lines+markers', name='Reddit', line=dict(color='#ff4500')))
            fig_sent.update_layout(
                title=f"Twitter & Reddit Sentiment Trend for {symbol}",
                xaxis_title="Date",
                yaxis_title="Sentiment Score (-1 to 1)",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            # --- Price Chart ---
            st.markdown("<div class='section-title'>📈 Price Chart</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['sma_20'], mode='lines', name='SMA 20'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['sma_50'], mode='lines', name='SMA 50'))
            fig.update_layout(title=f"{user_input.upper()} Price Chart", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # --- Backtest summary ---
            st.markdown("<div class='section-title'>⏳ Backtest Buy & Hold vs. Strategy</div>", unsafe_allow_html=True)
            buy_signals = df[(df['rsi'] < 35) & (df['macd'] > df['macd_signal']) & (df['sma_20'] > df['sma_50'])]
            if not buy_signals.empty:
                first_buy = buy_signals.iloc[0]['Close']
                last_close = df.iloc[-1]['Close']
                buy_hold_return = (last_close - df.iloc[0]['Close']) / df.iloc[0]['Close'] * 100
                strategy_return = (last_close - first_buy) / first_buy * 100
                st.markdown(f"- Buy & Hold Return: `{buy_hold_return:.2f}%`")
                st.markdown(f"- Strategy Return (from first buy): `{strategy_return:.2f}%`")
            else:
                st.info("No buy signals for backtest in this period.")

            # Always show target and stoploss
            st.markdown(f"<div class='metric-card'><b>🎯 Target (if Buy):</b> ₹{round(latest['Close']*1.03,2)}<br><b>🛑 Stop-loss (if Buy):</b> ₹{round(latest['Close']*0.98,2)}</div>", unsafe_allow_html=True)

            # --- Recommendation Highlight ---
            if decision == "Buy":
                st.markdown(f'<div class="highlight-buy">✅ <span class="buy-signal">Buy Signal</span><br>📍 Target: ₹{target}<br>🛑 Stop-loss: ₹{stoploss}</div>', unsafe_allow_html=True)
            elif decision == "Sell":
                st.markdown(f'<div class="highlight-sell">❌ <span class="sell-signal">Sell Signal</span> (Overbought/High RSI)</div>', unsafe_allow_html=True)
            elif decision == "Previous Buy":
                st.info(f"ℹ️ **No current Buy, but previous Buy signal found.**\n\n📍 Previous Target: ₹{target}\n🛑 Previous Stop-loss: ₹{stoploss}")
            elif decision == "Previous Sell":
                st.info("ℹ️ **No current Sell, but previous Sell signal found.")
            elif decision == "Hold":
                st.info("ℹ️ **Hold/No Trade Signal at the moment.**")
            else:
                st.warning("No data available for this stock.")

            # Show last 5 historical buy signals
            st.markdown("<div class='section-title'>📜 Historical Buy Signals (last 5)</div>", unsafe_allow_html=True)
            buy_signals = df[
                (df['rsi'] < 35) &
                (df['macd'] > df['macd_signal']) &
                (df['sma_20'] > df['sma_50'])
            ]
            if not buy_signals.empty:
                st.dataframe(
                    buy_signals[['Date', 'Close', 'rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50']]
                    .tail(5)
                    .reset_index(drop=True)
                )
            else:
                st.info("No historical buy signals in this period.")

            # --- Multi-stock screener with buy suggestions and targets/stoploss ---
            st.markdown("<div class='section-title'>🧮 Multi-Stock Screener (Current Signals)</div>", unsafe_allow_html=True)
            results = []
            buy_suggestions = []
            leaderboard_data = []
            for stock in suggested_stocks:
                try:
                    dec, t, sl = cached_get_advice(stock + ".NS")
                    sdf = cached_fetch_stock_data(stock + ".NS")
                    last_close = sdf.iloc[-1]['Close']
                    # Calculate simple return (last 30 days)
                    if len(sdf) > 30:
                        ret = (last_close - sdf.iloc[-31]['Close']) / sdf.iloc[-31]['Close'] * 100
                    else:
                        ret = np.nan
                    results.append({"Stock": stock, "Signal": dec, "Target": t, "Stoploss": sl})
                    leaderboard_data.append({"Stock": stock, "Signal": dec, "Return(30d)%": ret})
                    if dec == "Buy":
                        buy_suggestions.append({"Stock": stock, "Target": t, "Stoploss": sl})
                except Exception:
                    results.append({"Stock": stock, "Signal": "Error", "Target": None, "Stoploss": None})
                    leaderboard_data.append({"Stock": stock, "Signal": "Error", "Return(30d)%": np.nan})
            st.dataframe(pd.DataFrame(results))

            # --- Leaderboard/Heatmap ---
            st.markdown("<div class='section-title'>🔥 Stock Leaderboard / Heatmap (30d Return)</div>", unsafe_allow_html=True)
            lb_df = pd.DataFrame(leaderboard_data)
            if not lb_df.empty:
                import plotly.express as px
                lb_df_sorted = lb_df.sort_values("Return(30d)%", ascending=False)
                fig2 = px.imshow([lb_df_sorted["Return(30d)%"].values],
                                 labels=dict(x="Stock", color="Return(30d)%"),
                                 x=lb_df_sorted["Stock"],
                                 y=["Return(30d)%"],
                                 color_continuous_scale="RdYlGn",
                                 aspect="auto")
                fig2.update_layout(height=200, title="30-Day Return Heatmap", font=dict(family="Segoe UI,Roboto,sans-serif", size=14, color="#283593"))
                st.markdown('<div class="heatmap-title">30-Day Return Heatmap</div>', unsafe_allow_html=True)
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(lb_df_sorted.reset_index(drop=True), use_container_width=True)
            else:
                st.info("Leaderboard data not available.")

            # --- Suggest stocks to buy with target and stoploss ---
            st.markdown("<div class='section-title'>💡 Stocks to Consider Buying (Current Buy Signals)</div>", unsafe_allow_html=True)
            if buy_suggestions:
                for s in buy_suggestions:
                    st.success(f"{s['Stock']}: Target ₹{s['Target']} | Stop-loss ₹{s['Stoploss']}")
            else:
                st.info("No stocks have a current Buy signal.")
        except Exception as e:
            st.error(f"❌ An error occurred:\n\n{e}")