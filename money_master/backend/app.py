from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import requests
from groq import Groq 
from datetime import timedelta
import logging
import warnings
from nsetools import Nse
import yfinance as yf


app = Flask(__name__)
CORS(app)

file_path = os.path.join(os.path.dirname(__file__), 'NIFTY 50_minute.csv')
df = pd.read_csv(file_path)

GOOGLE_API_KEY = 'AIzaSyCHfXCqruaatQwfUp4sfYjCwJCOY7tnzRY'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

news_KEY = '442d358a3397407f8ff1d93295f60848'
url = 'https://newsapi.org/v2/top-headlines'

params = {
    'apiKey': news_KEY,
    'category': 'business'
}


@app.route('/art_create', methods=['POST'])
def art_create():
    try:
        response = requests.get(url, params=params)
        data = response.json()

        title_list = [title.get('title', 'No title available') for title in data['articles']]
        details = []

        predef = (
            "Read the title carefully",
            "Give me a summary of how this will influence the price of various stocks",
            "Give ONLY that, in a paragraph format",
            "Specifically name the stocks"
        )

        for title in title_list[:5]:
            x = model.generate_content(f"{predef} {title}")
            details.append(x.text)

        return jsonify({"messagehead": title_list[:5], "messagebody": details})
    
    except Exception as e:
        
        print("exception raised", e)
        temp = ['a', 'b', 'c']
        return jsonify({"messagehead": title_list, "messagebody": temp})

@app.route('/generate-text', methods=['POST'])
def generate_text():
    prompt = request.json.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            generated_text = response.text
        elif 'text' in response:
            generated_text = response['text']
        else:
            generated_text = "No text generated" 
        return jsonify({'text': generated_text})

    except Exception as e:
        print(f"Error generating text: {e}")
        return jsonify({'error': 'Error generating text'}), 500

GORQ_API_KEY = "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"
# Optionally, you can also set it via environment variable:
# os.environ["GROQ_API_KEY"] = GORQ_API_KEY

@app.route('/art_create_gorq', methods=['POST'])
def art_create_gorq():
    try:
        # Fetch global news articles as before
        response = requests.get(url, params=params)
        data = response.json()
        title_list = [article.get('title', 'No title available') for article in data.get('articles', [])]
        details = []
        
        # Predefined prompt for summarization
        predef = (
            "Read the title carefully. ",
            "Give me a summary of how this will influence the price of various stocks. ",
            "Give ONLY that, in a paragraph format. ",
            "Specifically name the stocks."
        )
        prompt_base = " ".join(predef)
        
        # Create a Groq client using the gorq API key
        client = Groq(api_key=GORQ_API_KEY)
        
        for title in title_list[:5]:
            prompt = f"{prompt_base} {title}"
            # Use the Groq client to generate a summary.
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            # Extract the generated text
            summary = resp.choices[0].message.content
            details.append(summary)
            
        return jsonify({"messagehead": title_list[:5], "messagebody": details})
    
    except Exception as e:
        print("Exception in art_create_gorq:", e)
        fallback_titles = title_list if 'title_list' in locals() else ['No data']
        return jsonify({"messagehead": fallback_titles, "messagebody": ['Error generating details']}), 500

@app.route('/generate-text_gorq', methods=['POST'])
def generate_text_gorq():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        # Create a Groq client using the gorq API key
        client = Groq(api_key=GORQ_API_KEY)
        
        # Generate completion using the provided prompt
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        generated_text = resp.choices[0].message.content
        return jsonify({'text': generated_text})
    
    except Exception as e:
        print(f"Error in generate-text_gorq: {e}")
        return jsonify({'error': 'Error generating text'}), 500


logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)


# Initialize NSE and get list of NSE tickers
nse = Nse()
nse_stock_list = nse.get_stock_codes()  # Returns a list of tickers
# Clean the list: remove empty entries and the header "SYMBOL"
nse_stock_set = set([s.strip().upper() for s in nse_stock_list if s.strip() and s.strip() != "SYMBOL"])

@app.route('/get-tickers', methods=['GET'])
def get_tickers():
    try:
        tickers = list(nse_stock_set)
        return jsonify({"tickers": tickers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

lstm_model = tf.keras.models.load_model("./nifty50_model_with_tech_indicators.keras", compile=False)
print("LSTM Model loaded from nifty50_model_with_tech_indicators.keras")

# Function to compute 14-day RSI.
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to prepare daily data with technical indicators for LSTM prediction.
def get_latest_window_features(ticker, window_size=30):
    # Download daily historical data (at least 100 days).
    hist = yf.Ticker(ticker).history(period="100d")
    if hist.empty:
        return None
    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        hist[col] = pd.to_numeric(hist[col], errors='coerce')
    hist = hist[hist['Volume'] > 0].copy()
    hist.reset_index(inplace=True)
    # Compute technical indicators.
    hist['RSI14'] = compute_RSI(hist['Close'], period=14)
    hist['SMA30'] = hist['Close'].rolling(window=30).mean()
    hist['Deviation'] = hist['Close'] - hist['SMA30']
    hist.dropna(inplace=True)
    if len(hist) < window_size:
        return None
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI14', 'SMA30', 'Deviation']
    feature_data = hist[features].values
    scaler_local = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_local.fit_transform(feature_data)
    window_data = scaled_features[-window_size:]
    last_close = hist['Close'].iloc[-1]
    window_data = np.reshape(window_data, (1, window_size, len(features)))
    return window_data, scaler_local, last_close

def inverse_transform_close(scaled_value, scaler_local):
    # Inverse transform the scaled 'Close' value (assumed at index 3).
    dummy = np.zeros((1, 8))
    dummy[0, 3] = scaled_value[0]
    inv = scaler_local.inverse_transform(dummy)
    return inv[0, 3]

@app.route('/generate-chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        input_ticker = data.get("ticker", "").strip().upper()
        if not input_ticker:
            return jsonify({"error": "Ticker is required"}), 400

        # Validate ticker using the dynamic NSE stock set and append .NS if necessary.
        if not input_ticker.endswith(".NS"):
            if input_ticker in nse_stock_set:
                ticker = input_ticker + ".NS"
            else:
                return jsonify({"error": f"Ticker {input_ticker} not found in NSE"}), 404
        else:
            base_ticker = input_ticker[:-3]
            if base_ticker in nse_stock_set:
                ticker = input_ticker
            else:
                return jsonify({"error": f"Ticker {input_ticker} not found in NSE"}), 404

        # Download minute-level data for Prophet prediction.
        try:
            df = yf.download(ticker, period="8d", interval="1m")
        except Exception as download_error:
            return jsonify({"error": f"Failed to download data for ticker {ticker}: {download_error}"}), 404

        if df.empty:
            return jsonify({"error": f"No data found for ticker {ticker}. It might be temporarily unavailable."}), 404

        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Prepare data for Prophet.
        df_prophet = df[['Datetime', 'Close']].rename(columns={'Datetime': 'ds', 'Close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')

        # Use the latter portion of the historical data (one-third index here).
        half_index = len(df_prophet) // 3
        historical_data_second_half = df_prophet.iloc[half_index:]

        # Fit the Prophet model.
        model_prophet = Prophet(daily_seasonality=1, yearly_seasonality=8, weekly_seasonality=0)
        model_prophet.fit(df_prophet)

        # Forecast the next 2160 minutes (1.5 days).
        future = model_prophet.make_future_dataframe(periods=2160, freq='min')
        forecast = model_prophet.predict(future)

        # Filter predictions beyond the last training timestamp.
        last_actual_date = df_prophet['ds'].iloc[-1]
        predicted_data = forecast[forecast['ds'] > last_actual_date][['ds', 'yhat']]

        actual_data = historical_data_second_half[['ds', 'y']].rename(
            columns={'ds': 'date', 'y': 'close'}
        ).to_dict(orient='records')
        forecast_data = predicted_data.rename(
            columns={'ds': 'date', 'yhat': 'predicted'}
        ).to_dict(orient='records')

        # --- LSTM Next Day Prediction (using daily data) ---
        lstm_result = get_latest_window_features(ticker, window_size=30)
        if lstm_result is not None:
            window_data, scaler_local, last_close = lstm_result
            pred_scaled = lstm_model.predict(window_data)
            predicted_close = inverse_transform_close(pred_scaled[0], scaler_local)
            lstm_pct_change = ((predicted_close - last_close) / last_close) * 100
            lstm_prediction = {
                "last_close": last_close,
                "predicted_close": predicted_close,
                "pct_change": lstm_pct_change
            }
        else:
            lstm_prediction = None

        return jsonify({
            'actual_data': actual_data,
            'forecast_data': forecast_data,
            'lstm_prediction': lstm_prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get-news-summary', methods=['POST'])
def get_news_summary():
    ticker_symbol = request.json.get('ticker')
    if not ticker_symbol:
        return jsonify({'error': 'Ticker is required'}), 400

    # Automatically append ".NS" if not present
    if not ticker_symbol.strip().upper().endswith(".NS"):
        ticker_symbol = ticker_symbol.strip().upper() + ".NS"
    
    try:
        # Create a Ticker object from yfinance.
        ticker = yf.Ticker(ticker_symbol)
        news_items = ticker.news
        if not news_items:
            return jsonify({'error': 'No news found for this ticker'}), 404
        
        # Flatten news items (each item has an "id" and "content" dictionary)
        flat_news = []
        for item in news_items:
            flat_item = {}
            flat_item["id"] = item.get("id")
            content = item.get("content", {})
            flat_item.update(content)
            flat_news.append(flat_item)
        
        df_news = pd.DataFrame(flat_news)
        
        # Combine all articles by iterating through each news item.
        articles_texts = []
        for idx, article in df_news.iterrows():
            title = article.get('title', '')
            description = article.get('description', '')
            summary = article.get('summary', '')
            combined = f"{title}\n{description}\n{summary}".strip()
            if combined:
                articles_texts.append(combined)
        # Join all articles with a separator.
        all_articles_text = "\n\n".join(articles_texts)
        
        # Create a new prompt that instructs the LLM to derive the stock name,
        # provide an extensive summary (at least 150 words) of all articles,
        # and then indicate overall sentiment (positive, negative, or neutral) exactly once.
        prompt = (
            f"Based on the ticker symbol {ticker_symbol}, use the stock name instead of ticker. "
            "Now, provide an extensive summary of the following news articles in at least 150 words. "
            "Include all key details from each article. Finally, at the end of your summary, indicate the overall sentiment "
            "of the articles as either 'positive', 'negative', or 'neutral' (only include this sentiment indicator once). "
            f"News articles:\n\n{all_articles_text}"
        )
        
        # Call your Groq API client to generate the summary.
        client = Groq(api_key=GORQ_API_KEY)
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        generated_text = resp.choices[0].message.content
        
        # Simple sentiment analysis based on generated text:
        lower_text = generated_text.lower()
        if "positive" in lower_text:
            sentiment = "green"
        elif "negative" in lower_text:
            sentiment = "red"
        else:
            sentiment = "neutral"
        
        return jsonify({'summary': generated_text, 'sentiment': sentiment})
    
    except Exception as e:
        print(f"Error in get_news_summary: {e}")
        return jsonify({'error': 'Error generating news summary'}), 500


if __name__ == '__main__':
    app.run(debug=True)