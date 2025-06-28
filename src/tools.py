import streamlit as st
from langchain.agents import tool
from pydantic import BaseModel, Field
import requests, datetime
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from collections import Counter
import re
from wordcloud import WordCloud
from alpha_vantage.timeseries import TimeSeries


# # Note: The 'headers' dictionary is defined for API keys but is not used in requests below.
# headers = {
#     'OPENAI_API_KEY': st.secrets['OPENAI_API_KEY'],
#     'TAVILY_API_KEY': st.secrets['TAVILY_API_KEY'],
#     'content_type': 'application/json'
# }

# OPENAI_API_KEY = headers['OPENAI_API_KEY']
# TAVILY_API_KEY = headers['TAVILY_API_KEY']

# ====== TOOL 1: Wikipedia ======
api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# ====== TOOL 2: Tavily Search ======
tavily_tool = TavilySearchResults()

# ====== TOOL 3: Weather ======
class OpenMeteoInput(BaseModel):
    latitude: str = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: str = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_temperature(latitude, longitude) -> str:
    """Get the current weather for the given latitude and longitude. Returns a string."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "forecast_days": 1
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception("Failed")
    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    return f'The current temperature is {current_temperature}°C'

# ====== TOOL 4: FINANCIAL DATA ======
class FinanceInput(BaseModel):
    symbol: str = Field(..., description="Stock or crypto symbol (e.g., AAPL, BTC-USD)")
    period: str = Field("1mo", description="Period for historical data (e.g., 1mo, 1y)")

@tool(args_schema=FinanceInput)
def finance_tool(symbol, period="1mo"):
    """Get financial data for stocks/crypto"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        info = ticker.info
        if hist.empty:
            return {"error": f"No financial data found for symbol: {symbol}. This may be due to an invalid symbol, rate limiting, or Yahoo Finance blocking automated requests. Please try again later or check the symbol on Yahoo Finance directly."}
        current_price = hist['Close'].iloc[-1]
        change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
        change_percent = (change / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
        # Create a simple price chart
        fig, ax = plt.subplots(figsize=(10, 6))
        hist['Close'].plot(ax=ax, title=f"{symbol} - {period} Price Chart")
        ax.set_ylabel('Price')
        ax.grid(True)
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 'N/A',
            'chart': chart_base64,
            'company_name': info.get('longName', symbol)
        }
    except Exception as e:
        return {"error": f"Financial data retrieval failed for symbol '{symbol}': {str(e)}. This may be due to rate limiting or Yahoo Finance blocking automated requests. Please try again later or check the symbol on Yahoo Finance directly."}

# ====== TOOL 5: TEXT ANALYSIS ======
class TextAnalysisInput(BaseModel):
    text: str = Field(..., description="Text to analyze")

@tool(args_schema=TextAnalysisInput)
def text_analysis_tool(text):
    """Analyze text for readability, sentiment, and statistics"""
    try:
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        flesch_score = flesch_reading_ease(text)
        fk_grade = flesch_kincaid_grade(text)
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        most_common = word_freq.most_common(10)
        if flesch_score >= 90:
            reading_level = "Very Easy"
        elif flesch_score >= 80:
            reading_level = "Easy"
        elif flesch_score >= 70:
            reading_level = "Fairly Easy"
        elif flesch_score >= 60:
            reading_level = "Standard"
        elif flesch_score >= 50:
            reading_level = "Fairly Difficult"
        elif flesch_score >= 30:
            reading_level = "Difficult"
        else:
            reading_level = "Very Difficult"
        return {
            'word_count': word_count,
            'character_count': char_count,
            'sentence_count': sentence_count,
            'flesch_reading_ease': round(flesch_score, 2),
            'flesch_kincaid_grade': round(fk_grade, 2),
            'reading_level': reading_level,
            'most_common_words': most_common,
            'avg_words_per_sentence': round(word_count / sentence_count, 2) if sentence_count > 0 else 0
        }
    except Exception as e:
        return {"error": f"Text analysis failed: {str(e)}"}

# ====== TOOL 6: WORD CLOUD GENERATOR ======
class WordCloudInput(BaseModel):
    text: str = Field(..., description="Text to generate word cloud from")

@tool(args_schema=WordCloudInput)
def generate_wordcloud_tool(text):
    """Generate a word cloud from text"""
    try:
        wordcloud_obj = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud_obj, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud', fontsize=16, pad=20)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        wordcloud_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return {
            'wordcloud': wordcloud_base64,
            'status': 'success'
        }
    except Exception as e:
        return {"error": f"Word cloud generation failed: {str(e)}"}

# ====== TOOL 7: ALPHA VANTAGE FINANCE DATA ======
class AlphaVantageFinanceInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, MSFT)")
    interval: str = Field("1min", description="Time interval between data points (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)")
    outputsize: str = Field("compact", description="compact (latest 100 points) or full (full-length data)")

@tool(args_schema=AlphaVantageFinanceInput)
def alpha_vantage_finance_tool(symbol, interval="daily", outputsize="compact"):
    """Get stock price data using Alpha Vantage API (free, requires API key)."""
    try:
        api_key = st.secrets.get('ALPHA_VANTAGE_API_KEY', None)
        if not api_key:
            return {"error": "Alpha Vantage API key not found in Streamlit secrets. Please add 'ALPHA_VANTAGE_API_KEY' to your secrets."}
        ts = TimeSeries(key=api_key, output_format='pandas')
        if interval in ["daily", "weekly", "monthly"]:
            func_map = {
                "daily": ts.get_daily,
                "weekly": ts.get_weekly,
                "monthly": ts.get_monthly
            }
            data, meta = func_map[interval](symbol=symbol, outputsize=outputsize)
        else:
            data, meta = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
        if data.empty:
            return {"error": f"No data found for symbol: {symbol}."}
        # Plot closing price
        fig, ax = plt.subplots(figsize=(10, 6))
        data['4. close'].plot(ax=ax, title=f"{symbol} - {interval.capitalize()} Closing Price")
        ax.set_ylabel('Price (USD)')
        ax.grid(True)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        latest_close = data['4. close'].iloc[-1]
        return {
            'symbol': symbol,
            'latest_close': round(latest_close, 2),
            'interval': interval,
            'chart': chart_base64,
            'meta': meta
        }
    except Exception as e:
        return {"error": f"Alpha Vantage data retrieval failed for symbol '{symbol}': {str(e)}"}
