from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import tool, load_tools
from pydantic import BaseModel, Field
import requests, datetime

# load keys from local
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')


import streamlit as st
# headers = {
#     'OPENAI_API_KEY': st.secrets['OPENAI_API_KEY'],
#     'SERPAPI_API_KEY': st.secrets['SERPAPI_API_KEY'],
#     'TAVILY_API_KEY': st.secrets['TAVILY_API_KEY'],
#     'content_type': 'application/json'
# }

# OPENAI_API_KEY = headers['OPENAI_API_KEY']
# SERPAPI_API_KEY = headers['SERPAPI_API_KEY']
# TAVILY_API_KEY = headers['TAVILY_API_KEY']

# Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Weather tool
class OpenMeteoInput(BaseModel):
    latitude: str = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: str = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_temperature(latitude, longitude) -> dict:
    """Get the current weather for the given latitude and longitude"""
    
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

# tavily search
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults()


