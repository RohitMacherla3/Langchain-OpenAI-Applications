from langchain_openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import os

#load OPEN AI KEY from .env
load_dotenv()

# function to load llm  and get output response
def get_response(input):
    llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0.2)
    response = llm.invoke(input)
    return response

# streamlit application

st.set_page_config(page_title = "Q&A Bot")
st.header('Q&A Bot Powered by OpenAI and LangChain')
input = st.text_input('Input: ', key ='input')

submit = st.button('Ask the question')

if submit:
    st.write('Generating...')
    output_response = get_response(input)
    st.write(output_response)

