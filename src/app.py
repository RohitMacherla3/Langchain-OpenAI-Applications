from agent import agent_main
from QnA_app import q_and_a_main
from pdf_chat_app import pdf_main
import streamlit as st


def main():
    st.set_page_config(page_title = "Generative AI Applicaions")
    st.sidebar.subheader("Select an application to use")
    # Track last selected app
    if 'last_app' not in st.session_state:
        st.session_state.last_app = None
    application = st.sidebar.radio(" ", ['Q & A Chat Bot', 'Conversational Agent', 'Chat with your PDFs'])

    # Clear chat history and related state on app switch
    if st.session_state.last_app != application:
        if 'chat_history' in st.session_state:
            st.session_state.chat_history = []
        if 'memory' in st.session_state:
            st.session_state.memory = None
        if 'conversation' in st.session_state:
            st.session_state.conversation = None
        if application == 'Chat with your PDFs':
            st.session_state.app_switched_to_pdf_chat = True
        st.session_state.last_app = application

    if application == 'Q & A Chat Bot':
        q_and_a_main()
    elif application == 'Conversational Agent':
        agent_main()
    elif application == 'Chat with your PDFs':
        pdf_main()

if __name__ == "__main__":
    
    # # load api keys from local
    # import os
    # from dotenv import load_dotenv
    # load_dotenv()
    
    # OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    # SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY')
    # TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')
    
    # load api keys from streamlit secrets
    headers = {
        'OPENAI_API_KEY': st.secrets['OPENAI_API_KEY'],
        'SERPAPI_API_KEY': st.secrets['SERPAPI_API_KEY'],
        'TAVILY_API_KEY': st.secrets['TAVILY_API_KEY'],
        'content_type': 'application/json'
    }

    OPENAI_API_KEY = headers['OPENAI_API_KEY']
    SERPAPI_API_KEY = headers['SERPAPI_API_KEY']
    TAVILY_API_KEY = headers['TAVILY_API_KEY']
    
    main()