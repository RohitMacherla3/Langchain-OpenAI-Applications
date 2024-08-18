from agent import agent_main
from QnA_app import q_and_a_main
from pdf_chat_app import pdf_main
import streamlit as st


def main():
    st.set_page_config(page_title = "Generative AI Applicaions")
    st.sidebar.subheader("Select an application to use")
    application = st.sidebar.radio(" ", ['Q & A Chat Bot', 'Conversational Agent', 'Chat with your PDFs'])
    
    if application == 'Q & A Chat Bot':
        q_and_a_main()
    elif application == 'Conversational Agent':
        agent_main()
    elif application == 'Chat with your PDFs':
        pdf_main()

if __name__ == "__main__":
    main()