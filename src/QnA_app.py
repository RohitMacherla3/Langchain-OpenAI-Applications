def q_and_a_main():
    from langchain_openai import ChatOpenAI
    import streamlit as st

    #initializing all secret keys (local app)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 
    
    # #initializing all secret keys (streamlit deployment)
    # headers = {
    #     'authorization': st.secrets['OPENAI_API_KEY'],
    #     'content_type': 'application/json'
    #     }
    # OPENAI_API_KEY = headers['authorization']

    # function to load llm  and get output response
    def q_and_a_chatbot(input):
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model='gpt-3.5-turbo')
        response = llm.invoke(input)
        return response.content
    

    # streamlit application

    # st.set_page_config(page_title = "Q&A Bot")
    st.header('Q&A Bot Powered by OpenAI and LangChain')
    st.markdown('<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #708090; padding: 10px; text-align: center;">&copy; 2024 Rohit Macherla. All Rights Reserved.</div>',
                    unsafe_allow_html=True
                    )
    st.write("Capabilities: Uses ChatGPT-3.5 to generate answers. It has no memory and each question is handled individually")
    input = st.text_input('Ask a question: ', key ='input')

    submit = st.button('Generate Answer')

    if submit:
        with st.spinner("Generating..."):
            output_response = q_and_a_chatbot(input)
            st.write(output_response)
    
if __name__ == '__main__':
    q_and_a_main()

