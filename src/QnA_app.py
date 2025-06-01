from langchain_openai import ChatOpenAI
import streamlit as st

# function to load llm  and get output response
def q_and_a_chatbot(input):
    llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')
    response = llm.invoke(input)
    return response.content

def q_and_a_main():
    st.header('Q&A Bot Powered by OpenAI and LangChain')
    st.markdown('<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #708090; padding: 10px; text-align: center;">&copy; 2024 Rohit Macherla. All Rights Reserved.</div>',
                    unsafe_allow_html=True
                    )
    st.write("Capabilities: Uses gpt-4o-mini to generate answers. It has no memory and each question is handled individually")

    if 'output_response' not in st.session_state:
        st.session_state['output_response'] = ''
    if 'input' not in st.session_state:
        st.session_state['input'] = ''

    def on_enter():
        st.session_state['output_response'] = q_and_a_chatbot(st.session_state['input'])
        st.session_state['input'] = ''

    st.text_input('Ask a question: ', key='input', on_change=on_enter)

    if st.session_state['output_response']:
        st.write(st.session_state['output_response'])
    
if __name__ == '__main__':
    q_and_a_main()

