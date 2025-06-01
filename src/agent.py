import streamlit as st
from tools import get_temperature, wiki_tool, tavily_tool, finance_tool, text_analysis_tool, generate_wordcloud_tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from htmlTemplates import css, bot_template, user_template
import logging


# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# define agent
def agent():
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    tools = [get_temperature, wiki_tool, tavily_tool, finance_tool, text_analysis_tool, generate_wordcloud_tool]

    # Validate tools and convert to OpenAI functions
    try:
        functions = [convert_to_openai_function(i) for i in tools]
        logger.debug("Tools successfully converted to OpenAI functions.")
    except Exception as e:
        logger.error(f"Error converting tools to OpenAI functions: {e}")
        raise

    # Bind model with functions
    try:
        agent_model = model.bind(functions=functions)
        logger.debug("Model successfully bound with functions.")
    except Exception as e:
        logger.error(f"Error binding model with functions: {e}")
        raise

    output_parser = OpenAIFunctionsAgentOutputParser()

    # Configure agent chain
    try:
        st.session_state.agent_chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | agent_model | output_parser
        logger.debug("Agent chain successfully configured.")
    except Exception as e:
        logger.error(f"Error configuring agent chain: {e}")
        raise

    # Initialize memory
    try:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        logger.debug("Memory successfully initialized.")
    except Exception as e:
        logger.error(f"Error initializing memory: {e}")
        raise

    # Initialize AgentExecutor
    try:
        st.session_state.agent_executor = AgentExecutor(
            agent=st.session_state.agent_chain, tools=tools, verbose=False, memory=st.session_state.memory
        )
        logger.debug("AgentExecutor successfully initialized.")
    except Exception as e:
        logger.error(f"Error initializing AgentExecutor: {e}")
        raise

    return st.session_state.agent_executor


# function to get the result from the agent and display to the user
def get_response(input_text):
    try:
        response = st.session_state.agent_executor.invoke({"input": input_text})
        # Use .get for safety in case 'output' key is missing
        output = response.get('output', str(response))
    except Exception as e:
        output = f"[Error generating response: {e}]"
        logger.error(f"Error during response generation: {e}")
    st.session_state.chat_history.append(HumanMessage(content=input_text))
    st.session_state.chat_history.append(AIMessage(content=output))
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 == 0:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# main function to encapsulate all the functionality
def agent_main():
    st.write(css, unsafe_allow_html=True)
    st.header('Conversational OpenAI Agent')
    st.markdown('<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #708090; padding: 10px; text-align: center;">&copy; 2024 Rohit Macherla. All Rights Reserved.</div>',
                    unsafe_allow_html=True
                    )
    st.write("Capabilities: ")
    st.write("1. Default ChatGPT-4o-mini chatbot to answer questions (has memory of previous questions)")
    st.write("2. Web search to get real-time data")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    # Only initialize agent_executor if not already present
    if 'agent_executor' not in st.session_state:
        try:
            agent()
        except Exception as e:
            st.error(f"Error initializing agent: {e}")
            return

    # Use session state for input text to allow clearing
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""

    # Create a form to handle Enter key submission
    with st.form(key='input_form', clear_on_submit=True):
        input_text = st.text_input('Ask a question: ', value=st.session_state.input_text, key='input_text_box')
        submit_button = st.form_submit_button('Generate Answer')

    col1, col2 = st.columns(2)
    clear_chat = col2.button('Clear Chat')

    if input_text and submit_button:
        with st.spinner('Generating answer...'):
            get_response(input_text)
            # Clear input after submission
            st.session_state.input_text = ""

    if clear_chat:
        st.session_state.chat_history = []
        # Also reset memory to clear conversation context
        if 'memory' in st.session_state:
            st.session_state.memory.clear()
        st.session_state.input_text = ""
        st.rerun()


if __name__ == '__main__':
    agent_main()