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
        ("system", "You are helpful assistant"),
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
        output = response.get('output', str(response))
    except Exception as e:
        output = f"[Error generating response: {e}]"
        logger.error(f"Error during response generation: {e}")
    st.session_state.chat_history_agent.append(HumanMessage(content=input_text))
    st.session_state.chat_history_agent.append(AIMessage(content=output))
    # Removed chat rendering from here


# main function to encapsulate all the functionality
def agent_main():
    st.write(css, unsafe_allow_html=True)
    st.header('Conversational OpenAI Agent')
    st.write("Capabilities: ")
    st.write("1. Default gpt-4o-mini chatbot to answer questions (has memory of previous questions)")
    st.write("2. Has tools to perform web search, Wikipedia lookup, stock data retrieval, text analysis, and word cloud generation.")

    # Use dedicated chat history for agent
    if 'chat_history_agent' not in st.session_state:
        st.session_state.chat_history_agent = []
    if 'agent_executor' not in st.session_state:
        try:
            agent()
        except Exception as e:
            st.error(f"Error initializing agent: {e}")
            return

    # Move tool selection to sidebar
    with st.sidebar:
        st.write("----------------------------------------------------------------")
        st.subheader('🛠️ Select from available Tools')
        tool_options = [
            ("web_search", "Web Search"),
            ("wikipedia", "Wikipedia Lookup"),
            ("finance", "Stock Data"),
            ("text_analysis", "Text Analysis"),
            ("wordcloud", "Word Cloud")
        ]
        tool_labels = [label for _, label in tool_options]
        tool_keys = [key for key, _ in tool_options]
        if 'selected_tools' not in st.session_state:
            st.session_state.selected_tools = []
        selected_tool_labels = st.multiselect(
            "Select tools to use for the next query (optional):",
            tool_labels,
            default=[]
        )
        st.session_state.selected_tools = [tool_keys[tool_labels.index(lbl)] for lbl in selected_tool_labels if lbl in tool_labels]
        st.write("----------------------------------------------------------------")

    if 'input_text_box' not in st.session_state:
        st.session_state.input_text_box = ""

    def on_enter():
        if st.session_state.selected_tools:
            tool_requests = " ".join([f"[{tool}:{st.session_state.input_text_box}]" for tool in st.session_state.selected_tools])
            input_text_with_tools = f"{st.session_state.input_text_box} {tool_requests}"
        else:
            input_text_with_tools = st.session_state.input_text_box
        get_response(input_text_with_tools)
        st.session_state.input_text_box = ""
        st.session_state.selected_tools = []

    st.text_input(
        'Ask a question: ',
        value=st.session_state.input_text_box,
        key='input_text_box',
        on_change=on_enter
    )

    col1, col2 = st.columns(2)
    clear_chat = col2.button('Clear Chat')

    if clear_chat:
        st.session_state.chat_history_agent = []
        if 'memory' in st.session_state:
            st.session_state.memory.clear()
        st.session_state.input_text_box = ""
        st.rerun()

    # Render chat messages below header and input
    if st.session_state.chat_history_agent:
        for i, message in enumerate(st.session_state.chat_history_agent):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


if __name__ == '__main__':
    agent_main()