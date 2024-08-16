import os
from dotenv import load_dotenv
import streamlit as st
from tools import get_temperature, wiki_tool

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions

from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor


st.set_page_config(page_title = "Conversational Agent")
st.header('Conversational Agent')

def agent():
    model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    st.write(prompt)

    tools = [get_temperature, wiki_tool]
    functions = [convert_to_openai_function(i)for i in  tools]

    agent_model = model.bind(functions = functions)

    output_parser = OpenAIFunctionsAgentOutputParser()

    st.session_state.agent_chain = RunnablePassthrough.assign(
        agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | agent_model | output_parser

    st.session_state.memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

    st.session_state.agent_executor = AgentExecutor(agent=st.session_state.agent_chain, tools=tools, verbose=True, memory=st.session_state.memory)
    
    return st.session_state.agent_executor

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.agent_executor = agent()

input_text = st.text_input('Ask a question')
query = st.button('Generate answer')

if input_text:
    if query:
        with st.spinner('Generate answer...'):
            result = st.session_state.agent_executor.invoke({"input": input_text})
            st.session_state.chat_history.append(HumanMessage(content=input_text))
            st.session_state.chat_history.append(AIMessage(content=result['output']))
            st.write(result['output'])