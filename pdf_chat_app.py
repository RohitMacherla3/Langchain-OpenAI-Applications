import os
import streamlit as st
from dotenv import load_dotenv

# load data
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter

# vector database
import cassio
from langchain_community.vectorstores import Cassandra

# chatmodel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# read pdf into a single text and splict into chunks
def get_text_chunks(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text +=page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000, 
        chunk_overlap = 200,
        length_function = len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_memory=True, input_key='question')

    conversation_chain = RetrievalQA.from_chain_type(
        llm = ChatOpenAI(api_key=OPENAI_API_KEY),
        chain_type = "stuff",
        retriever = vectorstore.as_retriever(),
        chain_type_kwargs = {"memory": memory}
    )
    
    return conversation_chain


def get_output_response(question):
    response = st.session_state.conversation.run(question)
    st.write(response)
    # st.session_state.chat_history = response['chat_history']
        
    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write('Hello, here is your answer: \n', message.content)
    #     else:
    #         st.write(message.content)
    
def main():
    
    st.set_page_config(page_title = "Chat with PDFs")
    st.header('Chat with PDFs')
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    
    with st.sidebar:
        st.subheader('Your documents')
        input_pdfs = st.file_uploader('Upload your PDFs here and click Process', type='pdf', accept_multiple_files=True)
        
        if st.button('Process'):
            if len(input_pdfs) > 0:
            
                with st.spinner('Processing...'):
                    
                    # get text chunks from pdfs
                    text_chunks = get_text_chunks(input_pdfs)
                    
                    # loading FAISS vector store with dynamic table name
                    pdf_name =""
                    for pdf in input_pdfs:
                        pdf_name += pdf.name[:-4]
                        
                    table_name_dynamic = 'pdf_query_' + str(pdf_name)
        
                    embeds = OpenAIEmbeddings(model='text-embedding-3-small')
                        
                    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_API_ENDPOINT)
                    vectorstore = Cassandra(
                        embedding=embeds,
                        table_name=table_name_dynamic,
                        session = None,
                        keyspace = None
                    )
                        
                    vectorstore.add_texts(text_chunks)
                    
                    st.write("Files Processed!")
                    
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    # output_response = get_response_pdf(input_file, question, domain)
                    
            else:
                st.write("Please upload a file to process")
    
    # domain = st.text_input('Choose a domain for the document(s): ')
    question = st.text_input('Ask questions about your document(s): ')
    click = st.button('Generate')
    
    if question and click:
        with st.spinner('Generating...'):
            get_output_response(question)


if __name__ == "__main__":
    
    #initializing all secret keys
    load_dotenv()
    ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
    ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')    
    main()