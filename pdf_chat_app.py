import os
from dotenv import load_dotenv
import streamlit as st


# load data
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter

# vector database
from langchain_community.vectorstores import FAISS

# chatmodel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from htmlTemplates import css, bot_template, user_template

# read pdf into a single text and splict into chunks
def get_text_chunks(pdfs):
    
    text = ""
    if pdfs == 'docs/attention.pdf':
        pdf_reader = PdfReader(pdfs)
        for page in pdf_reader.pages:
            text +=page.extract_text()
    else:
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
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.1)
    retriever = vectorstore.as_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "{context}"
    ) 
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) 
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


def get_output_response(question):
    response = st.session_state.conversation.invoke({"input": question, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([response["answer"], HumanMessage(content=question)])
        
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i %2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
    
def main():
    
    st.set_page_config(page_title = "Chat with PDFs", page_icon = ":books:")
    st.header('Chat with PDFs :books:')
    st.markdown('<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #2b313e; padding: 10px; text-align: center;">&copy; 2024 Rohit Macherla. All Rights Reserved.</div>',
                unsafe_allow_html=True
                )
    st.write(css, unsafe_allow_html=True)
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    with st.sidebar:
        st.subheader('Your documents')
        model_type = st.sidebar.radio("Select File", ['Use pdf of Attention is all you need paper', 'Upload pdf'])
        if model_type == 'Use pdf of Attention is all you need paper':
            input_pdfs = 'docs/attention.pdf'
        elif model_type == 'Upload pdf':
            input_pdfs = st.file_uploader('Upload your PDFs here and click Process', type='pdf', accept_multiple_files=True)
        
        st.write('Click the Process button to process the document(s)')
        if st.button('Process'):
            if len(input_pdfs) > 0:
            
                with st.spinner('Processing...'):
                    
                    # get text chunks from pdfs
                    text_chunks = get_text_chunks(input_pdfs)
                    
                    # convert text to embeddings
                    embeds = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-small')
                    
                    # store the embeddings into a vectore store
                    vectorstore = FAISS.from_texts(text_chunks, embedding=embeds)
                    
                    st.write("Files Processed!")
                    
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
            else:
                st.write("Please upload a file to process")
    
    if model_type != 'Upload pdf':
        
        st.write('<div style="text-align:center; margin-right:200px;">How can I help you today?</div>', unsafe_allow_html=True)
        st.write('\n')
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)
        default_quesition_1 = col3.button('What is Attention?')
        default_quesition_2 = col4.button('What is Self-Attention?')
        default_quesition_3 = col5.button('What is the difference between them?')
        default_quesition_4 = col6.button('Explain Transformers?')
        
        if default_quesition_1 or default_quesition_2 or default_quesition_3 or default_quesition_4:
            if st.session_state.conversation is None:
                st.write('Click the process button on the side menu to process the file')
            else:
                with st.spinner('Generating...'):
                    if default_quesition_1:
                        get_output_response('What is Attention?')
                    elif default_quesition_2:
                        get_output_response('What is Self-Attention?')
                    elif default_quesition_3:
                        get_output_response('What is the difference between them?')
                    elif default_quesition_4:
                        get_output_response('Explain Transformers?')
            
    st.write('\n')
    st.write('\n')
    question = st.text_input('Ask questions about your document(s): ')
    
    col1, col2 = st.columns(2)
    st.session_state.click = col1.button('Generate')
    st.session_state.clear_chat = col2.button('Clear Chat')
    
    if question and st.session_state.click:
        if st.session_state.conversation is not None:
            with st.spinner('Generating...'):
                get_output_response(question)
        else:
            st.write('Please upload a file or use the deafult file and click process on the side menu')
    
    if st.session_state.clear_chat:
        st.session_state.chat_history = []


if __name__ == "__main__":
    
    ##initializing all secret keys (local app)
    # load_dotenv()
    # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  
    
    #initializing all secret keys (streamlit deployment)
    headers = {
    'authorization': st.secrets['OPENAI_API_KEY'],
    'content_type': 'application/json'
    }
    OPENAI_API_KEY = headers['authorization']
    main()