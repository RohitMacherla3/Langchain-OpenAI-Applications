import streamlit as st
import os
import requests
import re
import base64
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from htmlTemplates import css, bot_template, user_template
import nltk
from tools import (
    tavily_tool, 
    wiki_tool, 
    finance_tool, 
    text_analysis_tool, 
    generate_wordcloud_tool
)


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configuration for development/production
SHOW_MODEL_SELECTION = os.getenv('SHOW_MODEL_SELECTION', 'false').lower() == 'true'

# Default Ollama URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Default PDF path
DEFAULT_PDF_PATH = "docs/attention.pdf"

def setup_default_pdf_directory():
    """Create docs directory if it doesn't exist and provide instructions"""
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        st.info(f"Created '{docs_dir}' directory. Please add 'attention.pdf' file to use the default option.")
    
    if not os.path.exists(DEFAULT_PDF_PATH):
        return False
    return True

def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        else:
            return []
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return []

def get_llm_and_embeddings(model_provider, model_name=None):
    """Get LLM and embeddings based on the selected provider"""
    
    if model_provider == "OpenAI":
        llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    
    elif model_provider == "Ollama":
        if not model_name:
            model_name = "llama3.2:latest"  # Default fallback
        
        llm = Ollama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.5
        )
        embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=OLLAMA_BASE_URL
        )
    
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    
    return llm, embeddings

def use_tool(tool_name, query, context=None):
    """Dispatch to appropriate tool based on tool_name"""
    
    if tool_name == "web_search":
        return tavily_tool(query)
    elif tool_name == "wikipedia":
        return wiki_tool(query)
    elif tool_name == "finance":
        # Extract stock symbol from query
        symbol_match = re.search(r'\b[A-Z]{1,5}\b', query.upper())
        symbol = symbol_match.group() if symbol_match else query.upper()
        return finance_tool(symbol)
    elif tool_name == "text_analysis":
        text_to_analyze = context if context else query
        return text_analysis_tool(text_to_analyze)
    elif tool_name == "wordcloud":
        text_for_cloud = context if context else query
        return generate_wordcloud_tool(text_for_cloud)
    else:
        return {"error": f"Unknown tool: {tool_name}"}

def get_text_chunks(pdfs):
    text = ""
    
    # Handle the case where pdfs is a string (default PDF path)
    if isinstance(pdfs, str):
        # Check if the default PDF file exists
        if not os.path.exists(pdfs):
            st.error(f"Default PDF file not found: {pdfs}")
            st.info("Please create a 'docs' folder and add 'attention.pdf' file, or upload your own PDF.")
            return None
            
        try:
            pdf_reader = PdfReader(pdfs)
            for page in pdf_reader.pages:
                text += page.extract_text()
            st.write(f"Processed {os.path.basename(pdfs)}")
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return None
    else:
        # Handle uploaded files
        if not pdfs:
            st.error("No files uploaded.")
            return None
            
        file_size = 0  # to check the file size constraint of <10MB
        file_names = []  # to only process unique files
        for pdf in pdfs:
            if pdf.name not in file_names:
                file_size += pdf.size
                if file_size <= 10 * 1024 * 1024:
                    file_names.append(pdf.name)
                    try:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        st.write(f"Processed {pdf.name}")
                    except Exception as e:
                        st.error(f"Error reading {pdf.name}: {e}")
                        continue
                else:
                    st.error("Overall size of the files exceeds the limit of 10 MB. Re-upload the files.")
                    return None
    
    # Check if we have any text to process
    if not text.strip():
        st.error("No text could be extracted from the PDF(s).")
        return None
        
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
        
    chunks = text_splitter.split_text(text)
    return chunks

# Enhanced conversation chain with tool integration
def get_conversation_chain(vectorstore, model_provider, model_name=None):
    llm, _ = get_llm_and_embeddings(model_provider, model_name)
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
        "You are an intelligent assistant with access to multiple tools. Use "
        "the following pieces of retrieved context to answer the question. "
        "If you need additional information beyond the context, you can suggest "
        "using these available tools:\n"
        "- web_search: Search the internet for current information\n"
        "- wikipedia: Search Wikipedia for encyclopedic information\n"
        "- finance: Get stock/financial data (use format: finance:SYMBOL)\n"
        "- text_analysis: Analyze text for readability and statistics\n"
        "- wordcloud: Generate a word cloud visualization\n\n"
        "When suggesting a tool, use the format: [TOOL:query] (e.g., [web_search:latest AI research])\n\n"
        "Context: {context}"
    ) 
        
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) 
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
    return rag_chain

def get_output_response(question):
    # Check if the question contains tool requests
    tool_pattern = r'\[(\w+):([^\]]+)\]'
    tool_matches = re.findall(tool_pattern, question)
    
    tool_results = []
    if tool_matches:
        for tool_name, query in tool_matches:
            # Get PDF context if available for text analysis tools
            context = None
            if tool_name in ['text_analysis', 'wordcloud'] and 'pdf_text' in st.session_state:
                context = st.session_state.pdf_text
            
            result = use_tool(tool_name.lower(), query, context)
            tool_results.append({
                'tool': tool_name,
                'query': query,
                'result': result
            })
    
    # Get response from conversational chain
    response = st.session_state.conversation.invoke({
        "input": question, 
        "chat_history": st.session_state.chat_history
    })
    
    # Add tool results to the response if any (but do not display them)
    if tool_results:
        response['tool_results'] = tool_results
    
    st.session_state.chat_history.extend([
        HumanMessage(content=question), 
        AIMessage(content=response.get('answer', str(response)))
    ])

    # Display the conversation in chronological order (oldest to newest)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Display tool results for the most recent AI message if any
    if tool_results:
        # Only display after the last bot message
        for tool_result in tool_results:
            st.write(f"**🔧 Tool Used: {tool_result['tool'].title()}**")
            if 'error' in tool_result['result']:
                st.error(f"Tool Error: {tool_result['result']['error']}")
            else:
                if tool_result['tool'] == 'web_search':
                    for result in tool_result['result']:
                        if 'url' in result:
                            st.write(f"**{result['title']}**")
                            st.write(f"URL: {result['url']}")
                            st.write(f"Snippet: {result['snippet'][:200]}...")
                            st.write("---")
                elif tool_result['tool'] == 'wikipedia':
                    for result in tool_result['result']:
                        st.write(f"**{result['title']}**")
                        st.write(result['summary'])
                        st.write(f"[Read more]({result['url']})")
                        st.write("---")
                elif tool_result['tool'] == 'finance':
                    result = tool_result['result']
                    st.write(f"**{result['company_name']} ({result['symbol']})**")
                    st.write(f"Current Price: ${result['current_price']}")
                    st.write(f"Change: ${result['change']} ({result['change_percent']:.2f}%)")
                    if 'chart' in result:
                        st.image(base64.b64decode(result['chart']))
                elif tool_result['tool'] == 'text_analysis':
                    result = tool_result['result']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Text Statistics:**")
                        st.write(f"- Words: {result['word_count']}")
                        st.write(f"- Characters: {result['character_count']}")
                        st.write(f"- Sentences: {result['sentence_count']}")
                    with col2:
                        st.write("**Readability:**")
                        st.write(f"- Reading Level: {result['reading_level']}")
                        st.write(f"- Flesch Score: {result['flesch_reading_ease']}")
                        st.write(f"- Grade Level: {result['flesch_kincaid_grade']}")
                elif tool_result['tool'] == 'wordcloud':
                    if 'wordcloud' in tool_result['result']:
                        st.image(base64.b64decode(tool_result['result']['wordcloud']))
            
            st.write("---")
    else:
        st.session_state.chat_history.pop()  # Remove the last AI message if no tool results

def pdf_main():
    # Removed footer and copyright
    st.write(css, unsafe_allow_html=True)
        
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None 
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'selected_model_type' not in st.session_state:
        st.session_state.selected_model_type = None
    
    if 'model_provider' not in st.session_state:
        st.session_state.model_provider = "OpenAI"
    
    if 'selected_model_name' not in st.session_state:
        st.session_state.selected_model_name = None
    
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
            
    with st.sidebar:
        st.write("----------------------------------------------------------------")
        
        # Available Tools Information
        st.subheader('🛠️ Select from available Tools')
        
        # Multi-select for tool selection
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
        # Map labels back to tool keys
        st.session_state.selected_tools = [tool_keys[tool_labels.index(lbl)] for lbl in selected_tool_labels if lbl in tool_labels]
        
        st.write("----------------------------------------------------------------")
        
        # Model Selection (only show if SHOW_MODEL_SELECTION is True)
        if SHOW_MODEL_SELECTION:
            st.subheader('Model Configuration')
            model_provider = st.selectbox(
                "Select Model Provider",
                ["OpenAI", "Ollama"],
                index=0 if st.session_state.model_provider == "OpenAI" else 1
            )
            
            selected_model_name = None
            if model_provider == "Ollama":
                ollama_models = get_available_ollama_models()
                if ollama_models:
                    selected_model_name = st.selectbox(
                        "Select Ollama Model",
                        ollama_models,
                        index=0
                    )
                else:
                    st.error("No Ollama models found. Make sure Ollama is running and has models installed.")
                    st.info("To install models, run: `ollama pull llama2` in your terminal")
            
            # Reset conversation if model provider or model changes
            if (model_provider != st.session_state.model_provider or 
                selected_model_name != st.session_state.selected_model_name):
                st.session_state.conversation = None
                st.session_state.model_provider = model_provider
                st.session_state.selected_model_name = selected_model_name
            
            st.write("----------------------------------------------------------------")
        
        st.subheader('Your documents')
        
        # Check if default PDF exists
        default_pdf_available = setup_default_pdf_directory()
        
        if default_pdf_available:
            model_type = st.radio("Select File", ['Use pdf of Attention is all you need paper', 'Upload pdf'])
        else:
            st.warning("Default PDF not found. Please upload your own PDF files.")
            model_type = st.radio("Select File", ['Upload pdf'], index=0)
            
        if model_type == 'Use pdf of Attention is all you need paper' and default_pdf_available:
            input_pdfs = DEFAULT_PDF_PATH
        elif model_type == 'Upload pdf':
            input_pdfs = st.file_uploader(
                'Upload your PDFs (upto overall size of 10MB)', 
                type='pdf',
                accept_multiple_files=True
            )
            
        # Reset session state when model type changes
        if model_type != st.session_state.selected_model_type:
            st.session_state.conversation = None
            st.session_state.selected_model_type = model_type
            
        st.write('Click the Process button to process the document(s)')
        if st.button('Process'):
            # Validate inputs based on model type
            process_valid = False
            if model_type == 'Use pdf of Attention is all you need paper':
                process_valid = default_pdf_available and input_pdfs
            elif model_type == 'Upload pdf':
                process_valid = input_pdfs and len(input_pdfs) > 0
            
            if process_valid:
                with st.spinner('Processing...'):
                    # get text chunks from pdfs
                    text_chunks = get_text_chunks(input_pdfs)
                        
                    if text_chunks is not None:
                        # Store full text for text analysis tools
                        st.session_state.pdf_text = " ".join(text_chunks)
                        
                        # Get embeddings based on selected provider
                        try:
                            _, embeddings = get_llm_and_embeddings(
                                st.session_state.model_provider, 
                                st.session_state.selected_model_name
                            )
                            
                            # store the embeddings into a vector store
                            vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
                            st.session_state.conversation = get_conversation_chain(
                                vectorstore, 
                                st.session_state.model_provider, 
                                st.session_state.selected_model_name
                            )
                            st.success(f"Documents processed successfully using {st.session_state.model_provider}!")
                            
                        except Exception as e:
                            st.error(f"Error processing documents: {e}")
                            
            else:
                if model_type == 'Use pdf of Attention is all you need paper' and not default_pdf_available:
                    st.warning("Default PDF file not found. Please add 'attention.pdf' to the 'docs' folder or upload your own PDF.")
                else:
                    st.warning("Please upload a file to process")
    
    # Display current model info (only in dev mode)
    if SHOW_MODEL_SELECTION:
        st.info(f"Current Model: {st.session_state.model_provider}" + 
                (f" - {st.session_state.selected_model_name}" if st.session_state.selected_model_name else ""))

    # Center the main heading and ensure it appears only once
    st.markdown('<h1 style="text-align:center; margin-top:0.5em; margin-bottom:0.5em; font-size:2.5rem; font-weight:800; color:#fff;">PDF Chat with Tools 🛠️</h1>', unsafe_allow_html=True)

    # Default questions for the default PDF
    default_questions = [
        "What is Attention?",
        "What is Self-Attention?",
        "What is the difference between them?",
        "Explain Transformers to a 5 year old"
    ]
    if model_type == 'Use pdf of Attention is all you need paper' and default_pdf_available:
        st.markdown('<div style="text-align:center; margin-bottom:18px; font-size:1.2rem; color:#e0e0e0;">How can I help you?</div>', unsafe_allow_html=True)
        st.write("")  # vertical space
        # First row
        row1_col1, row1_col2, _, _ = st.columns([1,1,1e-9,1e-9])
        btn_clicked = None
        with row1_col1:
            if st.button(default_questions[0], key="btn_0", use_container_width=True):
                btn_clicked = 0
        with row1_col2:
            if st.button(default_questions[1], key="btn_1", use_container_width=True):
                btn_clicked = 1
        # Second row
        row2_col1, row2_col2, _, _ = st.columns([1,1,1e-9,1e-9])
        with row2_col1:
            if st.button(default_questions[2], key="btn_2", use_container_width=True):
                btn_clicked = 2
        with row2_col2:
            if st.button(default_questions[3], key="btn_3", use_container_width=True):
                btn_clicked = 3
        def handle_default_question(idx):
            if st.session_state.conversation is None:
                st.warning("Please click the Process button on the sidebar to process the document before asking questions.")
            else:
                question = default_questions[idx]
                st.session_state.question_input = ""
                get_output_response(question)
        if btn_clicked is not None:
            handle_default_question(btn_clicked)
        st.write("")  # vertical space

    # Initialize question input state
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""

    # Create a form for the question input to handle Enter key
    with st.form(key='question_form', clear_on_submit=True):
        question = st.text_input(
            'Enter questions about your document(s) and press Enter or click Generate:', 
            value=st.session_state.question_input,
            key='question_text_input',
            help="Use [tool:query] format to use tools. Example: What is machine learning? [web_search:latest ML trends]"
        )
        col1, col2 = st.columns(2)
        generate_clicked = col1.form_submit_button('Generate Answer')
        clear_clicked = col2.form_submit_button('Clear Chat')

    # Handle question submission (both Enter key and button click)
    if question and generate_clicked:
        if st.session_state.conversation is not None:
            with st.spinner('Generating...'):
                if st.session_state.selected_tools:
                    tool_requests = " ".join([f"[{tool}:{question}]" for tool in st.session_state.selected_tools])
                    question_with_tools = f"{question} {tool_requests}"
                else:
                    question_with_tools = question
                get_output_response(question_with_tools)
                st.session_state.question_input = ""
                st.session_state.selected_tools = []
        else:
            st.warning('Please upload a file or use the default file and click process on the side menu')

    if clear_clicked:
        st.session_state.chat_history = []
        st.session_state.question_input = ""
        st.rerun()

    # Show chat with latest at the bottom
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if __name__ == "__main__":
    pdf_main()