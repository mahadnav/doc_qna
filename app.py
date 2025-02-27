import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, AnalyzeDocumentChain
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
import tempfile
import pysqlite3
import sys
import requests
import docling

# Fix the sqlite3 module issue
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Page configuration
st.set_page_config(page_title='🦜🔗 Ask Document', layout="wide")

# Initialize session state variables
if 'document_list' not in st.session_state:
    st.session_state['document_list'] = []
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []
if 'hf_api_key' not in st.session_state:
    st.session_state['hf_api_key'] = ''

def add_to_sidebar(doc_name):
    """Update the sidebar with the new document."""
    if doc_name not in st.session_state['document_list']:
        st.session_state['document_list'].append(doc_name)

def load_document(file=None, url=None):
    """Load a document from a file or URL."""
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        add_to_sidebar(file.name)
        return documents
    elif url is not None:
        loader = WebBaseLoader(url)
        documents = loader.load()
        add_to_sidebar(url)
        return documents

def generate_response(documents, hf_api_key, query_text):
    """Generate a response from the loaded documents."""
    try:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        db = Chroma.from_documents(texts, embeddings, persist_directory="chromadb_storage")
        db.persist()
        retriever = db.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 relevant chunks
        qa = RetrievalQA.from_chain_type(
            llm=HuggingFaceHub(
                repo_id="tiiuae/falcon-7b-instruct",  # Example free model
                model_kwargs={"temperature": 0.5, "max_length": 150}
            ),
            chain_type='stuff',
            retriever=retriever
        )
        return qa.run(query_text)
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_document(documents):
    """Summarize the loaded documents."""
    summarizer = AnalyzeDocumentChain()
    return summarizer.run(documents)

# Sidebar
with st.sidebar:
    st.session_state['hf_api_key'] = st.text_input("Hugging Face API Key", type="password", placeholder="Enter your Hugging Face API key")
    st.write("**Loaded Documents**")
    for doc in st.session_state['document_list']:
        st.write(f"- {doc}")

# Tabbed layout
tabs = st.tabs(["Document Q&A", "Document Summarization", "Download Document"])

# Tab 1: Document Q&A
with tabs[0]:
    st.title("Document Question & Answer")
    uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
    uploaded_url = st.text_input('Enter a website URL (optional)')
    documents = []
    if uploaded_file:
        documents = load_document(file=uploaded_file)
    elif uploaded_url:
        documents = load_document(url=uploaded_url)
    query_text = st.text_input('Enter your question:', placeholder='Ask something about the loaded documents.', disabled=not documents)
    if st.session_state['hf_api_key'] and query_text and documents:
        with st.spinner('Generating response...'):
            response = generate_response(documents, st.session_state['hf_api_key'], query_text)
            st.session_state['query_history'].append((query_text, response))
            st.write("**Response:**", response)

# Tab 2: Document Summarization
with tabs[1]:
    st.title("Summarize Documents")
    if documents:
        if st.button("Summarize Document"):
            summary = summarize_document(documents)
            st.write("**Summary:**", summary)

# Tab 3: Download Document
with tabs[2]:
    st.title("Download Loaded Documents")
    if documents:
        download_button = st.download_button(
            label="Download Document",
            data="\n".join([doc.page_content for doc in documents]),
            file_name="document.txt",
            mime="text/plain"
        )
