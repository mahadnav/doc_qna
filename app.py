# import streamlit as st
# from langchain.llms import OpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA, AnalyzeDocumentChain
# from langchain.document_loaders import PyPDFLoader, WebBaseLoader
# import tempfile
# import pysqlite3
# import sys
# import requests

# # Fix the sqlite3 module issue
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# # Page configuration
# st.set_page_config(page_title='🦜🔗 Ask Document', layout="wide")

# # Initialize session state variables
# if 'document_list' not in st.session_state:
#     st.session_state['document_list'] = []
# if 'query_history' not in st.session_state:
#     st.session_state['query_history'] = []
# if 'api_key' not in st.session_state:
#     st.session_state['api_key'] = ''
# if 'together_api_key' not in st.session_state:
#     st.session_state['together_api_key'] = ''

# def add_to_sidebar(doc_name):
#     """Update the sidebar with the new document."""
#     if doc_name not in st.session_state['document_list']:
#         st.session_state['document_list'].append(doc_name)

# def load_document(file=None, url=None):
#     """Load a document from a file or URL."""
#     if file is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file.read())
#             temp_file_path = temp_file.name
#         loader = PyPDFLoader(temp_file_path)
#         documents = loader.load()
#         add_to_sidebar(file.name)
#         return documents
#     elif url is not None:
#         loader = WebBaseLoader(url)
#         documents = loader.load()
#         add_to_sidebar(url)
#         return documents

# def generate_response(documents, openai_api_key, query_text):
#     """Generate a response from the loaded documents."""
#     try:
#         text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#         texts = text_splitter.split_documents(documents)
#         embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#         db = Chroma.from_documents(texts, embeddings, persist_directory="chromadb_storage")
#         db.persist()
#         retriever = db.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 relevant chunks
#         qa = RetrievalQA.from_chain_type(
#             llm=OpenAI(
#                 openai_api_key=openai_api_key,
#                 max_tokens=150  # Limit response length
#             ),
#             chain_type='stuff',
#             retriever=retriever
#         )
#         return qa.run(query_text)
#     except Exception as e:
#         return f"Error: {str(e)}"

# def generate_code(prompt, together_api_key):
#     """Generate code using Together.ai and Code Llama."""
#     try:
#         url = "https://api.together.ai/code"
#         headers = {"Authorization": f"Bearer {together_api_key}"}
#         data = {"prompt": prompt, "model": "code-llama"}
#         response = requests.post(url, headers=headers, json=data)
#         if response.status_code == 200:
#             return response.json().get('code', "No code returned")
#         else:
#             return f"Error: {response.json().get('message', 'Unknown error')}"
#     except Exception as e:
#         return f"Error: {str(e)}"

# def summarize_document(documents):
#     """Summarize the loaded documents."""
#     summarizer = AnalyzeDocumentChain()
#     return summarizer.run(documents)

# # Sidebar
# with st.sidebar:
#     st.session_state['api_key'] = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key")
#     st.session_state['together_api_key'] = st.text_input("Together.ai API Key", type="password", placeholder="Enter your Together.ai API key")
#     st.write("**Loaded Documents**")
#     for doc in st.session_state['document_list']:
#         st.write(f"- {doc}")

# # Tabbed layout
# tabs = st.tabs(["Document Q&A", "Code Generation", "Document Summarization", "Download Document"])

# # Tab 1: Document Q&A
# with tabs[0]:
#     st.title("Document Question & Answer")
#     uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
#     uploaded_url = st.text_input('Enter a website URL (optional)')
#     documents = []
#     if uploaded_file:
#         documents = load_document(file=uploaded_file)
#     elif uploaded_url:
#         documents = load_document(url=uploaded_url)
#     query_text = st.text_input('Enter your question:', placeholder='Ask something about the loaded documents.', disabled=not documents)
#     if st.session_state['api_key'] and query_text and documents:
#         with st.spinner('Generating response...'):
#             response = generate_response(documents, st.session_state['api_key'], query_text)
#             st.session_state['query_history'].append((query_text, response))
#             st.write("**Response:**", response)

# # Tab 2: Code Generation
# with tabs[1]:
#     st.title("Code Generation with Together.ai and Code Llama")
#     code_prompt = st.text_area("Enter your coding prompt:")
#     if st.session_state['together_api_key'] and code_prompt:
#         with st.spinner("Generating code..."):
#             generated_code = generate_code(code_prompt, st.session_state['together_api_key'])
#             st.code(generated_code, language="python")

# # Tab 3: Document Summarization
# with tabs[2]:
#     st.title("Summarize Documents")
#     if documents:
#         if st.button("Summarize Document"):
#             summary = summarize_document(documents)
#             st.write("**Summary:**", summary)


# # Tab 4: Download Document
# with tabs[4]:
#     st.title("Download Loaded Documents")
#     if documents:
#         download_button = st.download_button(
#             label="Download Document",
#             data="\n".join([doc.page_content for doc in documents]),
#             file_name="document.txt",
#             mime="text/plain"
#         )



import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from docling import DoclingClient  # Ensure you have Docling installed
import tempfile

# Initialize Docling Client (replace 'your_api_key' with an actual API key)
docling_client = DoclingClient(api_key='your_api_key')

# Initialize session state variables
if 'document_list' not in st.session_state:
    st.session_state['document_list'] = []
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

def add_to_sidebar(doc_name):
    """Update the sidebar with the new document."""
    if doc_name not in st.session_state['document_list']:
        st.session_state['document_list'].append(doc_name)

def extract_text_from_pdf(pdf_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
    with fitz.open(temp_file_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def extract_text_from_excel(excel_file):
    df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
    text = ""
    for sheet_name, sheet in df.items():
        text += sheet.to_string(index=False) + "\n"
    return text

def summarize_document(text):
    """Summarize the document using Docling."""
    return docling_client.summarize(text)

def extract_quantitative_data(excel_file):
    """Extract numerical data from an Excel file."""
    df = pd.read_excel(excel_file, sheet_name=None)
    return df


st.set_page_config(page_title='📄 Docling-Powered Q&A App', layout="wide")
st.title("📄 Docling-Powered Q&A App")

uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx", "xls"])
document_text = ""

if uploaded_file:
    file_type = uploaded_file.type
    
    with st.spinner("Extracting content..."):
        if "pdf" in file_type:
            document_text = extract_text_from_pdf(uploaded_file)
        elif "excel" in file_type or "spreadsheet" in file_type:
            document_text = extract_text_from_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
    
    add_to_sidebar(uploaded_file.name)
    st.success("File processed successfully!")

# Tab layout
tab1, tab2, tab3 = st.tabs(["Document Q&A", "Document Summary", "Quantitative Data"])

with tab1:
    st.header("Ask Questions About the Document")
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Fetching answer..."):
            response = docling_client.ask(document_text, question)
            st.session_state['query_history'].append((question, response))
            st.write("**Answer:**", response)

with tab2:
    st.header("Document Summary")
    if document_text:
        if st.button("Summarize Document"):
            with st.spinner("Summarizing..."):
                summary = summarize_document(document_text)
                st.write("**Summary:**", summary)

with tab3:
    st.header("Extracted Quantitative Data")
    if "excel" in uploaded_file.type:
        with st.spinner("Extracting data..."):
            df_dict = extract_quantitative_data(uploaded_file)
            for sheet, df in df_dict.items():
                st.subheader(f"Sheet: {sheet}")
                st.dataframe(df)

# Sidebar
with st.sidebar:
    st.write("**Loaded Documents**")
    for doc in st.session_state['document_list']:
        st.write(f"- {doc}")
    
    st.write("**Query History**")
    for query, response in st.session_state['query_history']:
        st.write(f"Q: {query}")
        st.write(f"A: {response}")