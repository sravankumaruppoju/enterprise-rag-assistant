import streamlit as st
import tempfile
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv(override=True)  # loads OPENAI_API_KEY from .env if present
# ---- ADD THIS BLOCK BELOW ----
# Load API key safely (works locally + cloud)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it in .env or Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# ---- END BLOCK ----
# -------- PASSWORD PROTECTION --------
def require_password():
    try:
        APP_PASSWORD = st.secrets["APP_PASSWORD"]
    except Exception:
        APP_PASSWORD = os.getenv("APP_PASSWORD")

    if not APP_PASSWORD:
        st.error("APP_PASSWORD not set.")
        st.stop()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("Enter password to access app", type="password")
        if st.button("Login"):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.success("Access granted ✅")
            else:
                st.error("Incorrect password ❌")
        st.stop()

require_password()
# -------- END PASSWORD PROTECTION --------

st.title("Enterprise RAG Assistant 🚀")
# --- Chat History Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("🗑 Clear Chat"):
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Embeddings + Vector DB
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")
# ---- Display Chat History ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ask question
query = st.chat_input("Ask a question about the document")

if query:
    docs = retriever.invoke(query)

context = ""
sources = []

for doc in docs:
    context += doc.page_content + "\n\n"
    sources.append(f"Page {doc.metadata.get('page', 'N/A')}")

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })

    st.markdown("### 📌 Answer")
st.markdown(response.content)

# Remove duplicate pages
unique_sources = list(set(sources))

st.markdown("### 📚 Sources")
st.markdown(", ".join(unique_sources))

    # cleanup temp file
    try:
        os.remove(tmp_path)
    except:
        pass