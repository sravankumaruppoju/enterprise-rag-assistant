import os
import tempfile
import hashlib
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# -------------------- CONFIG --------------------
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🚀", layout="centered")
load_dotenv(override=True)

# -------------------- SECRETS / ENV --------------------
def get_secret(name: str):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets or .env")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -------------------- PASSWORD PROTECTION --------------------
def require_password():
    APP_PASSWORD = get_secret("APP_PASSWORD")

    if not APP_PASSWORD:
        st.error("APP_PASSWORD not set. Add APP_PASSWORD in Streamlit Secrets or .env")
        st.stop()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔐 Secure Access")
        pwd = st.text_input("Enter password", type="password", key="login_pwd")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Login"):
                if pwd == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.success("Access granted ✅")
                    st.rerun()
                else:
                    st.error("Incorrect password ❌")
        with c2:
            st.caption("Tip: Set APP_PASSWORD in Streamlit Secrets.")
        st.stop()

def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.messages = []
        st.session_state.vector_ready = False
        st.session_state.db = None
        st.rerun()

require_password()


# -------------------- UI --------------------
st.title("Enterprise RAG Assistant 🚀")

st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Top-k chunks", 2, 10, 4)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)
logout_button()

st.sidebar.markdown("---")
st.sidebar.markdown("*How it works*")
st.sidebar.markdown(
    "- Upload PDF(s)\n"
    "- We chunk + embed + store in Chroma\n"
    "- Retrieve top chunks per question\n"
    "- Answer strictly from retrieved context\n"
)


# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "db" not in st.session_state:
    st.session_state.db = None


# -------------------- HELPERS --------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_resource(show_spinner=False)
def get_llm(model: str, temp: float):
    return ChatOpenAI(model=model, temperature=temp)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings()

def save_uploaded_to_temp(uploaded) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name

def load_docs_from_pdfs(temp_paths: List[Tuple[str, str]]):
    all_docs = []
    for pdf_name, path in temp_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = pdf_name
        all_docs.extend(docs)
    return all_docs

@st.cache_resource(show_spinner=False)
def build_vectorstore(file_hashes: Tuple[str, ...], temp_paths: Tuple[Tuple[str, str], ...]):
    documents = load_docs_from_pdfs(list(temp_paths))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    db = Chroma.from_documents(chunks, embeddings)
    return db

def render_sources(docs):
    seen = set()
    items = []
    for d in docs:
        src = d.metadata.get("source_file", "PDF")
        page = d.metadata.get("page", "N/A")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            preview = (d.page_content or "").strip().replace("\n", " ")
            preview = preview[:240] + ("..." if len(preview) > 240 else "")
            items.append((src, page, preview))

    st.markdown("### 📄 Sources")
    for src, page, preview in items:
        with st.expander(f"{src} — Page {page}"):
            st.write(preview)


# -------------------- PROMPTS --------------------
qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not present in the context, say: "I don't know based on the uploaded document."

Context:
{context}

Question:
{question}
""")

summary_prompt = ChatPromptTemplate.from_template("""
Create an executive summary from the context:
- 5 bullet highlights
- 3 key takeaways
- If it's a resume: summarize profile in 3 lines

Context:
{context}
""")

skills_prompt = ChatPromptTemplate.from_template("""
Extract and categorize skills from the context.
Categories:
- Programming
- ML/AI
- GenAI/LLM
- MLOps/DevOps
- Cloud
- Databases
Return as clean bullet points.

Context:
{context}
""")


# -------------------- MAIN: Upload --------------------
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

colA, colB = st.columns([1, 1])
with colA:
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
with colB:
    st.caption("Upload new PDF(s) anytime — index will refresh automatically.")

temp_paths = []
file_hashes = []

if uploaded_files:
    for f in uploaded_files:
        data = f.getvalue()
        file_hashes.append(sha256_bytes(data))
        temp_paths.append((f.name, save_uploaded_to_temp(f)))

    file_hashes_t = tuple(file_hashes)
    temp_paths_t = tuple(temp_paths)

    with st.spinner("Indexing PDFs (chunking + embeddings)..."):
        db = build_vectorstore(file_hashes_t, temp_paths_t)

    st.session_state.db = db
    st.session_state.vector_ready = True
    st.success("PDF(s) indexed successfully ✅")

    # Action Buttons
    b1, b2 = st.columns([1, 1])

    with b1:
        if st.button("📌 Generate Executive Summary"):
            llm = get_llm(model_name, temperature)
            docs_for_summary = load_docs_from_pdfs(temp_paths)[:8]
            summary_context = "\n\n".join([d.page_content for d in docs_for_summary if d.page_content])

            chain = summary_prompt | llm
            resp = chain.invoke({"context": summary_context})
            st.markdown("## 📌 Executive Summary")
            st.markdown(resp.content)

    with b2:
        if st.button("🧠 Extract Skills"):
            llm = get_llm(model_name, temperature)
            docs_for_skills = load_docs_from_pdfs(temp_paths)[:12]
            skills_context = "\n\n".join([d.page_content for d in docs_for_skills if d.page_content])

            chain = skills_prompt | llm
            resp = chain.invoke({"context": skills_context})
            st.markdown("## 🧠 Extracted Skills")
            st.markdown(resp.content)

else:
    st.session_state.vector_ready = False
    st.session_state.db = None


# -------------------- CHAT DISPLAY --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the document", key="main_chat_input")

if query:
    if not st.session_state.vector_ready or st.session_state.db is None:
        st.warning("Upload at least one PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    retriever = st.session_state.db.as_retriever(search_kwargs={"k": top_k})

    with st.spinner("Retrieving + generating answer..."):
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs if d.page_content])

        llm = get_llm(model_name, temperature)
        chain = qa_prompt | llm
        resp = chain.invoke({"context": context, "question": query})
        answer = resp.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
        render_sources(docs)

# Cleanup temp files (best effort)
for _, p in temp_paths:
    try:
        os.remove(p)
    except Exception:
        pass