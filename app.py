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

st.title("Enterprise RAG Assistant 🚀")

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

# Ask question
query = st.text_input("Ask a question about the document")

if query:
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })

    st.write(response.content)

    # cleanup temp file
    try:
        os.remove(tmp_path)
    except:
        pass