# 🧠 Enterprise RAG Assistant — v2 (Engineering Build)

🔬 Development Version of the production GenAI system.

Live Demo:
https://enterprise-rag-assistant-es37iz4sph6bxfg4autk4p.streamlit.app/

---

## Purpose of v2
This branch represents the engineering iteration of the Enterprise RAG Assistant where system reliability, security, and retrieval quality were improved before merging to main.

The goal of v2 was to transform a simple RAG prototype into a production-style GenAI application.

---

## What Changed From v1
v1 was a basic PDF Q&A chatbot.

v2 introduced system-level capabilities:

- Multi-document ingestion
- Retrieval caching
- Password protection
- Configurable retrieval parameters
- Controlled prompting to reduce hallucinations
- Deployment-ready configuration

---

## Architecture

User Query  
↓  
Streamlit Interface  
↓  
LangChain Retrieval Pipeline  
↓  
Chroma Vector Store  
↓  
OpenAI Embeddings  
↓  
LLM (gpt-4o-mini / gpt-4.1-mini)  
↓  
Grounded Response + Source Pages

---

## Retrieval Pipeline

1. Upload PDF(s)
2. Text extracted using PyPDF loader
3. Recursive chunking
4. OpenAI embeddings generated
5. Stored in Chroma vector DB
6. Top-K retrieval per query
7. Context passed to LLM
8. LLM restricted to retrieved context

If the answer is not found:
> “I don’t know based on the uploaded document.”

This prevents hallucination.

---

## Key Engineering Features

### 🔐 Secure Access
- Password-protected UI
- Streamlit Secrets API key management
- Environment configuration

### 📂 Multi-Document Support
- Upload multiple PDFs simultaneously
- Automatic index refresh
- Duplicate detection using SHA256 document hashing

### ⚙️ Retrieval Controls
- Adjustable Top-K
- Temperature control
- Model selection

### 🚀 Performance Optimization
- @st.cache_resource used for:
  - embeddings
  - vector store
  - LLM initialization

Reduces latency and cost.

---

## Why This Branch Matters
This branch demonstrates the transition from:

*Prototype → Reliable GenAI System*

Key learnings implemented:
- retrieval grounding
- caching strategy
- prompt control
- cloud deployment workflow
- secure API handling

---

## Tech Stack
Python  
Streamlit  
LangChain  
OpenAI API  
ChromaDB  
dotenv

---

## Future Work
- Persistent vector database (Pinecone)
- FastAPI backend
- RAG evaluation (RAGAS)
- LangSmith tracing

---

## Author
Sravan Kumar Uppoju  
Senior Data Scientist / GenAI Engineer
