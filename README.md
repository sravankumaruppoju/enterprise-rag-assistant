# 🧪 Enterprise RAG Assistant — Engineering Build (v2)

🔗 Live Demo (Engineering Build)  
https://enterprise-rag-assistant-es37iz4sph6bxfg4autk4p.streamlit.app/

---

## About This Branch

This branch represents the *development and system engineering stage* of the Enterprise RAG Assistant project.

The goal of this branch is to demonstrate how a Generative AI application is designed, tested, and improved before being promoted to a stable production deployment.

The stabilized production version of the project is available in the main branch.

Workflow used:

development (v2) → production release (main)

This reflects a real-world ML deployment lifecycle.

---

## Project Description

Enterprise RAG Assistant is a multi-document question-answering system built using Retrieval-Augmented Generation (RAG).

The application allows users to upload PDFs and ask contextual questions.  
Instead of guessing, the system retrieves relevant document sections and forces the LLM to answer only using retrieved evidence.

This reduces hallucinations and produces grounded responses.

---

## System Architecture

User  
↓  
Streamlit UI  
↓  
LangChain Retrieval Pipeline  
↓  
Chroma Vector Database  
↓  
OpenAI Embeddings  
↓  
OpenAI LLM

---

## Processing Pipeline

1. User uploads one or more PDF documents
2. Text is chunked using RecursiveCharacterTextSplitter
3. OpenAI embeddings are generated
4. Chunks stored inside Chroma vector database
5. Relevant chunks retrieved per question (Top-K retrieval)
6. LLM generates answer strictly from retrieved context
7. Sources displayed (file + page number)

---

## Key Engineering Features

### Multi-Document Retrieval
- Multiple PDFs uploaded simultaneously
- Automatic index refresh
- Cross-document search

### Hallucination Control
The prompt forces the model:

> “If the answer is not present in the context, respond that you don’t know.”

This prevents fabricated answers.

### Caching System
- SHA256 document fingerprinting
- Avoids re-embedding duplicate files
- Faster response time

### Secure Access
- Password protection
- API key stored using Streamlit Secrets
- Environment-based configuration

### Adjustable Retrieval
- Configurable Top-K chunk selection
- Adjustable temperature
- Model selection

---

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI (Embeddings + LLM)
- ChromaDB
- dotenv
- SHA256 hashing

---

## Purpose of v2

This branch focuses on *engineering the RAG pipeline*:

- retrieval quality
- hallucination reduction
- caching strategy
- document indexing
- prompt grounding

After validation, the system was stabilized and promoted to the main branch as the production deployment.

---

## What This Demonstrates

This branch shows practical Generative AI engineering skills:

- Retrieval-Augmented Generation architecture
- Vector database integration
- Prompt engineering for grounded answers
- Secure LLM usage
- Deployment workflow
- Versioned release management

---

## Author

*Sravan Kumar Uppoju*  
Senior Data Scientist | GenAI Enginee
