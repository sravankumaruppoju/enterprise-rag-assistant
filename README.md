# 🚀 Enterprise RAG Assistant

🔴 *Live AI Application*  
https://enterprise-rag-assistant-ai.streamlit.app

---

## What This Is
A production-style Retrieval-Augmented Generation (RAG) system that allows users to upload private documents and query them using grounded LLM responses.

This project demonstrates how companies connect Large Language Models to internal data securely.

This is not a chatbot — it is a document intelligence system.

---

## Problem
LLMs hallucinate because they don’t have access to company knowledge.

Organizations need AI assistants that:
- understand internal documents
- answer accurately
- do not fabricate information

---

## Solution
This system connects an LLM to a retrieval layer over uploaded PDFs.

The model *only answers using retrieved document context*.  
If the answer is not found → it refuses to guess.

---

## How It Works
1. User uploads PDFs
2. Documents are chunked
3. OpenAI embeddings convert text to vectors
4. Stored in Chroma vector database
5. Relevant chunks retrieved per question
6. LLM generates grounded answer
7. Source page references returned

---

## System Architecture
User → Streamlit UI → LangChain Retrieval → Chroma Vector DB → OpenAI Embeddings → LLM Response

---

## Key Features
- Multi-PDF document search
- Source citations (file + page)
- Hallucination control prompting
- Password-protected access
- Embedding caching (SHA256)
- Configurable retrieval parameters
- Cloud deployment (Streamlit)

---

## Tech Stack
Python • Streamlit • LangChain • OpenAI • ChromaDB

---

## Real-World Use Cases
- Enterprise knowledge assistants
- Legal document analysis
- Healthcare policy QA
- Financial report querying
- Compliance document review

---

## Why This Matters
This project demonstrates production GenAI engineering:
- Retrieval pipelines
- Vector databases
- Prompt grounding
- Secure deployment
- Versioned releases

---

## Author
*Sravan Kumar Uppoju*  
Senior Data Scientist / GenAI Engineer
