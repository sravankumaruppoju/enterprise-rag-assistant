🚀 Enterprise RAG Assistant (v2.0)

A production-ready Retrieval-Augmented Generation (RAG) system built using Streamlit, LangChain, OpenAI, and Chroma.

This application allows users to securely upload multiple PDFs and ask contextual questions powered by OpenAI embeddings and LLMs.

⸻

🔗 Live Demo

👉 Add your Streamlit URL here
🏷 Latest Release: v2.0

⸻

📌 Overview

Enterprise RAG Assistant is a secure multi-document question-answering system designed with real-world GenAI architecture principles.

The system:
	•	Accepts multiple PDF documents
	•	Chunks and embeds text using OpenAI embeddings
	•	Stores vectors in Chroma
	•	Retrieves relevant context dynamically
	•	Generates grounded answers using OpenAI LLMs
	•	Provides source references (file + page)
	•	Enforces password protection for secure access

This is not a simple chatbot — it is an enterprise-grade RAG implementation

🧠 System Architecture
User
  ↓
Streamlit UI
  ↓
LangChain Orchestration
  ↓
Chroma Vector Store
  ↓
OpenAI Embeddings
  ↓
OpenAI LLM (gpt-4o-mini / gpt-4.1-mini)

Processing Flow
	1.	User uploads one or more PDFs
	2.	Text is chunked using RecursiveCharacterTextSplitter
	3.	Embeddings are generated via OpenAI
	4.	Chunks are stored in Chroma vector DB
	5.	Top-K relevant chunks are retrieved per query
	6.	LLM generates answer strictly from retrieved context
	7.	Sources are displayed for transparency

⸻

✨ Key Features

🔐 Secure Access
	•	Password protection
	•	Secure API key handling via Streamlit Secrets
	•	Environment-based configuration

📂 Multi-PDF Support
	•	Upload multiple PDFs simultaneously
	•	Automatic indexing refresh
	•	SHA256-based caching to avoid duplicate embeddings

⚙️ Configurable Retrieval
	•	Adjustable Top-K chunk retrieval
	•	Adjustable temperature
	•	Model selection (gpt-4o-mini / gpt-4.1-mini)

🧠 Hallucination Reduction

The prompt enforces:

If the answer is not present in the context, respond with:
“I don’t know based on the uploaded document.”

This ensures grounded responses.

🚀 Production-Ready Design
	•	Cached embeddings & LLM
	•	Clean Git branching workflow (v2 → main)
	•	Versioned release tagging
	•	Cloud deployment on Streamlit

⸻

🛠 Tech Stack
	•	Python
	•	Streamlit
	•	LangChain
	•	OpenAI (Embeddings + LLM)
	•	Chroma (Vector Database)
	•	dotenv
	•	SHA256 hashing (document fingerprinting)

⸻

🔎 Engineering Highlights

1️⃣ Recursive Chunking

Improves semantic retrieval quality over simple character splitting.

2️⃣ Vector Caching

Prevents re-embedding identical documents using SHA256 fingerprinting.

3️⃣ Strict Context Prompting

Minimizes hallucination by restricting answers to retrieved context only.

4️⃣ Resource Caching

@st.cache_resource used for:
	•	LLM initialization
	•	Embedding model
	•	Vector store build

Improves performance and scalability.

⸻

📊 Scalability Path

This architecture can easily evolve to:
	•	Persistent vector storage (Pinecone / Weaviate)
	•	Hybrid retrieval (BM25 + Vector search)
	•	FastAPI backend for production
	•	Role-based access control
	•	RAG evaluation (RAGAS)
	•	Streaming LLM responses
	•	LangSmith observability integration

  🎯 Use Cases
	•	Enterprise knowledge base assistant
	•	Legal document analysis
	•	Healthcare policy QA
	•	Financial document retrieval
	•	Resume & compliance document parsing

⸻

👤 Author

Sravan Kumar Uppoju
Senior Data Scientist | GenAI Engineer

⸻

⭐ Why This Project Stands Out

This project demonstrates:
	•	Real-world RAG architecture
	•	Secure GenAI system design
	•	Vector database integration
	•	Prompt engineering for hallucination control
	•	Cloud deployment workflow
	•	Version control & release management

It reflects production-level AI engineering practices.
