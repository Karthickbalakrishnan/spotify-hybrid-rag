Markdown# Smart Document Assistant — Multimodal Hybrid RAG System

**Live Production Endpoint:** [https://hybridrag.karthick.tech/](https://hybridrag.karthick.tech/)

A production-deployed, multimodal **Hybrid Retrieval-Augmented Generation (RAG)** web application built with **Streamlit**, **LangChain**, **ChromaDB**, and **OpenAI**. Designed under a strict **RTFCR (Role, Task, Format, Context, Rules)** architectural specification, the system ingests multi-format documents into a persistent personal library and answers user queries strictly from uploaded evidence across four interaction modes: **Text Q&A**, **Live Voice Input & Spoken Replies**, **Visual Page Inspection**, and **Narrated Video Walkthroughs**.

---

## Table of Contents
1. [RTFCR System Specification](#rtfcr-system-specification)
2. [Core Capabilities](#core-capabilities)
3. [End-to-End Architecture](#end-to-end-architecture)
4. [Technology Stack](#technology-stack)
5. [Repository Structure](#repository-structure)
6. [Local Installation & Setup](#local-installation--setup)
7. [Production Cloud Deployment Pipeline](#production-cloud-deployment-pipeline)
8. [Author](#author)

---

## Core Capabilities

* **Hybrid Search with Tunable Precision:** Combines **ChromaDB** semantic embeddings with **BM25** lexical matching via **Reciprocal Rank Fusion (RRF)**, exposed through an interactive *Exact Words ↔ Overall* search balance control[cite: 4].
* **Persistent Document Library & Metadata Filtering:** Tracks saved files, extracted text sections, and upload dates, allowing targeted retrieval across all files or individual documents[cite: 4].
* **Privacy & Safety Guardrails:** Built-in prompt filtering and strict context-grounding rules protect against prompt injection and unsupported answers[cite: 4].
* **4-in-1 Multimodal Interface[cite: 4]:**
  * **Ask Questions (Text & Voice):** Type queries or record live audio via microphone with optional spoken read-aloud replies[cite: 4].
  * **View Pages as Images:** Inspect high-resolution rendered pages of uploaded documents directly in the UI[cite: 4].
  * **Watch Video Walkthrough:** Automatically generates narrated video summaries combining visual slides with OpenAI TTS audio[cite: 4].
* **LangSmith Observability:** Logs retrieval latency, chunk rankings, prompt tokens, and completion traces for continuous evaluation.

---

## End-to-End Architecture

```text
+-------------------------------------------------------------------+
|                        Client Browser (HTTPS)                     |
|                  [https://hybridrag.karthick.tech/](https://hybridrag.karthick.tech/)                 |
+---------------------------------+---------------------------------+
                                  | TLS 1.3 / SSL (Port 443)
                                  v
+-------------------------------------------------------------------+
|               Nginx Reverse Proxy (Ubuntu 24.04 VPS)              |
|     WebSocket Upgrade | 200MB Max Body | 300s Timeout | Certbot   |
+---------------------------------+---------------------------------+
                                  | HTTP Proxy (127.0.0.1:8501)
                                  v
+-------------------------------------------------------------------+
|               Docker Container: smart-doc-qa-live                 |
|  +-------------------------------------------------------------+  |
|  |                    Streamlit Web Interface                  |  |
|  +------------------------------+------------------------------+  |
|                                 |                                 |
|  +------------------------------v------------------------------+  |
|  |           LangChain Orchestration & Safety Guardrails       |  |
|  +---------------+-----------------------------+---------------+  |
|                  |                             |                  |
|  +---------------v---------------+ +-----------v---------------+  |
|  |     Hybrid Search Engine      | |   Multimodal Synthesis    |  |
|  |  - ChromaDB (Dense Vectors)   | |  - Whisper STT / TTS      |  |
|  |  - BM25 (Sparse Keywords)     | |  - PyMuPDF Page Renderer  |  |
|  |  - Reciprocal Rank Fusion     | |  - MoviePy + FFmpeg Video |  |
|  +-------------------------------+ +---------------------------+  |
+-------------------------------------------------------------------+
Hybrid Retrieval Formula (Reciprocal Rank Fusion)To balance conceptual similarity with exact keyword precision, retrieved candidates from ChromaDB ($r_{\text{dense}}$) and BM25 ($r_{\text{sparse}}$) are merged using RRF:$$\text{RRF}(d) = \sum_{r \in \{r_{\text{dense}},\, r_{\text{sparse}}\}} \frac{1}{k + \text{rank}_r(d)}$$(where $k = 60$ prevents top-ranked outliers from dominating fused relevance scores).Technology StackLayerTechnologiesFrontend & UIStreamlit (reactive WebSocket state, audio recorder, media player)LLM & OrchestrationPython 3.11, LangChain, OpenAI API (GPT, Embeddings, Whisper STT, TTS)Hybrid RetrievalChromaDB (persistent vector store), BM25, Reciprocal Rank Fusion (RRF)Document & Video EnginePyMuPDF (fitz), MoviePy, ffmpegObservabilityLangSmith (LANGCHAIN_TRACING_V2)DevOps & Cloud EdgeDocker, Docker Compose, Ubuntu 24.04 VPS (Hostinger), Nginx, Let's Encrypt (Certbot)Repository StructurePlaintextspotify-hybrid-rag/
├── app.py                 # Main Streamlit application, Hybrid RAG pipeline & multimodal UI
├── requirements.txt       # Python dependencies (LangChain, ChromaDB, PyMuPDF, MoviePy, etc.)
├── Dockerfile             # Python 3.11 + FFmpeg container image build definition
├── docker-compose.yml     # Service orchestration, port binding (8501) & persistent volumes
├── .env.example           # Template for required OpenAI & LangSmith environment variables
├── .gitignore             # Excludes .env, local ChromaDB persistence, and temporary media
└── README.md              # Complete architectural & deployment documentation

Local Installation & Setup1. Clone the RepositoryBashgit clone [https://github.com/Karthickbalakrishnan/spotify-hybrid-rag.git](https://github.com/Karthickbalakrishnan/spotify-hybrid-rag.git)
cd spotify-hybrid-rag
2. Configure Environment VariablesCreate a .env file in the project root (never commit this file to Git):Code snippetOPENAI_API_KEY=sk-your-openai-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-your-langsmith-api-key
LANGCHAIN_PROJECT=hybrid-rag-guarded-bot
3. Run via Docker Compose (Recommended)Bashdocker compose up -d --build
Open http://localhost:8501 in your browser.4. Run Locally via Python Virtual Environment (Alternative)Bashpython3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
Production Cloud Deployment PipelineThe production instance at https://hybridrag.karthick.tech/ follows a 6-stage hardened deployment architecture:Containerization: Packaged Python 3.11, Streamlit, ChromaDB, and OS-level ffmpeg dependencies into a reproducible Docker image (smart-doc-qa-live) with restart: unless-stopped.VPS Provisioning & Hardening: Configured an Ubuntu 24.04 VPS with a dedicated non-root sudo user, UFW firewall rules (Nginx Full), and restricted .env permissions (chmod 600).Port Conflict Resolution & Reverse Proxy: Freed port 80 from pre-installed default services and configured Nginx to proxy external traffic to 127.0.0.1:8501 with WebSocket upgrade headers (Upgrade and Connection "upgrade"), client_max_body_size 200M for large PDF ingestion, and 300s read timeouts for video rendering.DNS Subdomain Mapping: Pointed the hybridrag A record to the VPS public IPv4 address.   Automated SSL/TLS Encryption: Provisioned a Let's Encrypt TLS certificate via Certbot (--nginx --redirect) with automated 90-day renewal, enforcing HTTPS and enabling browser microphone access for voice queries[cite: 4].AuthorKarthick BalakrishnanUX Strategist & Technical Lead | AI & Spatial Computing ResearcherLive Application: https://hybridrag.karthick.tech/GitHub: github.com/Karthickbalakrishnan
