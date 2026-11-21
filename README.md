# Smart Document Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A multimodal document Q&A application that answers questions strictly from uploaded PDF, DOCX, and TXT files. It combines semantic and keyword retrieval, cross-encoder reranking, OpenAI generation, document-grounded evaluations, voice interaction, page rendering, and narrated video walkthroughs.

**Live demo:** [https://hybridrag.karthick.tech/](https://hybridrag.karthick.tech/)  
**Built with:** Streamlit · LangChain · ChromaDB · BM25 · FlashRank · OpenAI · LangSmith

## Features

- **Document library** — save PDFs, Word documents, and text files, inspect upload dates and file sections, filter retrieval by file or date, replace existing files, and delete the library.
- **Hybrid retrieval** — combine ChromaDB dense embeddings with BM25 sparse search using Reciprocal Rank Fusion (RRF).
- **Adjustable search strategy** — move from exact-word matching to broad semantic matching with a single slider.
- **Cross-encoder reranking** — optionally rerank retrieved chunks with FlashRank's `ms-marco-MiniLM-L-12-v2` model.
- **Grounded answers** — responses use only retrieved document excerpts and include file, page, and upload metadata.
- **Answer verification** — evaluate faithfulness and relevance before showing confidence indicators.
- **Voice Q&A** — record a question with Whisper transcription and optionally hear an answer through OpenAI text-to-speech.
- **Page viewer** — render uploaded PDF pages as high-resolution PNG images and browse them in the interface.
- **Video walkthrough** — generate an MP4 narrated with page summaries and OpenAI TTS, then preview or download it.
- **Safety and privacy controls** — redact common PII, run OpenAI moderation, detect prompt injection and jailbreak attempts, and enforce document-grounded response rules.
- **Accessible interface** — high-contrast surfaces, visible keyboard focus indicators, semantic status messages, and responsive layout.
- **LangSmith observability** — trace ingestion, retrieval, guardrails, generation, voice, and video operations; collect answer feedback and live evaluation scores.

## How It Works

1. A PDF, DOCX, or TXT file is uploaded and split into overlapping text chunks.
2. Each chunk is stored in persistent ChromaDB with source-file, upload-date, upload-time, and PDF-page metadata.
3. A follow-up question is rewritten into a standalone search query using recent conversation context.
4. ChromaDB retrieves semantically similar chunks while BM25 retrieves exact keyword matches.
5. Reciprocal Rank Fusion combines both result lists. FlashRank reranks the strongest candidates when enabled.
6. The final chunks are formatted with source metadata and sent to GPT-4o-mini with a strict grounding prompt.
7. A second model checks the generated answer for document support and relevance.

## Architecture

```text
Browser
  │ HTTPS
  ▼
Nginx Reverse Proxy
  │ WebSocket and HTTP proxy
  ▼
Docker Container — port 8501
  ├── Streamlit UI
  ├── LangChain orchestration
  ├── Input and output guardrails
  ├── OpenAI embeddings, GPT-4o-mini, Whisper, TTS, and moderation
  ├── ChromaDB persistent vector store
  ├── BM25 sparse retriever
  ├── FlashRank cross-encoder reranker
  ├── PyMuPDF page renderer
  ├── MoviePy and FFmpeg media pipeline
  └── LangSmith tracing and feedback

Persistent volumes
  ├── chroma_db       Vector and document metadata
  ├── flashrank_cache Cross-encoder model cache
  └── media_cache     PDF images and generated MP4 files
```

### Hybrid Retrieval Formula

For a dense rank `r_dense` and sparse rank `r_sparse`, each retriever contributes:

```text
RRF(doc) = dense_weight × 1 / (60 + r_dense)
         + sparse_weight × 1 / (60 + r_sparse)

where sparse_weight = 1 - dense_weight
```

The default dense/sparse weight is `0.5 / 0.5`. The slider changes this balance at runtime, and FlashRank performs a final accuracy reranking.

## Repository Structure

```text
.
├── app.py              Streamlit UI and complete RAG, multimodal, and guardrail pipeline
├── requirements.txt    Python dependencies
├── Dockerfile          Python 3.11 image with FFmpeg
├── docker-compose.yml  Container, port, environment, and persistent-volume configuration
├── .gitignore          Local secrets, databases, caches, media, and virtual environments
├── LICENSE             MIT License
└── README.md           Project documentation
```

## Technology Stack

| Area | Technology |
|---|---|
| Interface | Streamlit |
| RAG orchestration | LangChain, LangChain Core |
| Dense retrieval | OpenAI embeddings, ChromaDB |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Reranking | FlashRank cross-encoder |
| Language models | OpenAI GPT-4o-mini |
| Speech | OpenAI Whisper and TTS |
| Moderation | OpenAI Moderation API |
| Document parsing | PyPDF, PyMuPDF, python-docx tools, Docx2txt |
| Video | MoviePy, FFmpeg |
| Observability | LangSmith |
| Deployment | Docker, Docker Compose, Nginx, TLS |

## Local Setup

### Prerequisites

- Python 3.11+
- Docker with Docker Compose, or a Python virtual environment
- An OpenAI API key
- Optional: a LangSmith API key for tracing and feedback
- FFmpeg is included in the Docker image and is required when running Python directly

### 1. Clone the repository

```bash
git clone https://github.com/Karthickbalakrishnan/spotify-hybrid-rag.git
cd spotify-hybrid-rag
```

### 2. Create the environment file

Create `.env` in the project root:

```dotenv
OPENAI_API_KEY=sk-your-openai-api-key

# Optional LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-your-langsmith-api-key
LANGCHAIN_PROJECT=smart-document-assistant
```

Never commit `.env` or API keys to Git.

### 3A. Run with Docker Compose

```bash
docker compose up -d --build
```

Open [http://localhost:8501](http://localhost:8501).

Useful commands:

```bash
docker compose logs -f
docker compose down
```

### 3B. Run with Python

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Using the Application

1. Open the **Upload & Save a Document** panel.
2. Upload a PDF, DOCX, or TXT file and click **Save to Library**.
3. Ask a question in the chat, or expand the voice recorder to speak it.
4. Use **Search Settings** to select files and dates, tune dense/sparse weighting, or enable reranking.
5. Open **View Matching Paragraphs** to inspect the source excerpts behind an answer.
6. Use the image tab to browse PDF pages.
7. Use the video tab to generate and download a narrated walkthrough.

## Supported Files and Limits

- Supported uploads: `.pdf`, `.docx`, and `.txt`
- PDF images require a PDF; page rendering is not provided for DOCX or TXT files.
- Scanned PDFs must contain OCR text or selectable text. The app reports files as unreadable when no text is extracted.
- Text chunks use a size of 800 characters with a 150-character overlap.
- Retrieval considers up to 8 candidate chunks, with 4 final chunks sent to generation by default.
- Narrated videos summarize at most 8 PDF pages per generation.

## Environment Variables

| Variable | Required | Purpose |
|---|---:|---|
| `OPENAI_API_KEY` | Yes | Embeddings, chat, moderation, Whisper, and TTS |
| `LANGCHAIN_TRACING_V2` | No | Enables LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | Authenticates with LangSmith |
| `LANGCHAIN_PROJECT` | No | LangSmith project name; defaults to `hybrid-rag-guarded-bot` |

## Persistent Storage

Docker Compose mounts three directories from the host:

- `./chroma_db` — vector collection and document metadata
- `./flashrank_cache` — downloaded reranker model cache
- `./media_cache` — rendered PDF pages and generated videos

Deleting the saved document library removes the ChromaDB collection and media cache from the running instance.

## Deployment Notes

The production deployment uses a Dockerized Streamlit service behind Nginx. A typical VPS deployment requires:

1. Provision Ubuntu and install Docker, Docker Compose, Nginx, and Certbot.
2. Clone the repository and create `/opt/spotify-hybrid-rag/.env`.
3. Start the service with `docker compose up -d --build`.
4. Proxy Nginx port 8501 to port 443 over TLS.
5. Enable WebSocket upgrade headers and set an upload body limit appropriate for document size.
6. Add persistent storage paths for ChromaDB and media caches.
7. Renew the TLS certificate and monitor container and Nginx logs.

Example Nginx location settings:

```nginx
location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    client_max_body_size 200m;
    proxy_read_timeout 300s;
}
```

## Security and Data Handling

- API credentials are loaded from `.env` and should not be exposed in the browser or source code.
- Email addresses, phone numbers, and API-key-shaped values are masked before moderation and model calls.
- OpenAI moderation and an additional structured safety check screen user prompts.
- Answers must use retrieved document text and have a fallback for missing information.
- Uploaded and generated files are stored locally under the configured persistent volumes unless separate infrastructure is introduced.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Karthick Balakrishnan**  
GitHub: [@Karthickbalakrishnan](https://github.com/Karthickbalakrishnan)
