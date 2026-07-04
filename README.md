# GenAI Learning — LangChain & LLM Projects

A hands-on collection of **Generative AI** projects and experiments built with LangChain, Ollama (local LLMs), Streamlit, and various AI APIs. This repository covers the full spectrum of modern GenAI development — from basic LLM calls and embeddings to multi-agent pipelines, RAG systems, and production-ready AI applications.

---

## Table of Contents
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Projects](#-projects)
  - [AI Video Assistant](#1-ai-video-assistant)
  - [Multi-Agent Research System](#2-multi-agent-research-system)
  - [RAG Book Assistant](#3-rag-book-assistant)
  - [Movie Recommender AI](#4-movie-recommender-ai)
  - [CineSage — Movie Info Extractor](#5-cinesage--movie-info-extractor)
  - [AI Mode Chatbot](#6-ai-mode-chatbot)
- [Learning Modules](#-learning-modules)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Author](#-author)

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| LLM Framework | LangChain, LangChain-Ollama |
| Local LLMs | Ollama (phi3, nomic-embed-text) |
| External APIs | Tavily Search, Sarvam AI (STT), TMDB, OpenWeatherMap |
| Vector Stores | FAISS, ChromaDB |
| Embeddings | Ollama Embeddings, HuggingFace (all-MiniLM-L6-v2) |
| Audio/Video | Whisper (OpenAI), yt-dlp, pydub, FFmpeg |
| Web Framework | Streamlit, FastAPI |
| Deployment | Render (movie-rec API) |
| Language | Python 3.11+ |

---

## 📁 Project Structure

```
GenAI Learning/
├── AI-Video-Assistant/          # YouTube/video transcription + RAG chat
│   ├── core/
│   │   ├── extractor.py         # Action items, decisions, questions extraction
│   │   ├── rag_engine.py        # LCEL RAG pipeline over transcript
│   │   ├── summarizer.py        # Map-reduce summarization
│   │   ├── transcriber.py       # Whisper + Sarvam AI (Hinglish support)
│   │   └── vector_store.py      # ChromaDB + HuggingFace embeddings
│   ├── utils/
│   │   └── audio_processor.py   # yt-dlp download, WAV conversion, chunking
│   └── app.py                   # Streamlit UI
│
├── Multi-agent-research-system/ # 4-agent research pipeline
│   ├── agents.py                # Search agent, Reader agent, Writer & Critic chains
│   ├── pipeline.py              # CLI pipeline runner
│   ├── tools.py                 # Tavily web search + BeautifulSoup scraper tools
│   └── app.py                   # Streamlit UI with live pipeline status
│
├── rag/                         # RAG system with PDF ingestion
│   ├── app.py                   # Streamlit UI (upload PDF → ask questions)
│   ├── create_database.py       # PDF → FAISS vector store builder
│   ├── main.py                  # CLI RAG query interface
│   ├── document loaders/        # PDF, web page, text loaders (experiments)
│   ├── retrievers/              # MMR, multi-query, arXiv retriever experiments
│   └── vector store/            # ChromaDB experiments
│
├── movie-rec-ai/                # Movie recommendation system
│   ├── main.py                  # FastAPI backend (TF-IDF + TMDB API)
│   ├── app.py                   # Streamlit frontend
│   └── movies.ipynb             # Data preprocessing & TF-IDF model building
│
├── cinesage/                    # Movie info extractor using structured output
│   ├── core.py                  # CLI version (Pydantic + Ollama)
│   └── UICore.py                # Streamlit UI version
│
├── chatmodels/                  # LLM chat experiments
│   ├── chat.py                  # Basic LLM invocation (Ollama/OpenAI/Groq/Gemini)
│   ├── chatbot.py               # CLI chatbot with personality modes + memory
│   └── UIChatbot.py             # Streamlit chatbot with 5 AI personality modes
│
├── tools/                       # LangChain tools & agents experiments
│   ├── toolcalling.py           # Tool binding + manual tool call loop
│   ├── owntool.py               # Custom @tool decorator example
│   ├── Agents.py                # City agent (weather + news) with human approval
│   ├── newssummarizer.py        # Tavily search → LLM summarization chain
│   ├── sequencerunnable.py      # LCEL RunnableSequence experiments
│   ├── parallelrunnable.py      # LCEL RunnableParallel experiments
│   └── runnablepassthrough.py   # RunnablePassthrough experiments
│
├── embeddingmodels/             # Embedding model experiments
│   ├── embedding.py             # Ollama embeddings (nomic-embed-text)
│   └── huggingfaceembedding.py  # HuggingFace embeddings
│
├── .env                         # API keys (not committed)
├── requirements.txt             # All Python dependencies
└── pyproject.toml               # Project config (uv)
```

---

## 🚀 Projects

### 1. AI Video Assistant

> `AI-Video-Assistant/`

A full-stack meeting intelligence tool that takes any YouTube URL or local video file and produces a complete analysis.

**Pipeline:**
1. Audio extraction via `yt-dlp` + `pydub` (chunked into 10-min pieces)
2. Transcription — English via **OpenAI Whisper** (local), Hinglish via **Sarvam AI** STT-Translate API
3. Map-reduce **summarization** using Ollama phi3
4. Extraction of **action items**, **key decisions**, and **open questions**
5. **RAG chat** over the transcript using ChromaDB + HuggingFace embeddings + LCEL pipeline

**Key Features:**
- Supports both YouTube URLs and local video/audio files
- Bilingual: English + Hinglish (auto-translated to English via Sarvam)
- Live pipeline status in sidebar with animated step indicators
- Custom dark-themed Streamlit UI (JetBrains Mono + Syne fonts)
- In-session RAG chat with full conversation history

**Run:**
```bash
streamlit run AI-Video-Assistant/app.py
```

---

### 2. Multi-Agent Research System

> `Multi-agent-research-system/`

A 4-agent collaborative pipeline that researches any topic and produces a scored, structured report.

**Agents:**
| Agent | Role | Tool |
|---|---|---|
| Search Agent | Finds recent web information | Tavily Search API |
| Reader Agent | Scrapes top URLs for deep content | BeautifulSoup scraper |
| Writer Chain | Drafts structured research report | Ollama phi3 |
| Critic Chain | Reviews and scores the report (X/10) | Ollama phi3 |

**Key Features:**
- Real-time pipeline status cards (Waiting → Running → Done)
- Final report downloadable as `.md`
- Critic feedback with score, strengths, and improvement areas
- Custom dark UI with orange accent theme

**Run:**
```bash
# Streamlit UI
streamlit run Multi-agent-research-system/app.py

# CLI
python Multi-agent-research-system/pipeline.py
```

---

### 3. RAG Book Assistant

> `rag/`

Upload any PDF and ask questions — answers grounded strictly in the document.

**Architecture:**
- PDF ingestion via `PyPDFLoader`
- Chunking with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
- Embeddings via `OllamaEmbeddings` (nomic-embed-text)
- Vector store: **FAISS** (persisted locally)
- Retrieval: **MMR** (Maximal Marginal Relevance) — `k=4, fetch_k=10`
- LLM: Ollama phi3 with strict context-only prompt

**Also includes:**
- `create_database.py` — standalone script to pre-build FAISS index from a PDF
- `main.py` — CLI query interface for the saved vector store
- Experiments in `retrievers/` — multi-query retriever, arXiv retriever, MMR tuning

**Run:**
```bash
streamlit run rag/app.py
```

---

### 4. Movie Recommender AI

> `movie-rec-ai/`

A full-stack movie recommendation system with a FastAPI backend deployed on Render and a Streamlit frontend.

**Backend (FastAPI):**
- TF-IDF cosine similarity on movie metadata (local dataset, ~45k movies)
- TMDB API integration for posters, details, genre-based discovery
- Endpoints: `/home`, `/tmdb/search`, `/movie/id/{id}`, `/movie/search`, `/recommend/genre`
- Deployed at: `https://movie-rec-466x.onrender.com`

**Frontend (Streamlit):**
- Keyword search with autocomplete dropdown
- Poster grid with click-to-details navigation
- Movie details page: poster, overview, genres, backdrop
- Dual recommendations: TF-IDF similar movies + genre-based discovery

**Run:**
```bash
# Backend
uvicorn movie-rec-ai.main:app --reload

# Frontend
streamlit run movie-rec-ai/app.py
```

---

### 5. CineSage — Movie Info Extractor

> `cinesage/`

Paste any movie description paragraph and get structured JSON output using **Pydantic structured output** with Ollama.

**Output Schema:**
```python
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str
```

**Run:**
```bash
streamlit run cinesage/UICore.py
```

---

### 6. AI Mode Chatbot

> `chatmodels/`

A conversational chatbot with 5 distinct AI personality modes, built with full message history.

**Modes:** 😠 Angry · 😂 Funny · 😢 Sad · 😊 Happy · 🙄 Sarcastic

**Key Features:**
- Persistent conversation memory using `SystemMessage + HumanMessage + AIMessage`
- Mode switching resets chat context automatically
- Message count stats in sidebar
- Supports Ollama, OpenAI, Groq, Gemini (commented stubs included)

**Run:**
```bash
streamlit run chatmodels/UIChatbot.py
```

---

## 📚 Learning Modules

These modules document the learning journey through core LangChain concepts:

| Module | What's Covered |
|---|---|
| `chatmodels/chat.py` | Basic LLM invocation across providers (Ollama, OpenAI, Groq, Gemini) |
| `embeddingmodels/embedding.py` | Generating embeddings with Ollama (nomic-embed-text) |
| `embeddingmodels/huggingfaceembedding.py` | HuggingFace sentence-transformers embeddings |
| `tools/owntool.py` | Creating custom LangChain tools with `@tool` decorator |
| `tools/toolcalling.py` | Manual tool binding + tool call loop with `bind_tools` |
| `tools/Agents.py` | ReAct agent with weather + news tools + human approval middleware |
| `tools/newssummarizer.py` | Tavily search → LCEL summarization chain |
| `tools/sequencerunnable.py` | `RunnableSequence` (LCEL pipe operator) |
| `tools/parallelrunnable.py` | `RunnableParallel` for concurrent chain execution |
| `tools/runnablepassthrough.py` | `RunnablePassthrough` for context injection |
| `rag/retrievers/mmr.py` | MMR retrieval tuning |
| `rag/retrievers/multiquery.py` | Multi-query retriever for better recall |
| `rag/retrievers/arixv.py` | ArXiv academic paper retriever |

---

## 📋 Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally
- Required Ollama models pulled:
  ```bash
  ollama pull phi3
  ollama pull nomic-embed-text
  ```
- FFmpeg installed (for audio processing in AI Video Assistant)

---

## 🛠 Installation

```bash
# Clone the repository
git clone <repository-url>
cd "GenAI Learning"

# Install dependencies (using uv — recommended)
uv sync

# Or using pip
pip install -r requirements.txt

# Copy and fill environment variables
cp .env.example .env
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3
OLLAMA_CHAT_MODEL=phi3:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_TEMPERATURE=0.3

# External APIs
TAVILY_API_KEY=<your-tavily-api-key>
TMDB_API_KEY=<your-tmdb-api-key>
OPENWEATHER_API_KEY=<your-openweather-api-key>
SARVAM_API_KEY=<your-sarvam-api-key>        # For Hinglish transcription
```

---

## 👤 Author

**Sadik Mohammad**

- GitHub: [Your GitHub Profile]
- Email: [Your Email]

---

*Built with LangChain · Ollama · Streamlit · FastAPI · Whisper · FAISS · ChromaDB*
