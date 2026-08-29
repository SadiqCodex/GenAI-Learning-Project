# GenAI Learning

**Hands-on Generative AI projects spanning LangChain, LangGraph, RAG, agents, MCP, and computer vision.**

A learning-focused monorepo of Python experiments and demo apps—from basic LLM calls and embeddings to multi-agent research, travel planning with MCP, video meeting intelligence, and content-based movie recommendations.

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-FF4B4B)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-SadiqCodex-181717?logo=github)](https://github.com/SadiqCodex/GenAI-Learning-Project)

<br>

### 🚀 Live Demos

<p align="center">
  <a href="https://genai-ai-video-assistant.streamlit.app/"><img src="https://img.shields.io/badge/🎬_AI_Video_Assistant-Live-7C3AED?style=for-the-badge" alt="AI Video Assistant"></a>
  &nbsp;
  <a href="https://genai-multiagent-research-ai.streamlit.app/"><img src="https://img.shields.io/badge/🔬_Research_Agent-Live-FF8C32?style=for-the-badge" alt="Multi-Agent Research"></a>
  &nbsp;
  <a href="http://tripmates-ai.netlify.app/"><img src="https://img.shields.io/badge/✈️_TripMate_AI-Live-0EA5E9?style=for-the-badge" alt="TripMate AI"></a>
</p>

<p align="center">
  <a href="https://genai-rag-ai.streamlit.app/"><img src="https://img.shields.io/badge/📚_RAG_Book_Assistant-Live-10B981?style=for-the-badge" alt="RAG Book Assistant"></a>
  &nbsp;
  <a href="https://genai-movie-rec-ai.streamlit.app/"><img src="https://img.shields.io/badge/🎞️_Movie_Recommender-Live-E11D48?style=for-the-badge" alt="Movie Recommender"></a>
  &nbsp;
  <a href="https://genai-movie-extractorai.streamlit.app/"><img src="https://img.shields.io/badge/🍿_CineSage_Extractor-Live-F59E0B?style=for-the-badge" alt="CineSage"></a>
</p>

| App | Live URL |
| --- | --- |
| 🎬 AI Video Assistant | [genai-ai-video-assistant.streamlit.app](https://genai-ai-video-assistant.streamlit.app/) |
| 🔬 Multi-Agent Research | [genai-multiagent-research-ai.streamlit.app](https://genai-multiagent-research-ai.streamlit.app/) |
| ✈️ TripMate AI | [tripmates-ai.netlify.app](http://tripmates-ai.netlify.app/) |
| 📚 RAG Book Assistant | [genai-rag-ai.streamlit.app](https://genai-rag-ai.streamlit.app/) |
| 🎞️ Movie Recommender | [genai-movie-rec-ai.streamlit.app](https://genai-movie-rec-ai.streamlit.app/) |
| 🍿 CineSage Extractor | [genai-movie-extractorai.streamlit.app](https://genai-movie-extractorai.streamlit.app/) |

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture / Workflow](#architecture--workflow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration / Environment Variables](#configuration--environment-variables)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Live Demos (Footer)](#-try-the-live-apps)

---

## Overview

**GenAI Learning** is a personal learning repository that explores modern generative AI patterns in Python. Each folder is a self-contained experiment or demo app—not a single production product.

It addresses common GenAI learning goals:

| Goal | How this repo explores it |
| --- | --- |
| Talk to local LLMs | Ollama (`phi3`) via LangChain in chat, RAG, and agent demos |
| Ground answers in documents | PDF → chunk → embed → retrieve (FAISS / Chroma) |
| Orchestrate multi-step agents | LangGraph graphs, tool-calling agents, critic loops |
| Connect external tools | Tavily, OpenWeatherMap, TMDB, AviationStack, Sarvam STT, MCP servers |
| Ship simple UIs / APIs | Streamlit frontends and FastAPI backends |

> This is an educational codebase. Features, dependencies, and run instructions vary by subproject. Some apps expect local services (Ollama, FFmpeg, PostgreSQL) and API keys that are not bundled with the repo.

---

## Key Features

- **AI Video Assistant** — YouTube / local video → audio chunks → Whisper or Sarvam transcription → map-reduce summary → action items / decisions / questions → Chroma RAG chat
- **Multi-Agent Research System** — Search (Tavily) → scrape → write report → critic score (Ollama `phi3`), with Streamlit UI and CLI
- **TripMate AI** — LangGraph travel planner with MCP-backed flight, hotel, and weather tools; FastAPI + HTML UI; PostgreSQL checkpoints
- **College Assistant (`Agentic AI/`)** — LangGraph router over academic / fee PDFs with Groq + FAISS + HuggingFace embeddings
- **RAG Book Assistant** — Upload a PDF, build a FAISS index, ask questions with MMR retrieval and Ollama
- **Movie Recommender AI** — FastAPI (TF-IDF over ~45k movies + TMDB) and Streamlit UI (default API host on Render)
- **CineSage** — Structured movie metadata extraction with Pydantic + Ollama
- **AI Mode Chatbot** — Streamlit chatbot with five personality modes and message history
- **Learning modules** — LCEL runnables, tool calling, embeddings, retrievers, LangGraph patterns, MediaPipe hand/pose tracking

---

## Architecture / Workflow

### Repository map

```mermaid
flowchart TB
  subgraph apps [Demo Applications]
    AVA[AI-Video-Assistant]
    MARS[Multi-agent-research-system]
    TM[TripMate-AI-Using-MCP]
    RAG[rag]
    MR[movie-rec-ai]
    CS[cinesage]
    CB[chatmodels]
    AA[Agentic AI]
  end

  subgraph learn [Learning Modules]
    TOOLS[tools]
    EMB[embeddingmodels]
    CV[Computer_Vision]
  end

  LLM[(Ollama / Groq / OpenAI)]
  VS[(FAISS / Chroma)]
  APIs[(Tavily · TMDB · Weather · AviationStack · Sarvam)]

  AVA --> LLM
  AVA --> VS
  AVA --> APIs
  MARS --> LLM
  MARS --> APIs
  TM --> LLM
  TM --> APIs
  RAG --> LLM
  RAG --> VS
  AA --> LLM
  AA --> VS
  MR --> APIs
  CS --> LLM
  CB --> LLM
  TOOLS --> LLM
  TOOLS --> APIs
```

### AI Video Assistant pipeline

```mermaid
flowchart LR
  A[YouTube URL / local file] --> B[yt-dlp + pydub]
  B --> C[10-min WAV chunks]
  C --> D{Language}
  D -->|english| E[OpenAI Whisper]
  D -->|hinglish| F[Sarvam STT-Translate]
  E --> G[Transcript]
  F --> G
  G --> H[Summary · Title · Extractions]
  G --> I[Chroma + HF embeddings]
  I --> J[LCEL RAG chat]
```

### Multi-agent research pipeline

```mermaid
flowchart LR
  T[Topic] --> S[Search Agent + Tavily]
  S --> R[Reader Agent + scrape_url]
  R --> W[Writer Chain]
  W --> C[Critic Chain]
  C --> O[Report + Score X/10]
```

### TripMate LangGraph flow

```mermaid
flowchart LR
  U[User message] --> F[flight_agent]
  F --> H[hotel_agent]
  H --> W[weather_agent]
  W --> I[itinerary_agent]
  I --> X[final_agent]
  X --> Y[JSON response]
```

---

## Tech Stack

| Category | Technologies used in this repo |
| --- | --- |
| Language | Python (`pyproject.toml`: `>=3.13`; `.python-version`: `3.13`) |
| LLM frameworks | LangChain, LangGraph, LangChain Community / HuggingFace / text-splitters |
| Local LLMs | [Ollama](https://ollama.ai) — typically `phi3`, embeddings `nomic-embed-text` |
| Hosted LLMs | Groq (`llama-3.3-70b-versatile`), OpenAI (selected LangGraph demos) |
| Vector stores | FAISS (`faiss-cpu`), Chroma (`langchain-chroma`) |
| Embeddings | Ollama Embeddings, HuggingFace (`all-MiniLM-L6-v2`, `mixedbread-ai/mxbai-embed-xsmall-v1`) |
| Tools / APIs | Tavily, OpenWeatherMap, TMDB, AviationStack, Sarvam AI STT |
| MCP | `langchain-mcp-adapters`, `mcp`, Tavily MCP HTTP, `aviationstack-mcp` via `uvx`, custom weather MCP server |
| Audio / video | OpenAI Whisper, yt-dlp, pydub, FFmpeg |
| Web | Streamlit, FastAPI, Uvicorn, Jinja2 templates |
| ML / CV | scikit-learn (TF-IDF), OpenCV, MediaPipe |
| Persistence | FAISS local indexes, Chroma dirs, PostgreSQL (TripMate checkpoints via `langgraph-checkpoint-postgres`) |
| Package managers | `uv` (`pyproject.toml` / `uv.lock`) and `pip` (`requirements.txt`; TripMate has its own) |

---

## Project Structure

```text
GenAI Learning/
├── AI-Video-Assistant/           # Meeting / video intelligence (Streamlit + CLI)
│   ├── app.py
│   ├── main.py
│   ├── core/                     # transcriber, summarizer, extractor, RAG, vector store
│   └── utils/audio_processor.py
├── Multi-agent-research-system/  # Search → scrape → write → critique
│   ├── app.py                    # Streamlit UI
│   ├── pipeline.py               # CLI runner
│   ├── agents.py
│   └── tools.py
├── TripMate-AI-Using-MCP/        # Multi-agent travel planner (FastAPI + MCP)
│   ├── app.py
│   ├── backend.py
│   ├── mcp_client.py
│   ├── custom_weather_mcp_server.py
│   ├── templates/
│   ├── tools/
│   └── requirements.txt
├── Agentic AI/                   # LangGraph learning + College Assistant UI
│   ├── app.py                    # Streamlit college assistant
│   ├── sequential_base.py
│   ├── conditional_RAG.py
│   ├── parallel_reducers.py
│   ├── humanintheloop.py
│   ├── iterative_tools.py
│   └── states.py
├── rag/                          # PDF RAG (Streamlit + CLI helpers)
│   ├── app.py
│   ├── create_database.py
│   ├── main.py
│   ├── document loaders/
│   ├── retrievers/
│   └── vector store/
├── movie-rec-ai/                 # TF-IDF + TMDB recommender
│   ├── main.py                   # FastAPI
│   ├── app.py                    # Streamlit
│   ├── movies.ipynb
│   └── *.pkl                     # Precomputed TF-IDF artifacts
├── cinesage/                     # Structured movie extraction
│   ├── core.py                   # CLI (ollama chat)
│   └── UICore.py                 # Streamlit
├── chatmodels/                   # Chat experiments + mode chatbot
├── tools/                        # Tools, agents, LCEL runnables
├── embeddingmodels/              # Embedding experiments
├── Computer_Vision/              # MediaPipe hand / pose demos
├── main.py                       # Prints langchain.__version__
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── LICENSE
└── README.md
```

---

## Prerequisites

- **Python 3.13+** recommended for the root project (`pyproject.toml` / `.python-version`)
- **[Ollama](https://ollama.ai)** running locally for Ollama-based demos, with models pulled as needed:

```bash
ollama pull phi3
ollama pull nomic-embed-text
```

- **FFmpeg** on `PATH` for AI Video Assistant audio extraction
- **API keys** for the apps you run (see [Configuration](#configuration--environment-variables))
- **PostgreSQL** (and `DATABASE_URL`) for TripMate conversation checkpoints
- **`uvx`** available if you use TripMate’s AviationStack MCP (`uvx aviationstack-mcp`)
- Webcam / OpenCV for Computer Vision scripts

---

## Installation

```bash
git clone https://github.com/SadiqCodex/GenAI-Learning-Project.git
cd GenAI-Learning-Project
```

### Option A — `uv` (root project)

```bash
uv sync
```

### Option B — `pip` (root `requirements.txt`)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### TripMate-specific dependencies

```bash
pip install -r TripMate-AI-Using-MCP/requirements.txt
```

### Additional packages used by some apps

Root `pyproject.toml` / `requirements.txt` do not list every import used across subprojects. Depending on what you run, you may also need packages such as:

`langchain-ollama`, `langchain-chroma`, `langchain-groq`, `openai-whisper`, `yt-dlp`, `pydub`, `httpx`, `tavily-python`, `beautifulsoup4`

Install only what the target script imports.

Create a root `.env` (there is no `.env.example` in the repo):

```bash
# Create manually, then edit values
# Windows PowerShell:  New-Item .env
# macOS / Linux:       touch .env
```

---

## Configuration / Environment Variables

Place a `.env` in the **repository root**. Most modules call `load_dotenv()` and resolve keys from there (some also check a local `.env` beside the script).

| Variable | Used by | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | Many Ollama demos | Default `http://localhost:11434` |
| `OLLAMA_MODEL` | Research system, video cores | Default `phi3` |
| `OLLAMA_CHAT_MODEL` | RAG, tools | Default `phi3:latest` |
| `OLLAMA_EMBEDDING_MODEL` | RAG | Default `nomic-embed-text:latest` |
| `OLLAMA_TEMPERATURE` | Video / research | Float temperature |
| `WHISPER_MODEL` | AI Video Assistant | Whisper size (default `small`) |
| `SARVAM_API_KEY` | AI Video Assistant (Hinglish) | Sarvam STT-Translate |
| `SARVAM_STT_MODEL` | AI Video Assistant | Default `saaras:v2.5` |
| `TAVILY_API_KEY` | Research, tools, TripMate | Web search |
| `TMDB_API_KEY` | `movie-rec-ai` | Movie metadata / posters |
| `OPENWEATHER_API_KEY` | `tools/Agents.py`, TripMate weather MCP | Weather |
| `GROQ_API_KEY` | TripMate, Agentic AI demos | Groq chat models |
| `OPENAI_API_KEY` | Selected LangGraph scripts | e.g. human-in-the-loop writer |
| `DATABASE_URL` | TripMate | PostgreSQL for LangGraph checkpoints |
| `AVIATIONSTACK_API_KEY` | TripMate | Flight search / MCP |
| `DEFAULT_ORIGIN_IATA` | TripMate flight tool | Default origin (code default `DAC`) |
| `HUGGINGFACEHUB_API_TOKEN` | Optional HF usage | Present in local env patterns |
| `GEMINI_API_KEY` / `OPENROUTER_API_KEY` / `MISTRAL_API_KEY` | Optional / experiment | May appear in `.env`; not required for every app |

Example skeleton (replace placeholders; do not commit real secrets):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3
OLLAMA_CHAT_MODEL=phi3:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_TEMPERATURE=0.3

TAVILY_API_KEY=
TMDB_API_KEY=
OPENWEATHER_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
SARVAM_API_KEY=

DATABASE_URL=postgresql://user:password@localhost:5432/travel_db
AVIATIONSTACK_API_KEY=
DEFAULT_ORIGIN_IATA=DAC
```

`.env` is listed in `.gitignore` and should stay out of version control.

---

## Usage

Run commands from the **repository root** unless noted. Activate your virtual environment first.

### AI Video Assistant

```bash
streamlit run AI-Video-Assistant/app.py

# CLI
python AI-Video-Assistant/main.py
```

Provide a YouTube URL or local media path; choose `english` (Whisper) or `hinglish` (Sarvam).

### Multi-Agent Research System

```bash
streamlit run Multi-agent-research-system/app.py
python Multi-agent-research-system/pipeline.py
```

### TripMate AI

```bash
python TripMate-AI-Using-MCP/app.py
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). Update the weather MCP `command` / `args` paths in `TripMate-AI-Using-MCP/mcp_client.py` to your machine before relying on weather tools.

### College Assistant (Agentic AI)

```bash
streamlit run "Agentic AI/app.py"
```

Expects `academics_handbook.pdf` and `fee_structure.pdf` beside the app (both are present in `Agentic AI/`).

### RAG Book Assistant

```bash
streamlit run rag/app.py

# Pre-build FAISS from bundled sample PDF, then query via CLI
python rag/create_database.py
python rag/main.py
```

### Movie Recommender

```bash
# API (requires TMDB_API_KEY and pickle artifacts in movie-rec-ai/)
uvicorn movie-rec-ai.main:app --reload

# UI (defaults to hosted API base URL in app.py)
streamlit run movie-rec-ai/app.py
```

### CineSage

```bash
streamlit run cinesage/UICore.py
python cinesage/core.py
```

### AI Mode Chatbot

```bash
streamlit run chatmodels/UIChatbot.py
```

Modes: Angry · Funny · Sad · Happy · Sarcastic.

### Learning scripts (examples)

```bash
python chatmodels/chat.py
python embeddingmodels/embedding.py
python tools/toolcalling.py
python tools/Agents.py
python "Agentic AI/sequential_base.py"
python Computer_Vision/HandTracking.py
```

---

## API Reference

### Movie Recommender (`movie-rec-ai/main.py`)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `GET` | `/home` | TMDB feed (`category`: `trending`, `popular`, `top_rated`, `upcoming`, `now_playing`) |
| `GET` | `/tmdb/search` | TMDB keyword search |
| `GET` | `/movie/id/{tmdb_id}` | Movie details |
| `GET` | `/movie/search` | Search bundle: details + TF-IDF + genre recommendations |
| `GET` | `/recommend/tfidf` | TF-IDF similar titles |
| `GET` | `/recommend/genre` | Genre-based TMDB discovery |

Streamlit frontend currently sets `API_BASE` to `https://movie-rec-466x.onrender.com` (with a local URL as a non-effective fallback expression in code). Point it at `http://127.0.0.1:8000` when testing the API locally.

### TripMate (`TripMate-AI-Using-MCP/app.py`)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check |
| `POST` | `/api/travel` | Travel plan request |

Request body:

```json
{
  "message": "Plan a 3-day trip to Tokyo with a budget of $1200",
  "thread_id": null
}
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Plan a 3-day trip to Tokyo with a budget of $1200\"}"
```

---

## Examples

**CineSage structured schema** (`cinesage/UICore.py` / `core.py`):

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

**RAG retrieval settings** (`rag/app.py`): chunk size `1000`, overlap `200`, MMR with `k=4`, `fetch_k=10`, `lambda_mult=0.5`.

**Movie dataset artifact**: `movie-rec-ai/df.pkl` holds **45,447** rows with columns including `title`, `overview`, `genres`, `tagline`, `vote_average`, `popularity`, `tags`.

---

## Testing

There is **no automated pytest suite** in this repository. Manual / exploratory scripts include:

| File | Role |
| --- | --- |
| `AI-Video-Assistant/test.py` | Manual pipeline smoke run |
| `TripMate-AI-Using-MCP/test.py` | Manual TripMate checks |
| `TripMate-AI-Using-MCP/mcp_client_test.py` | MCP client experiment |
| `rag/document loaders/test.py` | Document-loader experiment |

`pytest` appears in `requirements.txt` but no `test_*.py` unit tests are defined for CI.

---

## Deployment

### Hosted apps

| Project | Platform | Live demo |
| --- | --- | --- |
| AI Video Assistant | Streamlit Cloud | [Open demo](https://genai-ai-video-assistant.streamlit.app/) |
| Multi-Agent Research | Streamlit Cloud | [Open demo](https://genai-multiagent-research-ai.streamlit.app/) |
| RAG Book Assistant | Streamlit Cloud | [Open demo](https://genai-rag-ai.streamlit.app/) |
| Movie Recommender | Streamlit Cloud | [Open demo](https://genai-movie-rec-ai.streamlit.app/) |
| CineSage Extractor | Streamlit Cloud | [Open demo](https://genai-movie-extractorai.streamlit.app/) |
| TripMate AI | Netlify | [Open demo](http://tripmates-ai.netlify.app/) |

### Other deployment notes

| Component | Notes present in repo |
| --- | --- |
| Movie API | `movie-rec-ai/runtime.txt` specifies `python-3.11.9` (Render-style). Streamlit UI references `https://movie-rec-466x.onrender.com`. |
| TripMate | `DATABASE_URL` error text mentions a Render PostgreSQL external URL; local run uses Uvicorn on `127.0.0.1:8000`. Frontend is also hosted at [tripmates-ai.netlify.app](http://tripmates-ai.netlify.app/). |

No Dockerfiles or CI workflows are included in this repository.

---

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `Connection refused` to Ollama | Ollama running; `OLLAMA_BASE_URL`; models pulled (`phi3`, `nomic-embed-text`) |
| `TMDB_API_KEY missing` | Set `TMDB_API_KEY` in root `.env` before importing `movie-rec-ai.main` |
| `SARVAM_API_KEY is not set` | Required for `hinglish` transcription |
| `DATABASE_URL is missing` / Groq errors | Required for TripMate (`backend.py`) |
| TripMate weather MCP fails | `mcp_client.py` currently hardcodes a machine-specific Python path and script path—update both to your local `custom_weather_mcp_server.py` |
| Audio / YouTube failures | Install FFmpeg; confirm yt-dlp / pydub imports |
| FAISS / Chroma lock on Windows | RAG app retries directory removal and may fall back to a timestamped folder |
| `ModuleNotFoundError` | Install the missing package for that subproject (root lockfile does not cover every demo) |
| Movie UI cannot reach API | Confirm API is up, or change `API_BASE` in `movie-rec-ai/app.py` to your local Uvicorn URL |

---

## Future Improvements

Ideas aligned with the current codebase gaps:

- Add a root `.env.example` documenting all keys without secrets
- Align `pyproject.toml` / `uv.lock` / `requirements.txt` with packages actually imported by each app
- Add a small pytest smoke suite for FastAPI `/health` routes and pure helpers
- Make TripMate MCP weather paths configurable via environment variables
- Add a root `static/` mount or serve TripMate CSS/JS consistently with FastAPI
- Optional Docker Compose for Ollama + PostgreSQL + selected APIs

---

## Contributing

Contributions that improve clarity, dependency hygiene, or demos are welcome.

1. Fork [SadiqCodex/GenAI-Learning-Project](https://github.com/SadiqCodex/GenAI-Learning-Project)
2. Create a feature branch
3. Keep changes scoped; do not commit `.env` or API keys
4. Open a pull request with a short description of what you ran and how you verified it

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

Copyright (c) 2025 Sadik Mohammad.

---

## Contact

**Sadik Mohammad** ([SadiqCodex](https://github.com/SadiqCodex))

- Repository: [github.com/SadiqCodex/GenAI-Learning-Project](https://github.com/SadiqCodex/GenAI-Learning-Project)
- Issues: [github.com/SadiqCodex/GenAI-Learning-Project/issues](https://github.com/SadiqCodex/GenAI-Learning-Project/issues)

---

## 🌐 Try the Live Apps

<p align="center">
  <strong>Click a badge to open the deployed demo</strong>
</p>

<p align="center">
  <a href="https://genai-ai-video-assistant.streamlit.app/"><img src="https://img.shields.io/badge/🎬_AI_Video_Assistant-Live-7C3AED?style=for-the-badge" alt="AI Video Assistant"></a>
  &nbsp;
  <a href="https://genai-multiagent-research-ai.streamlit.app/"><img src="https://img.shields.io/badge/🔬_Research_Agent-Live-FF8C32?style=for-the-badge" alt="Multi-Agent Research"></a>
  &nbsp;
  <a href="http://tripmates-ai.netlify.app/"><img src="https://img.shields.io/badge/✈️_TripMate_AI-Live-0EA5E9?style=for-the-badge" alt="TripMate AI"></a>
</p>

<p align="center">
  <a href="https://genai-rag-ai.streamlit.app/"><img src="https://img.shields.io/badge/📚_RAG_Book_Assistant-Live-10B981?style=for-the-badge" alt="RAG Book Assistant"></a>
  &nbsp;
  <a href="https://genai-movie-rec-ai.streamlit.app/"><img src="https://img.shields.io/badge/🎞️_Movie_Recommender-Live-E11D48?style=for-the-badge" alt="Movie Recommender"></a>
  &nbsp;
  <a href="https://genai-movie-extractorai.streamlit.app/"><img src="https://img.shields.io/badge/🍿_CineSage_Extractor-Live-F59E0B?style=for-the-badge" alt="CineSage"></a>
</p>

| # | Project | Live demo |
| --- | --- | --- |
| 1 | AI Video Assistant | https://genai-ai-video-assistant.streamlit.app/ |
| 2 | Multi-Agent Research System | https://genai-multiagent-research-ai.streamlit.app/ |
| 3 | TripMate AI | http://tripmates-ai.netlify.app/ |
| 4 | RAG Book Assistant | https://genai-rag-ai.streamlit.app/ |
| 5 | Movie Recommender AI | https://genai-movie-rec-ai.streamlit.app/ |
| 6 | CineSage Movie Extractor | https://genai-movie-extractorai.streamlit.app/ |

---

<p align="center">
  <em>Built as a hands-on GenAI learning lab — LangChain · LangGraph · Ollama · Streamlit · FastAPI · Whisper · FAISS · Chroma · MCP</em>
</p>
