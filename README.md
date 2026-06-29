# Enterprise AI Copilot

An intelligent AI assistant for company knowledge and automated workflows. Built with FastAPI, LangChain, ChromaDB, and Streamlit. Supports **OpenAI GPT** and **Anthropic Claude** via a pluggable provider abstraction.

## Architecture

```
┌─────────────┐     POST /api/ask     ┌──────────────────┐     ┌──────────────┐
│  Streamlit   │ ──────────────────►  │   FastAPI API    │────►│  LangChain   │
│  Frontend    │ ◄──────────────────  │   (backend)      │◄────│  Agent       │
│  (port 8501) │     JSON Response    │   (port 8000)    │     └──────┬───────┘
└─────────────┘                       └──────────────────┘            │
                                                              ┌───────┴────────┐
                                                              │   ChromaDB     │
                                                              │  Vector Store  │
                                                              └────────────────┘
```

## Features

- **RAG pipeline** — Query company documents stored in ChromaDB
- **AI Agent** — LangChain agent with tools for knowledge search and Jira ticket creation
- **Multi-provider** — Swap between OpenAI GPT and Anthropic Claude via config
- **Structured logging** — Timestamped logs to console and rotating files
- **Health checks** — Component-level status endpoint (`GET /health`)
- **Docker support** — Multi-stage builds with health checks and persistent volumes
- **CI/CD** — GitHub Actions for lint, test, and container build/push

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (or Anthropic API key for Claude)

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn backend.main:app --reload --port 8000

# Start frontend (separate terminal)
streamlit run frontend/app.py
```

Open http://localhost:8501

### Docker

```bash
docker compose up --build
```

Open http://localhost:8501

## Configuration

Set these in `.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes (OpenAI) | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes (Anthropic) | — | Anthropic API key |
| `LLM_PROVIDER` | No | `openai` | `openai` or `anthropic` |
| `OPENAI_MODEL` | No | `gpt-4o` | OpenAI model name |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-20250514` | Anthropic model name |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `JIRA_URL` | No | — | Jira instance URL |
| `JIRA_EMAIL` | No | — | Jira account email |
| `JIRA_API_TOKEN` | No | — | Jira API token |
| `JIRA_PROJECT_KEY` | No | `SCRUM` | Jira project key |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root status |
| `GET` | `/health` | Health check (LLM + vectorstore) |
| `POST` | `/api/ask` | Query the AI agent |
| `POST` | `/api/documents` | Ingest a document into the knowledge base |

### Example

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our leave policy?"}'
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
├── backend/
│   ├── main.py           # FastAPI server & routes
│   ├── agent.py          # LangChain agent with tools
│   ├── rag.py            # RAG pipeline (ChromaDB + LLM)
│   ├── providers.py      # LLM provider abstraction (OpenAI / Anthropic)
│   └── logging_config.py # Structured logging setup
├── frontend/
│   └── app.py            # Streamlit chat UI
├── tests/
│   ├── conftest.py       # Test configuration
│   ├── test_api.py       # API endpoint tests
│   └── test_agent.py     # Agent test
├── .github/workflows/
│   └── ci.yml            # CI/CD pipeline
├── Dockerfile            # Backend multi-stage build
├── Dockerfile.frontend   # Frontend build
├── docker-compose.yml    # Orchestration
└── requirements.txt      # Python dependencies
```
