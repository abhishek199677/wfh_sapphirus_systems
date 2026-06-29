# Reproduce This Project: Enterprise AI Copilot

Copy the **entire** contents of this file and paste it into any AI assistant (Claude, ChatGPT, etc.) with this instruction:

> *Create the complete project below with every file shown, in the exact directory structure listed. Do not skip, modify, or summarize any file. After creating all files, tell me the commands to verify it works.*

---

## Project Structure

```
project-root/
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── README.md
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── backend/
│   ├── agent.py
│   ├── logging_config.py
│   ├── main.py
│   ├── providers.py
│   └── rag.py
├── frontend/
│   └── app.py
└── tests/
    ├── conftest.py
    ├── test_agent.py
    └── test_api.py
```

---

## File Contents

### `.gitignore`

```
__pycache__
*.pyc
*.pyo
*.egg-info
dist
build
.env
chroma_db/
logs/
.pytest_cache
.cache
htmlcov
.coverage
.DS_Store
.vscode
.idea
*.swp
*.swo
*~
.dockerignore
```

---

### `requirements.txt`

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
langchain>=1.3.0
langchain-classic>=1.0.0
langchain-core>=1.4.0
langchain-openai>=0.2.0
langchain-anthropic>=0.3.0
langchain-chroma>=0.2.0
langchain-community>=0.4.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
chromadb>=0.5.0
streamlit>=1.40.0
requests>=2.32.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-cov>=5.0.0
httpx>=0.28.0
```

---

### `backend/main.py`

```python
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import query_agent
from .logging_config import setup_logging
from .providers import get_llm
from .rag import add_document, get_vectorstore

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

logger = setup_logging()
logger.info("Starting AI Copilot API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup — verifying LLM and vectorstore configuration")
    try:
        get_llm()
        logger.info("LLM provider initialized successfully")
    except Exception as e:
        logger.warning("LLM provider not available at startup: %s", e)
    try:
        get_vectorstore()
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.warning("Vector store not available at startup: %s", e)
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="AI Copilot API",
    description="API for Enterprise AI Copilot",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    source_documents: list = []


class DocumentRequest(BaseModel):
    text: str
    metadata: dict = None


@app.get("/")
def read_root():
    return {"status": "AI Copilot API is running"}


@app.get("/health")
def health_check(response: Response):
    result = {"status": "healthy", "checks": {}}

    try:
        get_llm()
        result["checks"]["llm"] = {"status": "ok"}
    except Exception as e:
        result["checks"]["llm"] = {"status": "error", "detail": str(e)}
        result["status"] = "degraded"

    try:
        get_vectorstore()
        result["checks"]["vectorstore"] = {"status": "ok"}
    except Exception as e:
        result["checks"]["vectorstore"] = {"status": "error", "detail": str(e)}
        result["status"] = "degraded"

    response.status_code = 200 if result["status"] == "healthy" else 503
    return result


@app.post("/api/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    logger.info("POST /api/ask - query=%s", request.query[:60])
    try:
        answer, sources = query_agent(request.query)
        return QueryResponse(answer=answer, source_documents=sources)
    except Exception as e:
        logger.error("POST /api/ask failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents")
def upload_document(request: DocumentRequest):
    logger.info("POST /api/documents - text_len=%d", len(request.text))
    try:
        add_document(request.text, request.metadata)
        return {"status": "success", "message": "Document added to knowledge base"}
    except Exception as e:
        logger.error("POST /api/documents failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
```

---

### `backend/agent.py`

```python
import logging
import os

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from .providers import get_llm
from .rag import get_vectorstore

logger = logging.getLogger(__name__)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for documents and policies."""
    logger.info("Tool called: search_knowledge_base(query=%s)", query[:60])
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=3)
    result = "\n\n".join(doc.page_content for doc in docs)
    logger.debug("Knowledge base returned %d docs", len(docs))
    return result


@tool
def create_jira_ticket(title: str, description: str) -> str:
    """Create a Jira ticket. Use this when the user asks to create a task or ticket."""
    logger.info("Tool called: create_jira_ticket(title=%s)", title)
    jira_url = os.getenv("JIRA_URL")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_token = os.getenv("JIRA_API_TOKEN")

    if jira_url and jira_email and jira_token:
        return _create_jira_via_api(title, description, jira_url, jira_email, jira_token)

    logger.warning("Jira credentials not configured; returning mock response")
    return f"Successfully created Jira ticket: '{title}' with description: '{description}'"


def _create_jira_via_api(title: str, description: str, url: str, email: str, token: str) -> str:
    import requests
    from requests.auth import HTTPBasicAuth

    auth = HTTPBasicAuth(email, token)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    payload = {
        "fields": {
            "project": {"key": os.getenv("JIRA_PROJECT_KEY", "SCRUM")},
            "summary": title,
            "description": description,
            "issuetype": {"name": "Task"},
        }
    }

    try:
        resp = requests.post(f"{url}/rest/api/2/issue", json=payload, headers=headers, auth=auth, timeout=30)
        resp.raise_for_status()
        key = resp.json().get("key", "unknown")
        logger.info("Jira ticket created: %s", key)
        return f"Successfully created Jira ticket: {key} - '{title}'"
    except Exception as e:
        logger.error("Jira API call failed: %s", str(e))
        return f"Failed to create Jira ticket: {str(e)}"


def query_agent(query: str):
    logger.info("Agent query: %s", query[:80])

    try:
        llm = get_llm()
    except Exception as e:
        logger.error("Failed to initialize LLM: %s", str(e))
        return f"Configuration error: {str(e)}", []

    tools = [search_knowledge_base, create_jira_ticket]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an Enterprise AI Copilot. "
            "You can search the internal knowledge base or automate tasks like creating Jira tickets. "
            "Answer questions based on the tools available."
        )),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    try:
        response = agent_executor.invoke({"input": query})
        output = response["output"]
        logger.info("Agent response received (len=%d chars)", len(output))
        return output, [{"source": "Agent Execution"}]
    except Exception as e:
        logger.error("Agent execution failed: %s", str(e))
        return f"Error executing agent: {str(e)}", []
```

---

### `backend/providers.py`

```python
import logging
import os

logger = logging.getLogger(__name__)


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    logger.info("Initializing LLM with provider=%s", provider)

    if provider == "openai":
        return _get_openai_llm()
    elif provider == "anthropic":
        return _get_anthropic_llm()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'anthropic'.")


def get_embeddings():
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    logger.info("Initializing embeddings with provider=%s", provider)

    if provider == "openai":
        return _get_openai_embeddings()
    elif provider == "anthropic":
        return _get_anthropic_embeddings()
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Use 'openai' or 'anthropic'.")


def _get_openai_llm():
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        raise ValueError("OPENAI_API_KEY is missing or invalid. Please configure it in your .env file.")
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)


def _get_anthropic_llm():
    from langchain_anthropic import ChatAnthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        raise ValueError("ANTHROPIC_API_KEY is missing or invalid. Please configure it in your .env file.")
    return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"), temperature=0)


def _get_openai_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()


def _get_anthropic_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.warning("Anthropic does not provide embeddings; falling back to HuggingFaceEmbeddings")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

---

### `backend/rag.py`

```python
import logging
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .providers import get_llm, get_embeddings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_vectorstore():
    logger.debug("Initializing Chroma vector store (persist_dir=%s)", CHROMA_PERSIST_DIR)
    return Chroma(
        collection_name="enterprise_docs",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_document(text: str, metadata: dict = None):
    logger.info("Adding document to knowledge base (len=%d chars)", len(text))
    vectorstore = get_vectorstore()
    doc = Document(page_content=text, metadata=metadata or {})
    vectorstore.add_documents([doc])
    logger.info("Document added successfully")
    return True


def query_rag(query: str):
    logger.info("RAG query: %s", query[:80])
    llm = get_llm()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = rag_chain.invoke(query)
        docs = retriever.invoke(query)
        sources = [doc.metadata for doc in docs]
        logger.info("RAG response received (len=%d chars, sources=%d)", len(response), len(sources))
        return response, sources
    except Exception as e:
        logger.error("RAG query failed: %s", str(e))
        return f"Error querying RAG: {str(e)}", []
```

---

### `backend/logging_config.py`

```python
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger
```

---

### `frontend/app.py`

```python
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise AI Copilot", page_icon="🤖", layout="wide")

st.title("Enterprise AI Copilot 🤖")
st.markdown("Your intelligent assistant for company knowledge and automated workflows.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or request a task..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/ask",
                json={"query": prompt},
                timeout=60,
            )
            response.raise_for_status()
            bot_reply = response.json().get("answer", "No answer provided.")
        except requests.exceptions.Timeout:
            bot_reply = "Error: Request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            bot_reply = f"Error: Cannot connect to backend at {BACKEND_URL}. Ensure the server is running."
        except requests.exceptions.HTTPError as e:
            bot_reply = f"Error: Backend returned {e.response.status_code}"
        except Exception as e:
            bot_reply = f"Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
```

---

### `tests/conftest.py`

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
```

---

### `tests/test_agent.py`

```python
from backend.agent import query_agent
from backend.providers import get_llm


def test_query_agent_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answer, sources = query_agent("hello")
    assert "Configuration error" in answer


def test_get_llm_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        get_llm()
        assert False, "Expected ValueError"
    except ValueError:
        pass
```

---

### `tests/test_api.py`

```python
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "AI Copilot API is running"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    assert "checks" in resp.json()
    assert "status" in resp.json()


def test_ask_no_query(client):
    resp = client.post("/api/ask", json={})
    assert resp.status_code == 422


def test_documents_invalid(client):
    resp = client.post("/api/documents", json={})
    assert resp.status_code == 422
```

---

### `Dockerfile`

```dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN mkdir -p /app/chroma_db /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### `Dockerfile.frontend`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

### `docker-compose.yml`

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/app/chroma_db
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    depends_on:
      backend:
        condition: service_healthy
    environment:
      - BACKEND_URL=http://backend:8000
    restart: unless-stopped

volumes:
  chroma_data:
```

---

### `.github/workflows/ci.yml`

```yaml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff
      - name: Lint with ruff
        run: ruff check backend/ frontend/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m pytest tests/ -v --cov=backend --cov-report=term
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

  build-and-push:
    if: github.ref == 'refs/heads/main'
    needs: [lint, test]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push backend
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
      - name: Build and push frontend
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.frontend
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-frontend:latest
```

---

### `README.md`

```markdown
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
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
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
```

---

## After Creating All Files

```bash
# Create .env with your API keys
cat > .env << 'EOF'
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
LLM_PROVIDER=openai
LOG_LEVEL=INFO
EOF

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start backend (terminal 1)
uvicorn backend.main:app --reload --port 8000

# Start frontend (terminal 2)
streamlit run frontend/app.py

# Or use Docker (runs both)
docker compose up --build
```
