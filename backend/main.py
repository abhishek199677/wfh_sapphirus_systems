import os

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from dotenv import load_dotenv

from .rag import add_document
from .agent import query_agent
from .logging_config import setup_logging

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

logger = setup_logging()
logger.info("Starting AI Copilot API")

app = FastAPI(title="AI Copilot API", description="API for Enterprise AI Copilot")


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
    from .providers import get_llm
    from .rag import get_vectorstore

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
