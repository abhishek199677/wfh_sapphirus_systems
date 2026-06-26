from backend.agent import query_agent
from backend.rag import query_rag
from backend.providers import get_llm
import os


def test_query_agent_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answer, sources = query_agent("hello")
    assert "Configuration error" in answer


def test_get_llm_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        llm = get_llm()
        assert False, "Expected ValueError"
    except ValueError:
        pass
