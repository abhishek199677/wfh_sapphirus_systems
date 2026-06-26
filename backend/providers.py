import os
import logging

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
