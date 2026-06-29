import logging
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .providers import get_embeddings, get_llm

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
