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
