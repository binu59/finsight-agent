"""
web_search_tool.py
-------------------
TOOL 1: Web search via Tavily.

Wraps Tavily's search API as a LangChain Tool so the agent can pull
recent news, sentiment, and events about a company. .
"""

from langchain_core.tools import Tool
from tavily import TavilyClient

from src.config import TAVILY_API_KEY

# One shared client 
_client = TavilyClient(api_key=TAVILY_API_KEY)


def search_company_news(query: str) -> str:
    """
    Run a Tavily search and return a readable, formatted string of the
    top results (title, snippet, source URL).

    """
    response = _client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer=False,
    )

    results = response.get("results", [])
    if not results:
        return f"No recent news found for '{query}'."

    formatted = []
    for i, r in enumerate(results, 1):
        snippet = r.get("content", "")[:300]
        formatted.append(
            f"{i}. {r.get('title', 'Untitled')}\n"
            f"   {snippet}...\n"
            f"   Source: {r.get('url', 'unknown')}"
        )

    return "\n\n".join(formatted)


# The description below is read by the LLM at decision time — it's the
# ONLY information the agent has for deciding when to call this tool
# versus the other two. Be explicit about what it's for AND what it's
# NOT for, since vague descriptions are the #1 cause of agents calling
# the wrong tool in the ReAct loop.
web_search_tool = Tool(
    name="web_search_news",
    func=search_company_news,
    description=(
        "Use this to find RECENT NEWS, sentiment, or events about a company. "
        "Input should be a search query combining the company name and what "
        "you're looking for, e.g. 'Tesla Q4 2025 earnings news' or "
        "'Apple stock recent controversy'. "
        "Do NOT use this for financial metrics like P/E ratio or market cap "
        "— use the financial_data tool for those instead. "
        "Do NOT use this for information found inside the company's own "
        "annual report — use the document_search tool for that instead."
    ),
)


if __name__ == "__main__":
    
    test_query = "Tesla recent news"
    print(f"Testing Tavily search with query: '{test_query}'\n")
    print(search_company_news(test_query))