"""
test_web_search_tool.py
------------------------
Basic smoke test for Tool 1 (Tavily web search).

This is an INTEGRATION test — it makes a real API call to Tavily.
Run with: pytest tests/test_web_search_tool.py -v
"""

from src.tools.web_search_tool import search_company_news


def test_returns_nonempty_string():
    result = search_company_news("Tesla recent news")
    assert isinstance(result, str)
    assert len(result) > 0


def test_includes_a_source_url():
    result = search_company_news("Apple recent news")
    assert "Source:" in result


def test_handles_obscure_query_gracefully():
    
    result = search_company_news("asdkjfh293847 nonexistent company xyz")
    assert isinstance(result, str)