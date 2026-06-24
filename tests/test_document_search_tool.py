"""
test_document_search_tool.py
------------------------------
Basic smoke test for Tool 3 (LlamaIndex document search).

"""

from src.tools.document_search_tool import search_annual_report


def test_returns_nonempty_string():
    result = search_annual_report("What are the main risk factors?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_finds_relevant_content_not_just_error():
    result = search_annual_report("revenue and financial performance")
    
    assert "Annual report PDF not found" not in result


def test_includes_relevance_scores():
    result = search_annual_report("risk factors")
    assert "relevance:" in result