"""

Basic smoke test for Tool 2 (yfinance financial data).

Integration test — makes a real call to Yahoo Finance via yfinance.
Run with: pytest tests/test_financial_data_tool.py -v
"""

from src.tools.financial_data_tool import get_financial_metrics, _format_large_number


def test_returns_nonempty_string_for_valid_ticker():
    result = get_financial_metrics("AAPL")
    assert isinstance(result, str)
    assert "Market Cap" in result


def test_handles_lowercase_and_whitespace():
    result = get_financial_metrics("  aapl  ")
    assert "AAPL" in result


def test_handles_invalid_ticker_gracefully():
    result = get_financial_metrics("ZZZZINVALIDTICKER")
    assert isinstance(result, str)
    assert "No financial data found" in result or "Could not retrieve" in result


def test_format_large_number_trillions():
    assert _format_large_number(2_900_000_000_000) == "$2.90T"


def test_format_large_number_none():
    assert _format_large_number(None) == "N/A"