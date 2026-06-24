"""

TOOL 2: Live financial metrics via yfinance.

Wraps yfinance as a LangChain Tool so the agent can pull market cap,
P/E ratio, 52-week high/low, and revenue growth for a given ticker.
yfinance pulls public data from Yahoo Finance.
"""

from langchain_core.tools import Tool
import yfinance as yf


def _format_large_number(value):
    """Convert a raw number like 2900000000000 into '$2.90T' for readability."""
    if value is None:
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value / 1e6:.2f}M"
    else:
        return f"${value:,.2f}"


def get_financial_metrics(ticker: str) -> str:
    """
    Fetch key financial metrics for a ticker and return a formatted string.

    yfinance is notorious for returning None or missing keys depending
    on the company/exchange, so every field is fetched defensively with
    .get() and a fallback. A tool that crashes on a None value breaks
    the agent's reasoning loop, which is worse than just reporting
    "N/A" for that one field and letting the LLM work with the rest.
    """
    ticker = ticker.strip().upper()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as e:
        return f"Could not retrieve financial data for '{ticker}': {e}"

    has_price = info.get("regularMarketPrice") is not None or info.get("currentPrice") is not None
    if not info or not has_price:
        return f"No financial data found for ticker '{ticker}'. Double-check the ticker symbol."

    company_name = info.get("longName", ticker)
    market_cap = _format_large_number(info.get("marketCap"))

    pe_ratio = info.get("trailingPE")
    pe_ratio_str = f"{pe_ratio:.2f}" if pe_ratio else "N/A"

    week_high = info.get("fiftyTwoWeekHigh")
    week_low = info.get("fiftyTwoWeekLow")
    week_high_str = f"${week_high:.2f}" if week_high is not None else "N/A"
    week_low_str = f"${week_low:.2f}" if week_low is not None else "N/A"

    revenue_growth = info.get("revenueGrowth")
    revenue_growth_str = f"{revenue_growth * 100:.1f}%" if revenue_growth is not None else "N/A"

    return (
        f"Financial metrics for {company_name} ({ticker}):\n"
        f"- Market Cap: {market_cap}\n"
        f"- P/E Ratio (trailing): {pe_ratio_str}\n"
        f"- 52-Week High: {week_high_str}\n"
        f"- 52-Week Low: {week_low_str}\n"
        f"- Revenue Growth (YoY): {revenue_growth_str}"
    )


financial_data_tool = Tool(
    name="financial_data",
    func=get_financial_metrics,
    description=(
        "Use this to get QUANTITATIVE FINANCIAL METRICS for a company: "
        "market cap, P/E ratio, 52-week high/low, and revenue growth. "
        "Input should be ONLY the stock ticker symbol, e.g. 'AAPL' or 'TSLA' "
        "— not the company name. "
        "Do NOT use this for news, sentiment, or qualitative information "
        "— use the web_search_news tool for that instead. "
        "Do NOT use this for information from the company's annual report "
        "— use the document_search tool for that instead."
    ),
)


if __name__ == "__main__":
    
    test_ticker = "AAPL"
    print(f"Testing yfinance with ticker: '{test_ticker}'\n")
    print(get_financial_metrics(test_ticker))