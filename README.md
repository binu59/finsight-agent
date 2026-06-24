# FinSight Agent

An agentic financial analyst that takes a company name or ticker and produces
a structured **Investment Brief** by combining live web search, financial
market data, and a company's own annual report — using a LangChain ReAct
agent over three tools.

> Status: under active development. This README will gain an architecture
> diagram, example outputs, and a limitations section as the project nears
> completion.

## Build progress

- [x] Step 1 — Project structure & environment setup
- [ ] Step 2 — Build & test each tool individually (Tavily, yfinance, LlamaIndex)
- [ ] Step 3 — Wire tools into a ReAct AgentExecutor
- [ ] Step 4 — Add conversational memory for follow-ups
- [ ] Step 5 — Enforce structured Investment Brief output format
- [ ] Step 6 — Final README, architecture diagram, example outputs

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd finsight-agent

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# then open .env and fill in your real API keys

# 5. Verify config loads correctly
python src/config.py
```

You'll need:
- A free [Tavily](https://app.tavily.com) API key (Tool 1 — web search)
- A Gemini or Claude API key (LLM backbone — set `LLM_PROVIDER` in `.env` accordingly)
- A sample annual report PDF placed at `data/annual_reports/` (see Step 2 notes
  on sourcing one from SEC EDGAR)

## Project structure

```
finsight-agent/
├── .env.example          # Template for secrets — copy to .env, never commit .env
├── .gitignore
├── requirements.txt
├── README.md
├── main.py                # CLI entry point
├── data/
│   └── annual_reports/    # PDFs for Tool 3 (gitignored — not committed)
├── src/
│   ├── config.py          # Centralized env var loading & validation
│   ├── agent.py            # AgentExecutor assembly
│   ├── memory.py           # Conversation memory wiring
│   └── tools/
│       ├── web_search_tool.py        # Tool 1: Tavily
│       ├── financial_data_tool.py    # Tool 2: yfinance
│       └── document_search_tool.py   # Tool 3: LlamaIndex over PDF
└── tests/                 # One test file per tool, run independently
```

## Tech stack

LangChain (AgentExecutor, ReAct prompting) · Tavily Search API · yfinance ·
LlamaIndex (PDF indexing/retrieval) · Gemini / Claude · Python · python-dotenv
