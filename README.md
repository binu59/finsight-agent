# FinSight Agent 📊

An agentic AI financial analyst that takes a company name or ticker and produces a structured **Investment Brief** by autonomously reasoning across three data sources: live web search, real-time market data, and an uploaded annual report PDF.

Built on a LangChain ReAct agent loop — the agent decides *which tool to call and why* at each step, visible as a live Thought → Action → Observation trace in the UI.

---

## Demo

**Query:** `Investment brief for Apple`

**Output:**
```
INVESTMENT BRIEF — Apple Inc. (AAPL)

📊 Key Metrics: Market Cap: $4.38T, P/E Ratio (trailing): 36.08,
   52-Week High: $317.40, 52-Week Low: $196.86, Revenue Growth (YoY): 16.6%

📰 Recent News:
• Apple is making strategic moves in the chip sector and facing rising memory
  costs, impacting partnerships and market dynamics. (Source: Yahoo Finance)
• Apple has unveiled new features and intelligence experiences across its
  services. (Source: Apple Newsroom)
• Intel shares leap after Trump announces chip deal with Apple.
  (Source: CBS News)

📋 From Annual Report:
• Apple faces risks from varied stakeholder expectations regarding social and
  environmental issues, which can lead to liabilities and reputational harm.
• Total net sales for fiscal year ended September 27, 2025 reached $416.16B,
  with Services growing to $109.16B and Products at $307.00B.

⚠️ Risk Flags:
• Exposure to regulatory and stakeholder expectations on ESG issues.
• Supply chain concentration risk cited repeatedly in risk factors section.

🔮 Outlook: Apple continues to demonstrate strong financial performance with
significant market capitalisation and 16.6% revenue growth. The company is
actively innovating in chip development and expanding its services ecosystem.
Strategic focus on AI features and manufacturing partnerships suggests continued
market leadership, though regulatory scrutiny remains a watchpoint.
```

The UI also shows an expandable **Agent Reasoning** section revealing the full Thought/Action/Observation trace for every response.

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────┐
│         LangChain ReAct AgentExecutor   │
│  (Thought → Action → Observation loop)  │
│         + ConversationBufferMemory      │
└──────────┬──────────┬──────────┬────────┘
           │          │          │
           ▼          ▼          ▼
    ┌──────────┐ ┌─────────┐ ┌──────────────┐
    │  Tavily  │ │yfinance │ │  LlamaIndex  │
    │  Search  │ │   API   │ │  PDF Search  │
    │ (Tool 1) │ │(Tool 2) │ │  (Tool 3)   │
    └──────────┘ └─────────┘ └──────────────┘
    Recent news   Market cap   Annual report
    & sentiment   P/E, 52W     (uploaded PDF,
                  high/low,    persisted index)
                  revenue
                  growth
           │          │          │
           └──────────┴──────────┘
                      │
                      ▼
            Structured Investment Brief
            (enforced via prompt engineering)
                      │
                      ▼
         ┌────────────────────────┐
         │   Streamlit Chat UI    │
         │  + reasoning trace     │
         │  + PDF uploader        │
         └────────────────────────┘
```

**Key design decisions:**

- **Classic ReAct AgentExecutor** (LangChain 0.3.30, pinned) rather than LangGraph's newer `create_agent()` — the explicit Thought/Action/Observation trace is pedagogically and demonstrably valuable, and LangChain 1.x removed this API in a breaking change
- **Tool descriptions** are written with explicit "use this for X, NOT for Y" language — the most common cause of agents calling the wrong tool is vague descriptions
- **Tool 3 is retrieval-only** — LlamaIndex returns raw relevant chunks; the agent's own LLM (Gemini/Claude) synthesises them, avoiding a hidden second LLM call inside the tool
- **Local HuggingFace embeddings** (BAAI/bge-small-en-v1.5) for document indexing — free, no extra API key, runs on CPU; index is persisted to disk so it's only built once per PDF
- **`handle_parsing_errors=True`** on the AgentExecutor — LLMs occasionally output malformed Action blocks; this feeds the parse error back as an Observation so the LLM self-corrects rather than crashing

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangChain 0.3.30 (ReAct AgentExecutor) |
| LLM backbone | Google Gemini 2.5 Flash Lite (swappable to Claude via config) |
| Tool 1 — Web search | Tavily Search API |
| Tool 2 — Financial data | yfinance |
| Tool 3 — Document search | LlamaIndex + HuggingFace embeddings |
| Memory | LangChain ConversationBufferMemory |
| Frontend | Streamlit + StreamlitCallbackHandler |
| Secrets | python-dotenv |
| Testing | pytest (integration tests per tool) |

---

## Project Structure

```
finsight-agent/
├── .env.example              # Template for secrets — copy to .env, never commit .env
├── .gitignore
├── requirements.txt          # Pinned dependencies (see note on LangChain versioning)
├── README.md
├── app.py                    # Streamlit chat UI (entry point for the web app)
├── main.py                   # CLI entry point (multi-turn conversation loop)
├── data/
│   └── annual_reports/       # Drop your PDF here (gitignored — not committed)
├── src/
│   ├── config.py             # Centralised env var loading & LLM provider selection
│   ├── agent.py              # AgentExecutor assembly + ReAct prompt
│   ├── memory.py             # ConversationBufferMemory wiring
│   └── tools/
│       ├── web_search_tool.py        # Tool 1: Tavily
│       ├── financial_data_tool.py    # Tool 2: yfinance
│       └── document_search_tool.py   # Tool 3: LlamaIndex over PDF
└── tests/
    ├── test_web_search_tool.py
    ├── test_financial_data_tool.py
    └── test_document_search_tool.py
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/binu59/finsight-agent.git
cd finsight-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on LangChain versioning:** dependencies are pinned to `langchain==0.3.30` deliberately. LangChain 1.x removed `AgentExecutor` and `create_react_agent` in a breaking API change (migrating to LangGraph). This project uses the pre-1.0 API to preserve the explicit ReAct trace. Do not upgrade LangChain without reading the migration guide.

### 4. Get your API keys

| Key | Where to get it | Required? |
|---|---|---|
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | Yes |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com)  | Yes (if using Gemini) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com)  | Only if switching to Claude |

### 5. Configure secrets

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
TAVILY_API_KEY=your_real_tavily_key
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_real_gemini_key
ANNUAL_REPORT_PATH=data/annual_reports/sample_10k.pdf
```

### 6. Add an annual report PDF

Download any company's 10-K filing from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany):
1. Search for your company (e.g. `AAPL`, `TSLA`)
2. Filter by form type `10-K`
3. Open the most recent filing → open the main `.htm` document
4. Print → Save as PDF
5. Save to `data/annual_reports/` and update `ANNUAL_REPORT_PATH` in `.env`

### 7. Verify setup

```bash
python src/config.py
# Should print: Config loaded successfully.
```

### 8. Run

**Streamlit web app (recommended):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py
```

---

## Running Tests

Each tool has its own integration test that makes real API calls:

```bash
# Test all tools
pytest tests/ -v

# Test individually
pytest tests/test_web_search_tool.py -v
pytest tests/test_financial_data_tool.py -v
pytest tests/test_document_search_tool.py -v
```

> Tests make real API/network calls to Tavily, Yahoo Finance, and HuggingFace. First run of `test_document_search_tool.py` will be slow (~30-60s) as it downloads the embedding model and builds the vector index.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| Single document per session | Tool 3 indexes one PDF at a time. Uploading a new PDF rebuilds the index for that session only |
| Free-tier rate limits | Gemini 2.5 Flash Lite has a daily request cap. Each agent run consumes several LLM calls (one per ReAct step). If you hit the limit, wait for the daily reset or switch `LLM_PROVIDER=claude` |
| yfinance data availability | Some financial metrics (e.g. `revenueGrowth`) return `N/A` for certain tickers depending on Yahoo Finance's data availability |
| ReAct formatting sensitivity | Smaller or less instruction-following LLMs may occasionally produce malformed `Action:`/`Action Input:` blocks. `handle_parsing_errors=True` handles self-correction, but a very small model may loop |
| No persistent conversation history | Memory resets when the app restarts. Chat history is session-only |

---

## Future Work

- [ ] Add Groq-hosted Llama as a third LLM provider option (removes Gemini daily quota constraint entirely)
- [ ] Multi-document indexing with per-company PDF routing
- [ ] Response caching to reduce redundant API calls for repeated queries
- [ ] Deploy to Streamlit Community Cloud

---

## License

MIT
