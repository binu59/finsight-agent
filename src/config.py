"""
config.py
---------
Single source of truth for environment variables and secrets.

Every other module should import settings FROM HERE rather than calling
os.getenv() directly — that way there's exactly one place that knows
about .env, and exactly one place that validates required keys exist.
"""

import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load variables from a local .env file into the process environment.

load_dotenv()


def _require(var_name: str) -> str:
    
   
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {var_name}\n"
            f"Did you copy .env.example to .env and fill it in?"
        )
    return value


# Tavily web search (Tool 1)
TAVILY_API_KEY = _require("TAVILY_API_KEY")

#  LLM backbone selection 
# "gemini" or "claude" 
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()


if LLM_PROVIDER == "gemini":
    GOOGLE_API_KEY = _require("GOOGLE_API_KEY")
elif LLM_PROVIDER == "claude":
    ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
else:
    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'gemini' or 'claude'."
    )

# Document search (Tool 3) 
ANNUAL_REPORT_PATH = os.getenv(
    "ANNUAL_REPORT_PATH", "data/annual_reports/sample_10k.pdf"
)


def get_llm() -> BaseChatModel:
    """
    Initializes and returns the language model based on the provider.
    """
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )

    elif LLM_PROVIDER == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0,
        )
    else:
        print("LLM is explicitly disabled. Using MockLLM.")
        return None


if __name__ == "__main__":
    
    print(f"LLM_PROVIDER = {LLM_PROVIDER}")
    print(f"TAVILY_API_KEY loaded: {'yes' if TAVILY_API_KEY else 'no'}")
    print(f"ANNUAL_REPORT_PATH = {ANNUAL_REPORT_PATH}")
    print("Config loaded successfully.")
