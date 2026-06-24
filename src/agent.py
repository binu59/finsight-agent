"""
agent.py
--------
Assembles all three tools into a single LangChain ReAct AgentExecutor.

Uses the classic text-based ReAct loop (Thought -> Action -> Action
Input -> Observation, repeated) rather than LangChain's newer
LangGraph-based create_agent(). The verbose Thought/Action/Observation
trace this produces is the whole point of building it this way — it
lets you SEE the agent decide which tool to call and why, step by step.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.config import get_llm
from src.memory import build_memory
from src.tools.web_search_tool import web_search_tool
from src.tools.financial_data_tool import financial_data_tool
from src.tools.document_search_tool import create_document_search_tool


# The classic ReAct prompt (Yao et al., 2022 — "ReAct: Synergizing
# Reasoning and Acting in Language Models"). Written out explicitly
# here rather than pulled from LangChain Hub, so you can see and edit
# exactly what instructions the LLM is reasoning against.
REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer

IMPORTANT: When the user is asking for an investment brief, overview,
or analysis of a company (as opposed to a narrow follow-up question
like "what's its P/E ratio?"), gather information using financial_data,
web_search_news, AND document_search before answering, then your Final
Answer MUST follow this exact structure with no extra preamble or
commentary outside it:

Final Answer:
INVESTMENT BRIEF — {{COMPANY NAME}} ({{TICKER}})

📊 Key Metrics: [market cap, P/E ratio, 52-week high/low, revenue growth]

📰 Recent News: [2-3 bullet points with sources]

📋 From Annual Report: [2-3 relevant findings from the document_search tool, or "Not covered in this request" if it wasn't needed]

⚠️ Risk Flags: [1-2 identified risks]

🔮 Outlook: [2-3 sentence synthesis]

For narrow follow-up questions that only need one piece of information
(e.g. "what's its 52-week high?" or "any recent news?"), answer
directly and concisely instead — the full brief structure above is
only for first-time / full-overview requests.

Previous conversation history (use this to resolve follow-up questions
that refer back to a company discussed earlier, e.g. "what about its
revenue growth?"):
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)


def build_agent_executor(verbose: bool = True, document_index=None) -> AgentExecutor:
    """
    Build and return the AgentExecutor with all three tools wired in.

    verbose=True prints the full Thought/Action/Observation trace to
    the console as the agent runs. Keep it on while learning/debugging;
    you can switch it off later for clean CLI output.
    """
    llm = get_llm()
    doc_tool = create_document_search_tool(document_index)
    tools = [web_search_tool, financial_data_tool, doc_tool]
    
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=build_memory(),
        verbose=verbose,
        # LLMs occasionally produce a malformed Action/Action Input
        # block. Instead of crashing the whole run, feed the parsing
        # error back to the LLM as an Observation so it can self-correct.
        handle_parsing_errors=True,
        # Safety net against the agent looping indefinitely between
        # tools without ever reaching a Final Answer.
        max_iterations=8,
    )


if __name__ == "__main__":
    # Manual standalone test — run from the project root with:
    #   python -m src.agent
    # Demonstrates memory: the second question has no company name in
    # it at all, and only works because chat_history carries it over.
    executor = build_agent_executor(verbose=True)

    print("\n=== TURN 1 ===")
    question1 = "What is Apple's current P/E ratio?"
    print(f"Question: '{question1}'\n")
    result1 = executor.invoke({"input": question1})
    print("\n--- ANSWER 1 ---")
    print(result1["output"])

    print("\n\n=== TURN 2 (follow-up, no company name mentioned) ===")
    question2 = "What about its revenue growth?"
    print(f"Question: '{question2}'\n")
    result2 = executor.invoke({"input": question2})
    print("\n--- ANSWER 2 ---")
    print(result2["output"])