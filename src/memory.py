"""
memory.py
---------
ConversationBufferMemory wiring for multi-turn follow-up questions.


"""

from langchain.memory import ConversationBufferMemory



def build_memory() -> ConversationBufferMemory:
    """
    Build a fresh conversation memory buffer.

    memory_key="chat_history" must match the {chat_history} placeholder
    in the ReAct prompt template in agent.py -- AgentExecutor uses this
    key to inject prior turns into each new prompt.
    """
    return ConversationBufferMemory(memory_key="chat_history")