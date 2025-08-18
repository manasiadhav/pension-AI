# File: app/agents/fraud_agent.py
from langchain.agents import create_react_agent
from langchain import hub
# MODIFIED: Import the FULL list of tools
from ..tools.tools import all_pension_tools

def create_fraud_agent(llm):
    """Factory for the Fraud Detection Agent."""
    # MODIFIED: Give the agent access to ALL available tools
    tools = all_pension_tools
    prompt = hub.pull("hwchase17/react")
    
    # MODIFIED: Update the prompt to teach the agent how to choose
    system_prompt = """You are a fraud detection expert. Your primary goal is to use the `detect_fraud` tool.

- **PRIORITY 1:** If the user asks for a 'fraud check' or about 'suspicious activity', you MUST use the `detect_fraud` tool.
- **PRIORITY 2:** If the user asks a related question (e.g., 'how do I report a transaction?'), use the `knowledge_base_search` tool.
- Do not use the risk or projection tools yourself."""
    
    prompt = prompt.partial(instructions=system_prompt)
    return create_react_agent(llm, tools, prompt)