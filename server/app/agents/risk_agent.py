# File: app/agents/risk_agent.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
# MODIFIED: Import the FULL list of tools, not just one
from ..tools.tools import all_pension_tools

def create_risk_agent(llm):
    """Factory for the Risk Analysis Agent."""
    # MODIFIED: Give the agent access to ALL available tools
    tools = all_pension_tools
    prompt = hub.pull("hwchase17/react")
    
    # MODIFIED: Update the prompt to teach the agent how to choose between tools
    system_prompt = """You are a financial risk analysis expert. Your primary goal is to use the `analyze_risk_profile` tool.

- **PRIORITY 1:** If the user asks for a 'risk analysis', 'risk profile', or 'risk assessment', you MUST use the `analyze_risk_profile` tool.
- **PRIORITY 2:** If the user asks a related definitional question (e.g., 'what is volatility?'), use the `knowledge_base_search` tool.
- Do not use the fraud or projection tools yourself; the supervisor will route those tasks if needed."""
    
    prompt = prompt.partial(instructions=system_prompt)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)