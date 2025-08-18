# File: app/agents/pension_agent.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
# MODIFIED: Import the FULL list of tools
from ..tools.tools import all_pension_tools

def create_pension_agent(llm):
    """Factory for the Pension Projection Agent."""
    # MODIFIED: Give the agent access to ALL available tools
    tools = all_pension_tools
    prompt = hub.pull("hwchase17/react")
    
    # MODIFIED: Update the prompt to teach the agent how to choose
    system_prompt = """You are a pension projection expert. Your primary goal is to use the `project_pension` tool.

- **PRIORITY 1:** If the user asks for a 'projection', 'forecast', or 'future balance', you MUST use the `project_pension` tool.
- **PRIORITY 2:** If the user asks a related financial question (e.g., 'what is compound interest?'), use the `knowledge_base_search` tool.
- Do not use the risk or fraud tools yourself."""

    prompt = prompt.partial(instructions=system_prompt)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)