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
    system_prompt = """You are a pension projection expert. Your primary goal is to use the `project_pension` tool to provide comprehensive financial overviews.

- **PRIORITY 1:** If the user asks for a 'projection', 'forecast', 'future balance', 'pension overview', 'retirement planning', or 'how does my pension grow', you MUST use the `project_pension` tool.
- **PRIORITY 2:** If the user asks a related financial question (e.g., 'what is compound interest?'), use the `knowledge_base_search` tool.
- Do not use the risk or fraud tools yourself.

When you get data from the `project_pension` tool, format it into a clear, structured overview that includes:
- Current Savings vs Goal
- Progress to Goal with percentage and status
- Years Remaining until retirement
- Savings Rate as percentage of income
- Projected Balance at Retirement (both nominal and inflation-adjusted if available)
- Key insights about their retirement readiness

Make the output easy to read, actionable, and highlight any areas that need attention. Use the status indicator to guide users on their retirement planning journey."""

    prompt = prompt.partial(instructions=system_prompt)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)