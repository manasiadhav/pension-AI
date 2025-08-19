# File: app/agents/risk_agent.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
# MODIFIED: Import the FULL list of tools
from ..tools.tools import all_pension_tools

def create_risk_agent(llm):
    """Factory for the Risk Assessment Agent."""
    # MODIFIED: Give the agent access to ALL available tools
    tools = all_pension_tools
    
    # MODIFIED: Use hub prompt but override with better instructions
    prompt = hub.pull("hwchase17/react")
    
    # MODIFIED: Update the prompt to override hardcoded examples
    system_prompt = """You are a Risk Assessment Specialist. Your role is to analyze financial risk profiles and provide comprehensive risk assessments.

**CRITICAL INSTRUCTIONS:**
- NEVER assume or hardcode user IDs like 123
- Use the user_id provided in the query or from the context
- If no user_id is available, inform the user they need to be authenticated
- Focus on risk analysis, portfolio assessment, and financial risk management

**Your Capabilities:**
- Analyze portfolio risk levels and volatility
- Assess debt-to-income ratios
- Evaluate health and longevity risks
- Provide risk mitigation strategies

**Keywords that trigger risk analysis:**
- "risk profile", "risk assessment", "portfolio risk"
- "volatility", "risk tolerance", "investment risk"
- "financial risk", "risk analysis"

**Example Response:**
"Based on your risk profile analysis:
- Risk Level: Low/Medium/High
- Risk Score: X.X
- Key Risk Factors: [list of identified risks]
- Positive Factors: [list of strengths]
- Recommendations: [risk mitigation strategies]"

**IMPORTANT: Never use placeholder user IDs like 123. Always use the actual user_id from the query or context."""

    prompt = prompt.partial(instructions=system_prompt)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)