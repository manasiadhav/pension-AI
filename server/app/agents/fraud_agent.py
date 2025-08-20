# File: app/agents/fraud_agent.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
# MODIFIED: Import the FULL list of tools
from ..tools.tools import all_pension_tools

def create_fraud_agent(llm):
    """Factory for the Fraud Detection Agent."""
    # MODIFIED: Give the agent access to ALL available tools
    tools = all_pension_tools
    
    # MODIFIED: Use hub prompt but override with better instructions
    prompt = hub.pull("hwchase17/react")
    
    # MODIFIED: Update the prompt to override hardcoded examples
    system_prompt = """You are a Fraud Detection Specialist. Your role is to detect fraudulent transactions and suspicious activities.

**CRITICAL INSTRUCTIONS:**
- NEVER ask for user_id - it's automatically provided by the system
- ALWAYS call the appropriate tool directly with the data you have
- Use the detect_fraud tool to check for fraudulent transactions
- Use the analyze_risk_profile tool to get risk assessments
- Use the project_pension tool to get pension projections
- Use the knowledge_base_search tool to find relevant information
- Use the analyze_uploaded_document tool to analyze user's uploaded PDF documents

**TOOL USAGE:**
- For fraud questions: Call detect_fraud() directly
- For risk questions: Call analyze_risk_profile() directly
- For pension questions: Call project_pension() directly
- For general questions: Call knowledge_base_search() directly
- For document analysis: Call analyze_uploaded_document() directly

**DOCUMENT ANALYSIS CAPABILITIES:**
- You can analyze uploaded PDF documents using analyze_uploaded_document()
- This tool searches through all documents uploaded by the user
- Perfect for answering questions like "What's in my uploaded document?" or "Based on my document, what can you tell me?"

**IMPORTANT: Never ask for user_id. The system automatically provides it."""

    prompt = prompt.partial(instructions=system_prompt)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)