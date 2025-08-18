# File: app/agents/supervisor.py
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel
from langchain.prompts import ChatPromptTemplate

class Router(BaseModel):
    next: Literal["risk_analyst", "fraud_detector", "projection_specialist", "summarizer", "visualizer", "FINISH"]

def create_supervisor_chain(llm):
    """Factory for the Supervisor's routing logic."""
    supervisor_prompt = ChatPromptTemplate.from_template(
        """You are a supervisor of a team of expert AI agents. Your job is to route the user's request to the appropriate agent based on context and available data.

Your available agents are:
- `risk_analyst`: For questions about financial risk, volatility, and portfolio diversity.
- `fraud_detector`: For questions about suspicious transactions and fraud.
- `projection_specialist`: For questions about future pension growth and projections.
- `visualizer`: To be used when the user explicitly requests charts/visualizations OR when you have data that would benefit from visualization (like projections, risk scores, or fraud analysis).
- `summarizer`: To be used as the VERY LAST STEP to consolidate all findings and give a final, friendly answer to the user.

ROUTING LOGIC (Three-Stage Process):
1. **First Stage**: Route to the appropriate specialist agent(s) based on the user's question.
2. **Second Stage**: When control returns to you after specialist agents, analyze:
   - The original user query (does it mention charts, graphs, visualization?)
   - The available data from specialist agents
   - Whether the data would benefit from visualization
   - Route to `visualizer` if visualization is needed, otherwise to `summarizer`
3. **Third Stage**: When control returns to you after visualization, route to `summarizer` for final consolidation.

IMPORTANT: Agents NEVER communicate directly with each other. All routing goes through you (the supervisor).

EXAMPLES:
- "Show me a chart of my pension growth" → projection_specialist → visualizer → summarizer
- "What's my risk profile?" → risk_analyst → visualizer → summarizer  
- "Is this transaction fraudulent?" → fraud_detector → visualizer → summarizer
- "How does my pension grow over time? Show me a chart." → projection_specialist → visualizer → summarizer
- "How much will my pension be worth?" → projection_specialist → summarizer (no visualization needed)

User's question:
{messages}"""
    )
    return supervisor_prompt | llm.with_structured_output(Router)