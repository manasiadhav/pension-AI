# File: app/agents/supervisor.py
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel
from langchain.prompts import ChatPromptTemplate

class Router(BaseModel):
    next: Literal["risk_analyst", "fraud_detector", "projection_specialist", "summarizer", "FINISH"]

def create_supervisor_chain(llm):
    """Factory for the Supervisor's routing logic."""
    supervisor_prompt = ChatPromptTemplate.from_template(
        """You are a supervisor of a team of expert AI agents. Your job is to route the user's request to the appropriate agent.

Your available agents are:
- `risk_analyst`: For questions about financial risk, volatility, and portfolio diversity.
- `fraud_detector`: For questions about suspicious transactions and fraud.
- `projection_specialist`: For questions about future pension growth and projections.
- `summarizer`: To be used as the VERY LAST STEP to consolidate all findings and give a final, friendly answer to the user.

First, route the user's request to the appropriate specialist agent. The agents will continue to work until all parts of the user's request have been addressed.
Once all specialist tasks are complete, route to the `summarizer`.
Only respond with `FINISH` if the user is saying goodbye or the conversation is truly over.

User's question:
{messages}"""
    )
    return supervisor_prompt | llm.with_structured_output(Router)