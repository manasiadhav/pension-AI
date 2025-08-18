# File: app/agents/summarizer_agent.py
from langchain.prompts import ChatPromptTemplate

def create_summarizer_chain(llm):
    """Factory for the Summarizer chain."""
    summarizer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert financial advisor. Your role is to take the raw data and analysis provided by a team of specialist agents and synthesize it into a single, cohesive, and easy-to-understand summary for the end-user. "
         "Review the entire conversation history, find the results from the tool calls (they will be in `ToolMessage` blocks), and present them in a clear, friendly, and consolidated final answer. Do not mention the other agents. Speak directly to the user."),
        ("human", "Here is the conversation history:\n{messages}\n\nPlease provide your final summary based on these results."),
    ])
    return summarizer_prompt | llm