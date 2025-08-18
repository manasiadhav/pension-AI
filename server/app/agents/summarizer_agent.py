# File: app/agents/summarizer_agent.py
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

def create_summarizer_chain(llm):
    """Factory for the Summarizer chain."""
    summarizer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert financial advisor. Your role is to take the raw data and analysis provided by a team of specialist agents and synthesize it into a single, cohesive, and easy-to-understand summary for the end-user. "
         "Review the entire conversation history, find the results from the tool calls (they will be in `ToolMessage` blocks), and present them in a clear, friendly, and consolidated final answer. Do not mention the other agents. Speak directly to the user."
         "\n\nIMPORTANT: If there are charts or visualizations available, mention them in your summary and indicate that chart data is available for the frontend to render."),
        ("human", "Here is the conversation history:\n{messages}\n\nPlease provide your final summary based on these results."),
    ])
    
    def summarizer_with_charts(state):
        # Get the text summary from the LLM
        summary_result = (summarizer_prompt | llm).invoke({"messages": state["messages"]})
        
        # Extract the summary text
        if isinstance(summary_result, str):
            summary_text = summary_result
        elif hasattr(summary_result, 'content'):
            summary_text = summary_result.content
        else:
            summary_text = str(summary_result)
        
        # Create the final response with both summary and chart data
        final_response = {
            "summary": summary_text,
            "charts": state.get("charts", {}),
            "plotly_figs": state.get("plotly_figs", {}),
            "chart_images": state.get("chart_images", {})
        }
        
        # Add the structured response as a message
        new_messages = list(state["messages"])
        new_messages.append(AIMessage(content=f"[FINAL_RESPONSE] {str(final_response)}"))
        
        return {"messages": new_messages, "final_response": final_response}
    
    return summarizer_with_charts