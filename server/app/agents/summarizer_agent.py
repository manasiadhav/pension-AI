# File: app/agents/summarizer_agent.py
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import re

def create_summarizer_chain(llm):
    """Factory for the Summarizer chain."""
    summarizer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert financial advisor. Your role is to take the raw data and analysis provided by a team of specialist agents and synthesize it into a single, cohesive, and easy-to-understand summary for the end-user. "
         "Review the entire conversation history, find the results from the tool calls (they will be in `ToolMessage` blocks), and present them in a clear, friendly, and consolidated final answer. Do not mention the other agents. Speak directly to the user."
         "\n\nIMPORTANT: If there are charts or visualizations available, mention them in your summary and indicate that chart data is available for the frontend to render."
         "\n\nCRITICAL: Focus ONLY on pension data analysis, risk assessment, and fraud detection. Do NOT provide religious advice, political opinions, or specific investment strategies."),
        ("human", "Here is the conversation history:\n{messages}\n\nPlease provide your final summary based on these results."),
    ])
    
    def apply_content_guardrails(text: str) -> str:
        """Apply content guardrails to filter inappropriate content"""
        
        # Define blocked content patterns
        blocked_patterns = {
            'religious': [
                r'\b(pray|prayer|god|jesus|allah|buddha|hindu|islam|christian|jewish|religious|spiritual|faith|blessing|divine|heaven|hell)\b',
                r'\b(amen|hallelujah|om|namaste|shalom|salaam)\b',
                r'\b(church|mosque|temple|synagogue|worship|meditation)\b'
            ],
            'political': [
                r'\b(democrat|republican|liberal|conservative|left|right|wing|party|election|vote|campaign|politician|senator|congress|president)\b',
                r'\b(government|administration|policy|legislation|bill|law|regulation)\b',
                r'\b(progressive|moderate|radical|extremist|activist|protest|rally)\b'
            ],
            'investment_strategy': [
                r'\b(buy|sell|hold|stock|shares|equity|market|timing|entry|exit|position|portfolio|allocation)\b',
                r'\b(day trading|swing trading|momentum|value|growth|dividend|yield)\b',
                r'\b(cryptocurrency|bitcoin|ethereum|blockchain|ico|token|coin)\b',
                r'\b(real estate|property|mortgage|loan|credit|debt|leverage)\b',
                r'\b(hedge fund|private equity|venture capital|startup|ipo|merger|acquisition)\b'
            ]
        }
        
        # Check for blocked content
        found_issues = []
        for category, patterns in blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    found_issues.append(category)
                    break
        
        if found_issues:
            # Replace blocked content with appropriate message
            replacement_text = (
                f"I apologize, but I cannot provide advice related to {', '.join(found_issues)}. "
                f"Please focus your questions on pension analysis, risk assessment, or fraud detection. "
                f"Here is the relevant financial data: {text[:200]}..."
            )
            return replacement_text
        
        return text
    
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
        
        # Apply content guardrails
        summary_text = apply_content_guardrails(summary_text)
        
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