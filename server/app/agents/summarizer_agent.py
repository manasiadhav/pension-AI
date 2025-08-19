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
        """
        Apply content guardrails to filter out inappropriate content.
        Only blocks clearly inappropriate content, not legitimate financial analysis.
        """
        blocked_patterns = {
            'religious': [
                r'\b(pray|prayer|god|jesus|allah|buddha|hindu|islam|christian|jewish|religious|spiritual|faith|blessing|divine|heaven|hell)\b',
                r'\b(amen|hallelujah|om|namaste|shalom|salaam)\b',
                r'\b(church|mosque|temple|synagogue|worship|meditation)\b'
            ],
            'political': [
                r'\b(democrat|republican|liberal|conservative|left|right|wing|party|election|vote|campaign|politician|senator|congress|president)\b',
                r'\b(government|administration|policy|legislation|bill|law|regulation)\b',
                r'\b(progressive|moderate|radical|extremist|activist|protest|rally)\b',
                r'\bpolitic\w*\b'  # political, politics
            ],
            'investment_strategy': [
                r'\b(buy\s+this\s+stock|sell\s+that\s+stock|invest\s+in\s+bitcoin|buy\s+crypto|day\s+trading|swing\s+trading)\b',
                r'\b(you\s+should\s+buy|you\s+should\s+sell|i\s+recommend\s+buying|i\s+recommend\s+selling)\b',
                r'\b(put\s+all\s+your\s+money\s+in|move\s+to\s+cash|market\s+timing|entry\s+point|exit\s+point)\b',
                r'\b(hedge\s+fund|private\s+equity|venture\s+capital|startup|ico|token|coin)\b'
            ]
        }
        
        # Check for blocked content
        should_block = False
        blocked_category = ""
        for category, patterns in blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    should_block = True
                    blocked_category = category
                    break
            if should_block:
                break
        
        if should_block:
            print(f"🔍 Summarizer: Guardrail triggered for {blocked_category} content")
            # Replace blocked content with appropriate message
            replacement_text = (
                "I apologize, but I cannot provide advice related to "
                f"{blocked_category}. Please focus your questions on pension analysis, "
                "risk assessment, or fraud detection. Here is the relevant financial data: "
            )
            
            # Find the blocked content and replace it
            for pattern in blocked_patterns[blocked_category]:
                text = re.sub(pattern, replacement_text, text, flags=re.IGNORECASE)
        
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