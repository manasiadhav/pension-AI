# File: app/agents/supervisor.py
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel
from langchain.prompts import ChatPromptTemplate
import re

class Router(BaseModel):
    next: Literal["risk_analyst", "fraud_detector", "projection_specialist", "summarizer", "visualizer", "FINISH"]

def create_supervisor_chain(llm):
    """Factory for the Supervisor's routing logic."""
    
    def validate_query_content(query: str) -> tuple[bool, str]:
        """Validate query content and return (is_valid, reason_if_invalid)"""
        
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
        for category, patterns in blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query.lower()):
                    return False, f"Query contains {category} content which is not allowed"
        
        return True, ""
    
    supervisor_prompt = ChatPromptTemplate.from_template(
        """You are a supervisor of a team of expert AI agents. Your job is to route the user's request to the appropriate agent based on context and available data.

IMPORTANT: You are a PENSION ANALYSIS SYSTEM ONLY. You cannot and will not provide:
- Religious advice or spiritual guidance
- Political opinions or political analysis  
- Specific investment strategies or stock recommendations
- Cryptocurrency advice
- Real estate investment advice

If a user asks about these topics, route to "FINISH" and explain that you can only help with pension analysis, risk assessment, and fraud detection.

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
    
    # Create a chain that validates input first
    def supervisor_with_guardrails(state):
        # Extract the user query from messages
        messages = state.get("messages", [])
        user_query = ""
        for msg in messages:
            if isinstance(msg, str):
                user_query = msg
                break
            elif hasattr(msg, 'content'):
                user_query = msg.content
                break
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                user_query = str(msg[1])
                break
        
        print(f"🔍 Supervisor: Processing query: '{user_query}'")
        
        # Apply input validation
        is_valid, reason = validate_query_content(user_query)
        print(f"🔍 Supervisor: Content validation - is_valid: {is_valid}, reason: {reason}")
        
        if not is_valid:
            # Return the format the workflow expects
            result = {"next": "FINISH"}
            print(f"🔍 Supervisor: Guardrail triggered, returning: {result}")
            return result
        
        # If valid, proceed with normal routing
        try:
            router_result = (supervisor_prompt | llm.with_structured_output(Router)).invoke({"messages": messages})
            # Convert Router object to expected format
            result = {"next": router_result.next}
            print(f"🔍 Supervisor: Normal routing, returning: {result}")
            return result
        except Exception as e:
            print(f"🔍 Supervisor: Error in routing: {e}")
            # Fallback to projection_specialist for pension-related queries
            if any(word in user_query.lower() for word in ["pension", "retirement", "savings", "growth"]):
                result = {"next": "projection_specialist"}
            else:
                result = {"next": "risk_analyst"}
            print(f"🔍 Supervisor: Fallback routing, returning: {result}")
            return result
    
    return supervisor_with_guardrails