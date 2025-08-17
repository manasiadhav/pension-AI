# File: app/agents/agent_system.py

from typing import TypedDict, List, Literal
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain.agents import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.prompts import ChatPromptTemplate

# Import our tools
from .tools import all_pension_tools

# --- LLM Initialization ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

# --- Agent State ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

# --- Agent Definitions (Step 2) ---
def create_agent(llm, tools, system_prompt: str):
    prompt = hub.pull("hwchase17/react")
    prompt = prompt.partial(instructions=system_prompt)
    agent_runnable = create_react_agent(llm, tools, prompt)
    return agent_runnable

risk_agent = create_agent(llm, [all_pension_tools[0]], "You are a financial risk analysis expert...")
fraud_agent = create_agent(llm, [all_pension_tools[1]], "You are a fraud detection expert...")
projection_agent = create_agent(llm, [all_pension_tools[2]], "You are a pension projection expert...")

### NEW: Define the Summarization Agent ###
# This agent has no tools. Its only job is to synthesize the final answer.
summarizer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert financial advisor. Your role is to take the raw data and analysis provided by a team of specialist agents and synthesize it into a single, cohesive, and easy-to-understand summary for the end-user. "
            "Review the entire conversation history, find the results from the tool calls (they will be in `ToolMessage` blocks), and present them in a clear, friendly, and consolidated final answer. Do not mention the other agents. Speak directly to the user."
        ),
        ("human", "Here is the conversation history:\n{messages}\n\nPlease provide your final summary based on these results."),
    ]
)

# The summarizer is a simple chain, not a ReAct agent
summarizer_agent = summarizer_prompt | llm

# --- Supervisor Definition (Step 3) ---
### MODIFIED: Add "summarizer" to the Router's possible next steps ###
class Router(BaseModel):
    next: Literal["risk_analyst", "fraud_detector", "projection_specialist", "summarizer", "FINISH"]

### MODIFIED: Update the supervisor's instructions ###
supervisor_prompt = ChatPromptTemplate.from_template(
    """You are a supervisor of a team of expert AI agents...

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

supervisor_chain = (
    supervisor_prompt
    | llm.with_structured_output(Router)
)

def supervisor_router(state: AgentState):
    print("--- SUPERVISOR: Deciding next action... ---")
    # We get the content of all messages except the last one (which is the tool output)
    message_contents = [msg.content for msg in state["messages"][:-1]]
    response = supervisor_chain.invoke({"messages": message_contents})
    print(f"--- SUPERVISOR: Routing to {response.next} ---")
    return {"next": response.next}

# --- Build Graph (Step 4) ---
workflow = StateGraph(AgentState)

# Add nodes for each agent and the tool executor
workflow.add_node("risk_analyst", risk_agent)
workflow.add_node("fraud_detector", fraud_agent)
workflow.add_node("projection_specialist", projection_agent)
workflow.add_node("tool_executor", ToolNode(all_pension_tools))
workflow.add_node("supervisor", supervisor_router)
### NEW: Add the summarizer node ###
workflow.add_node("summarizer", lambda state: {"messages": [summarizer_agent.invoke({"messages": [msg.content for msg in state['messages']]})] })


### MODIFIED: Update the routing logic to include the summarizer ###
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "risk_analyst": "risk_analyst",
        "fraud_detector": "fraud_detector",
        "projection_specialist": "projection_specialist",
        "summarizer": "summarizer", ### NEW EDGE
        "FINISH": END,
    },
)

# Define edges from agents back to the tool executor, and from the executor back to the supervisor
workflow.add_edge("risk_analyst", "tool_executor")
workflow.add_edge("fraud_detector", "tool_executor")
workflow.add_edge("projection_specialist", "tool_executor")
workflow.add_edge("tool_executor", "supervisor")

### NEW: The summarizer is the new end of the line ###
workflow.add_edge("summarizer", END)

workflow.set_entry_point("supervisor")
graph = workflow.compile()
print("Multi-agent graph with Summarizer compiled successfully.")

# ... (The save_graph_image function remains the same) ...


def save_graph_image():
    """Generates and saves a PNG image of the compiled graph."""
    try:
        # Get the graphviz object from the compiled graph
        graph_viz = graph.get_graph()
        
        # Draw the graph and get the PNG image data
        image_data = graph_viz.draw_mermaid_png()
        
        # Save the image to a file
        with open("pension_agent_supervisor_graph.png", "wb") as f:
            f.write(image_data)
        
        print("\n✅ Graph visualization saved to 'pension_agent_supervisor_graph.png'")

    except ImportError as e:
        print(f"\n❌ ERROR: Could not generate graph image. Please ensure you have installed the prerequisites.")
        print("   System-level: 'graphviz' (e.g., 'sudo apt-get install graphviz')")
        print("   Python packages: 'pip install pygraphviz Pillow'")
        print(f"   Original error: {e}")
    except Exception as e:
        print(f"\n❌ An error occurred while generating the graph: {e}")

# This allows you to generate the image by running the file directly
if __name__ == "__main__":
    save_graph_image()