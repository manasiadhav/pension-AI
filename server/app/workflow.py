# File: app/workflow.py
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import partial

# Import all our modular components
from .tools.tools import all_pension_tools
from .agents.risk_agent import create_risk_agent
from .agents.fraud_agent import create_fraud_agent
from .agents.pension_agent import create_pension_agent
from .agents.summarizer_agent import create_summarizer_chain
from .agents.supervisor import create_supervisor_chain


# --- State Definition (Now includes intermediate_steps) ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    intermediate_steps: List[BaseMessage]


# --- Graph Builder Function ---
def build_agent_workflow():
    """
    Builds the LangGraph workflow by creating instances of all agents
    and wiring them together.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

    # Create all agent and supervisor runnables
    risk_agent_runnable = create_risk_agent(llm)
    fraud_agent_runnable = create_fraud_agent(llm)
    projection_agent_runnable = create_pension_agent(llm)
    summarizer_chain_runnable = create_summarizer_chain(llm)
    supervisor_chain_runnable = create_supervisor_chain(llm)

    # --- Supervisor Node ---
    def supervisor_node(state: AgentState):
        # Pass the full messages list instead of just string content
        response = supervisor_chain_runnable.invoke({"messages": state["messages"]})
        return {"next": response.next}

    # --- Generic Agent Runner ---
    def agent_node(state: AgentState, agent_runnable):
        last_user_message = next(
            (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            None
        )
        if not last_user_message:
            return {"messages": state["messages"] + [AIMessage(content="⚠️ No user message found to process.")]}

        result = agent_runnable.invoke({
            "input": last_user_message.content,
            "agent_scratchpad": state["intermediate_steps"]
        })

        # Normalize result into an AIMessage
        if isinstance(result, str):
            result_msg = AIMessage(content=result)
        elif isinstance(result, dict) and "output" in result:
            result_msg = AIMessage(content=result["output"])
        else:
            result_msg = AIMessage(content=str(result))

        return {"messages": state["messages"] + [result_msg]}

    # --- Summarizer Node ---
    def summarizer_node(state: AgentState):
        summary = summarizer_chain_runnable.invoke({"messages": state["messages"]})
        return {"messages": state["messages"] + [AIMessage(content=summary)]}

    # --- Build the graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)

    # Specific nodes from generic agent_node
    workflow.add_node("risk_analyst", partial(agent_node, agent_runnable=risk_agent_runnable))
    workflow.add_node("fraud_detector", partial(agent_node, agent_runnable=fraud_agent_runnable))
    workflow.add_node("projection_specialist", partial(agent_node, agent_runnable=projection_agent_runnable))

    workflow.add_node("tool_executor", ToolNode(all_pension_tools))
    workflow.add_node("summarizer", summarizer_node)

    # --- Wire up the graph ---
    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges("supervisor", lambda x: x["next"], {
        "risk_analyst": "risk_analyst",
        "fraud_detector": "fraud_detector",
        "projection_specialist": "projection_specialist",
        "summarizer": "summarizer",
        "FINISH": END,
    })

    # After specialist agent -> tool executor
    workflow.add_edge("risk_analyst", "tool_executor")
    workflow.add_edge("fraud_detector", "tool_executor")
    workflow.add_edge("projection_specialist", "tool_executor")

    # After tools -> back to supervisor
    workflow.add_edge("tool_executor", "supervisor")

    # Summarizer ends the flow
    workflow.add_edge("summarizer", END)

    return workflow.compile()


# Compile and make available
graph = build_agent_workflow()
print("✅ Modular multi-agent graph compiled successfully.")


def save_graph_image():
    """Generates and saves a PNG image of the compiled graph."""
    try:
        graph_viz = graph.get_graph()
        image_data = graph_viz.draw_mermaid_png()
        with open("pension_agent_supervisor_graph.png", "wb") as f:
            f.write(image_data)
        print("\n✅ Graph visualization saved to 'pension_agent_supervisor_graph.png'")
    except ImportError as e:
        print(f"\n❌ ERROR: Could not generate graph image. Please install prerequisites.")
        print("   System-level: 'graphviz' (e.g., 'sudo apt-get install graphviz')")
        print("   Python packages: 'pip install pygraphviz Pillow'")
        print(f"   Original error: {e}")
    except Exception as e:
        print(f"\n❌ An error occurred while generating the graph: {e}")


if __name__ == "__main__":
    save_graph_image()
