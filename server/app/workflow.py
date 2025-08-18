# File: app/workflow.py
from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import partial
from langgraph.graph.message import add_messages

# Import all our modular components
from .tools.tools import all_pension_tools
from .agents.risk_agent import create_risk_agent
from .agents.fraud_agent import create_fraud_agent
from .agents.pension_agent import create_pension_agent
from .agents.summarizer_agent import create_summarizer_chain
from .agents.supervisor import create_supervisor_chain


# --- State Definition (Now includes intermediate_steps) ---
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    intermediate_steps: List[BaseMessage]
    turns: int


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
        resp = supervisor_chain_runnable.invoke({"messages": state["messages"]})
        if isinstance(resp, dict):
            next_value = resp.get("next") or resp.get("output") or resp.get("text")
        else:
            next_value = getattr(resp, "next", None) or (resp if isinstance(resp, str) else None)
        # Increment loop counter to avoid infinite routing
        new_turns = int(state.get("turns", 0)) + 1
        if new_turns >= 5:
            next_value = "FINISH"
        if not next_value:
            next_value = "FINISH"
        return {"next": next_value, "turns": new_turns}

    # --- Generic Agent Runner ---
    def agent_node(state: AgentState, agent_runnable):
        # Find the latest user message content, supporting both HumanMessage and tuple-style ("user", text)
        last_user_text = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_text = msg.content
                break
            # Support tuple/dict-like messages e.g., ("user", "..."), {"role":"user","content":"..."}
            if isinstance(msg, tuple) and len(msg) >= 2 and str(msg[0]).lower() in ("user", "human"):
                last_user_text = msg[1]
                break
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() in ("user", "human"):
                last_user_text = msg.get("content")
                break

        if not last_user_text:
            return {"messages": list(state["messages"]) + [AIMessage(content="⚠️ No user message found to process.")]}

        result = agent_runnable.invoke({
            "input": last_user_text
        })

        # Normalize result into messages and propagate intermediate steps (tools used)
        new_messages: List[BaseMessage] = list(state["messages"])  # type: ignore[arg-type]
        new_intermediate_steps = list(state.get("intermediate_steps", []))

        final_text = None
        tools_summary = None

        if isinstance(result, str):
            final_text = result
        elif isinstance(result, dict):
            final_text = result.get("output") or result.get("content") or result.get("text")
            steps_result = result.get("intermediate_steps")
            if steps_result:
                # steps_result is typically a list of (AgentAction, observation) tuples
                try:
                    tools_summary = []
                    for action, observation in steps_result:
                        tool_name = getattr(action, "tool", None) or getattr(action, "tool_name", None) or "tool"
                        tool_input = getattr(action, "tool_input", None) or getattr(action, "input", None)
                        tools_summary.append(f"{tool_name}({tool_input}) -> {str(observation)[:200]}")
                    new_intermediate_steps.extend(steps_result)
                    tools_summary = "\n".join(tools_summary)
                except Exception:
                    pass
        else:
            final_text = str(result)

        if final_text:
            new_messages.append(AIMessage(content=final_text))
        if tools_summary:
            new_messages.append(AIMessage(content=f"[Tools executed]\n{tools_summary}"))

        updates = {"messages": new_messages}
        if new_intermediate_steps:
            updates["intermediate_steps"] = new_intermediate_steps
        return updates

    # --- Summarizer Node ---
    def summarizer_node(state: AgentState):
        summary = summarizer_chain_runnable.invoke({"messages": state["messages"]})
        if isinstance(summary, str):
            summary_text = summary
        elif isinstance(summary, dict):
            summary_text = summary.get("output") or summary.get("content") or summary.get("text") or str(summary)
        else:
            summary_text = getattr(summary, "content", None) or str(summary)
        return {"messages": state["messages"] + [AIMessage(content=summary_text)]}

    # --- Build the graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)

    # Specific nodes from generic agent_node
    workflow.add_node("risk_analyst", partial(agent_node, agent_runnable=risk_agent_runnable))
    workflow.add_node("fraud_detector", partial(agent_node, agent_runnable=fraud_agent_runnable))
    workflow.add_node("projection_specialist", partial(agent_node, agent_runnable=projection_agent_runnable))

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

    # After specialist agent -> back to supervisor
    workflow.add_edge("risk_analyst", "supervisor")
    workflow.add_edge("fraud_detector", "supervisor")
    workflow.add_edge("projection_specialist", "supervisor")

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
        try:
            image_data = graph_viz.draw_mermaid_png()
        except AttributeError:
            image_data = graph_viz.draw_png()
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
