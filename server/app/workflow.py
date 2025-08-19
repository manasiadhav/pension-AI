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
from .agents.visualizer_agent import create_visualizer_node
from .agents.supervisor import create_supervisor_chain


# --- State Definition (Now includes intermediate_steps) ---
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    intermediate_steps: List[BaseMessage]
    turns: int
    charts: dict
    chart_images: dict
    plotly_figs: dict
    final_response: dict


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
        # Check if we're returning from a specialist agent (have intermediate_steps)
        has_specialist_data = bool(state.get("intermediate_steps"))
        
        # Check if we have visualization data (charts, plotly_figs, etc.)
        has_visualization_data = bool(
            state.get("charts") or 
            state.get("plotly_figs") or 
            state.get("chart_images")
        )
        
        if has_visualization_data:
            # We have visualization data, route to summarizer for final consolidation
            return {"next": "summarizer", "turns": state.get("turns", 0)}
        
        elif has_specialist_data:
            # We have data from specialist agents, now make intelligent routing decision
            # Check if the original query requested visualization or if data would benefit from it
            original_query = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_query = msg.content.lower()
                    break
                elif isinstance(msg, dict) and str(msg.get("role", "")).lower() in ("user", "human"):
                    original_query = msg.get("content", "").lower()
                    break
                elif isinstance(msg, tuple) and len(msg) >= 2 and str(msg[0]).lower() in ("user", "human"):
                    original_query = str(msg[1]).lower()
                    break
            
            # Check what data we have
            has_projection = any(
                step[1] if isinstance(step, (list, tuple)) and len(step) == 2 else None
                for step in state.get("intermediate_steps", [])
                if hasattr(step[0], "tool") and step[0].tool == "project_pension"
            )
            has_risk = any(
                step[1] if isinstance(step, (list, tuple)) and len(step) == 2 else None
                for step in state.get("intermediate_steps", [])
                if hasattr(step[0], "tool") and step[0].tool == "analyze_risk_profile"
            )
            has_fraud = any(
                step[1] if isinstance(step, (list, tuple)) and len(step) == 2 else None
                for step in state.get("intermediate_steps", [])
                if hasattr(step[0], "tool") and step[0].tool == "detect_fraud"
            )
            
            # Enhanced decision logic for visualization
            should_visualize = (
                "chart" in original_query or 
                "graph" in original_query or 
                "visual" in original_query or
                "visualize" in original_query or
                "show me" in original_query or
                "display" in original_query or
                "plot" in original_query or
                (has_projection and ("growth" in original_query or "time" in original_query or "progress" in original_query or "goal" in original_query or "retirement" in original_query or "pension" in original_query or "projection" in original_query)) or
                (has_risk and "risk" in original_query) or
                (has_fraud and "fraud" in original_query)
            )
            
            print(f"🔍 Supervisor Decision: original_query='{original_query}', should_visualize={should_visualize}")
            print(f"   has_projection={has_projection}, has_risk={has_risk}, has_fraud={has_fraud}")
            
            if should_visualize:
                print("   ✅ Routing to visualizer")
                return {"next": "visualizer", "turns": state.get("turns", 0)}
            else:
                print("   📝 Routing to summarizer")
                return {"next": "summarizer", "turns": state.get("turns", 0)}
        
        # First pass - route to specialist agents based on user query
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

        result = agent_runnable({
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
        # Call the summarizer function directly (it returns a function, not a chain)
        summary_result = summarizer_chain_runnable(state)
        
        # Extract the final response if available
        final_response = summary_result.get("final_response", {})
        
        # Add the summary message
        new_messages = list(state["messages"])
        if final_response:
            # Add the structured final response
            new_messages.append(AIMessage(content=final_response.get("summary", "Summary completed.")))
            # Store the final response in state for frontend access
            return {
                "messages": new_messages,
                "final_response": final_response
            }
        else:
            # Fallback to old behavior
            if isinstance(summary_result, str):
                summary_text = summary_result
            elif isinstance(summary_result, dict):
                summary_text = summary_result.get("output") or summary_result.get("content") or summary_result.get("text") or str(summary_result)
            else:
                summary_text = getattr(summary_result, "content", None) or str(summary_result)
            return {"messages": state["messages"] + [AIMessage(content=summary_text)]}

    # --- Visualization Node ---
    from .agents.visualizer_agent import create_visualizer_node as _make_vis
    visualizer_node = _make_vis()

    # --- Build the graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)

    # Specific nodes from generic agent_node
    workflow.add_node("risk_analyst", partial(agent_node, agent_runnable=risk_agent_runnable))
    workflow.add_node("fraud_detector", partial(agent_node, agent_runnable=fraud_agent_runnable))
    workflow.add_node("projection_specialist", partial(agent_node, agent_runnable=projection_agent_runnable))

    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("visualizer", visualizer_node)

    # --- Wire up the graph ---
    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges("supervisor", lambda x: x["next"], {
        "risk_analyst": "risk_analyst",
        "fraud_detector": "fraud_detector",
        "projection_specialist": "projection_specialist",
        "summarizer": "summarizer",
        "visualizer": "visualizer",
        "FINISH": END,
    })

    # After specialist agents -> return to supervisor for intelligent routing
    workflow.add_edge("risk_analyst", "supervisor")
    workflow.add_edge("fraud_detector", "supervisor")
    workflow.add_edge("projection_specialist", "supervisor")

    # After visualizer -> return to supervisor for final routing decision
    workflow.add_edge("visualizer", "supervisor")

    # After summarizer -> workflow ends
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
