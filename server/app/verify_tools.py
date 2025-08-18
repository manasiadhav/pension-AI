# File: test_agent_execution.py
import sys
import os
import asyncio
from datetime import datetime
from langchain_core.messages import HumanMessage

# Add the 'app' directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import the compiled graph from your workflow.py
from app.workflow import graph

async def run_query(user_query, user_id=1, user_role="resident"):
    """
    Simulates a user query and streams the output to show agent execution.
    """
    print(f"\n{'='*80}")
    print(f"TESTING QUERY: '{user_query}'")
    print(f"User Info: ID={user_id}, Role={user_role}")
    print(f"{'='*80}")

    # Prepare the initial state with the user's message and context
    initial_state = {
        "messages": [HumanMessage(content=f"User Info: id={user_id}, role={user_role}. Query: {user_query}")],
        # The agent's will populate this as needed
        "intermediate_steps": [],
        "next": "supervisor"
    }

    final_answer = ""
    try:
        # Use astream() to see the step-by-step progress
        async for event in graph.astream(initial_state):
            # Print the node being entered and its output
            for key, value in event.items():
                if key == "__end__":
                    continue

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Entering Node: {key.upper()}")
                
                # Check for tool calls made by the agent
                if key == "tool_executor":
                    # ToolExecutor's output is a list of messages. We're interested in the tool call
                    tool_call_data = value.get("messages", [])[0].tool_calls[0]
                    tool_name = tool_call_data['name']
                    tool_args = tool_call_data['args']
                    print(f"   --> Tool called: `{tool_name}` with args: {tool_args}")
                elif key == "supervisor":
                    print(f"   --> Supervisor's Decision: Route to '{value.get('next')}'")
                elif key == "summarizer":
                    print("   --> Summarizer Agent is compiling the final response...")
                
                # Check for the final response from the agent
                if "messages" in value and value["messages"] and key != "tool_executor":
                    last_message = value["messages"][-1]
                    if last_message.type == "ai":
                        final_answer = last_message.content

    except Exception as e:
        print(f"❌ An error occurred during the test: {e}")
        final_answer = f"Test failed due to an error: {e}"

    print(f"\n{'-'*80}")
    print("FINAL RESPONSE:")
    print(final_answer)
    print(f"{'-'*80}\n")

async def main():
    print("Starting agent execution tests...")

    # Query 1: Triggers the Pension Projection Agent
    await run_query("What will my pension be in 10 years?")

    # Query 2: Triggers the Risk Analysis Agent
    await run_query("Can you give me a risk assessment of my portfolio?")

    # Query 3: Triggers the Fraud Detection Agent
    await run_query("I need to check for any suspicious transactions.")

    # Query 4: Triggers the Knowledge Base Search via a general question
    await run_query("What is a defined contribution plan?")

    # Query 5: Triggers the Knowledge Base Search for a private document
    # NOTE: Assumes a document has been ingested for this user ID.
    await run_query("What is the penalty for early withdrawal from my pension?")

    print("All agent execution tests complete.")

if __name__ == "__main__":
    asyncio.run(main())