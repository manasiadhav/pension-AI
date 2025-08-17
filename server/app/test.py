import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Import the specific components we need to test from your agent system
from app.agents import supervisor_router, AgentState

# A list of test cases, each with a query and the expected agent to be called
TEST_CASES = [
    {
        "name": "Specific Risk Analysis Query",
        "query": "I need a detailed risk profile for my account.",
        "expected_agent": "risk_analyst"
    },
    {
        "name": "Specific Fraud Detection Query",
        "query": "Can you check my recent transactions for suspicious activity?",
        "expected_agent": "fraud_detector"
    },
    {
        "name": "Specific Pension Projection Query",
        "query": "What will my savings look like in 10 years?",
        "expected_agent": "projection_specialist"
    },
    {
        "name": "Multi-Step Query (Fraud First)",
        "query": "Check for fraud and then give me a risk analysis.",
        "expected_agent": "fraud_detector"  # Supervisor should pick the FIRST task
    },
    {
        "name": "Ambiguous Query",
        "query": "How is my pension doing overall?",
        "expected_agent": "risk_analyst"  # We predict it will choose risk, but projection is also plausible
    },
    {
        "name": "Finishing Query",
        "query": "Thank you, that's all I needed.",
        "expected_agent": "FINISH"
    }
]

def run_supervisor_tests():
    """
    Runs a series of tests against the supervisor_router to check its routing logic.
    """
    print("--- Starting Supervisor Routing Test ---")
    
    passed_count = 0
    for case in TEST_CASES:
        print(f"\n--- Testing Case: {case['name']} ---")
        print(f"Query: \"{case['query']}\"")
        
        # 1. Create the input state for the supervisor function
        # The state is a dictionary that must match the AgentState structure
        initial_state = AgentState(
            messages=[HumanMessage(content=case['query'])],
            next="" # Start with next as empty
        )
        
        # 2. Call the supervisor router function directly
        result = supervisor_router(initial_state)
        
        # 3. Get the actual decision made by the supervisor
        actual_agent = result['next']
        
        print(f"Expected Agent: {case['expected_agent']}")
        print(f"Actual Agent Routed To: {actual_agent}")
        
        # 4. Check if the actual decision matches the expected one
        if actual_agent == case['expected_agent']:
            print("✅ PASS")
            passed_count += 1
        else:
            print(f"❌ FAIL - Supervisor routed to '{actual_agent}' instead of '{case['expected_agent']}'")

    print("\n--- Test Summary ---")
    print(f"{passed_count} / {len(TEST_CASES)} tests passed.")

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("Please set your API Key in the .env file.")
    else:
        run_supervisor_tests()