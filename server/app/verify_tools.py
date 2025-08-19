# save as run_workflow_stream.py (in repo root)
import asyncio, json
from app.workflow import graph

async def main():
    # Use a numeric user ID that actually exists in your database
    # Change this to a real user ID from your database
    query = "What's the distribution of my pension fund for user 102."  # Use actual numeric ID
    
    print(f"🔍 Testing query: {query}")
    print("Starting workflow...")
    
    try:
        # Use invoke instead of astream to avoid recursion issues
        result = graph.invoke({'messages': [('user', query)]})
        print("\n✅ Workflow completed successfully!")
        print("Result keys:", list(result.keys()))
        
        # Check if we got a final response
        if 'final_response' in result:
            print("\n📊 Final Response:")
            print(json.dumps(result['final_response'], indent=2, default=str))
        else:
            print("\n📝 Messages:")
            for msg in result.get('messages', []):
                if hasattr(msg, 'content'):
                    print(f"  - {msg.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())