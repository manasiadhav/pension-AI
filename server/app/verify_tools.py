# save as run_workflow_stream.py (in repo root)
import asyncio, json
from app.workflow import graph

async def main():
    query = 'Check risk, possible fraud, and projections for id 123.'
    async for event in graph.astream({'messages': [('user', query)]}):
        # Pretty-print each step so you can see which node ran and what it produced
        print(json.dumps(event, indent=2, default=str))

if __name__ == '__main__':
    asyncio.run(main())