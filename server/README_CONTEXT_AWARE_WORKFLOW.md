# Context-Aware Workflow Routing (Three-Stage Process)

## Overview

The pension AI system now uses **three-stage context-aware routing** where the supervisor makes ALL routing decisions and agents NEVER communicate directly with each other. The supervisor acts as the central hub for all agent coordination.

## Key Architecture Principles

1. **No Direct Agent Communication**: Agents only communicate through the supervisor
2. **Centralized Routing**: All decisions made by the supervisor node
3. **Three-Stage Process**: Clear separation of routing phases
4. **State-Based Decisions**: Supervisor analyzes state to make routing decisions

## How It Works

### Stage 1: Initial Routing
```
User Query → Supervisor → Specialist Agent
```

The supervisor analyzes the user's question and routes to the appropriate specialist:
- **Risk Analysis**: `risk_analyst` for risk profile questions
- **Fraud Detection**: `fraud_detector` for suspicious transaction questions  
- **Pension Projection**: `projection_specialist` for growth/time questions

### Stage 2: Post-Specialist Decision
```
Specialist Agent → Supervisor → [visualizer] OR [summarizer]
```

After the specialist agent completes its work:
- Tool execution results are captured in `intermediate_steps`
- Control returns to the supervisor
- The supervisor analyzes original query + available data
- Makes intelligent decision: visualizer or summarizer?

### Stage 3: Post-Visualization Decision
```
visualizer → Supervisor → summarizer
```

After visualization (if called):
- Chart data is added to state (`charts`, `plotly_figs`, `chart_images`)
- Control returns to supervisor
- Supervisor routes to summarizer for final consolidation

### Final Flow
```
summarizer → END
```

The summarizer consolidates everything into a final answer.

## Example: "How does my pension grow over time?"

### Stage 1: Initial Routing
```
User: "How does my pension grow over time?"
↓
Supervisor: Analyzes query, sees "grow over time" → routes to projection_specialist
↓
Projection Specialist: Calls project_pension tool, gets projection data
```

### Stage 2: Post-Specialist Decision
```
Projection Specialist → Supervisor (with projection data)
↓
Supervisor: Has projection data + "grow over time" in original query
↓
Decision: Route to visualizer (✅ growth + time context = visualization needed)
↓
Visualizer: Generates pension growth charts
```

### Stage 3: Post-Visualization Decision
```
Visualizer → Supervisor (with chart data)
↓
Supervisor: Has charts, plotly_figs, chart_images
↓
Decision: Route to summarizer (final consolidation needed)
↓
Summarizer: Final consolidated answer with charts
```

## Code Implementation

### Supervisor Node Logic (Three-Stage)
```python
def supervisor_node(state: AgentState):
    # Stage 3: Check if we have visualization data
    has_visualization_data = bool(
        state.get("charts") or 
        state.get("plotly_figs") or 
        state.get("chart_images")
    )
    
    if has_visualization_data:
        # Route to summarizer for final consolidation
        return {"next": "summarizer"}
    
    # Stage 2: Check if we have specialist data
    elif has_specialist_data:
        # Make intelligent visualization decision
        should_visualize = (
            "chart" in original_query or 
            "graph" in original_query or 
            "visual" in original_query or
            "show me" in original_query or
            (has_projection and ("growth" in original_query or "time" in original_query)) or
            (has_risk and "risk" in original_query) or
            (has_fraud and "fraud" in original_query)
        )
        
        if should_visualize:
            return {"next": "visualizer"}
        else:
            return {"next": "summarizer"}
    
    # Stage 1: First pass - route to specialist agents
    # ... existing logic ...
```

### Graph Structure (No Direct Agent Communication)
```python
# All routing goes through supervisor
workflow.add_edge("risk_analyst", "supervisor")
workflow.add_edge("fraud_detector", "supervisor") 
workflow.add_edge("projection_specialist", "supervisor")
workflow.add_edge("visualizer", "supervisor")

# Only summarizer terminates
workflow.add_edge("summarizer", END)
```

## Benefits

1. **Centralized Control**: All routing decisions made by supervisor
2. **No Agent Coupling**: Agents are completely independent
3. **Clear Data Flow**: State flows through supervisor at each stage
4. **Easy Debugging**: Single point of control for routing logic
5. **Scalable Architecture**: Easy to add new agents or modify routing

## Testing Scenarios

### Scenario 1: Visualization Requested
```
Query: "Show me a chart of my pension growth"
Flow: Supervisor → projection_specialist → Supervisor → visualizer → Supervisor → summarizer
```

### Scenario 2: No Visualization Needed
```
Query: "How much will my pension be worth?"
Flow: Supervisor → projection_specialist → Supervisor → summarizer
```

### Scenario 3: Risk Analysis with Charts
```
Query: "What's my risk profile? Show it visually"
Flow: Supervisor → risk_analyst → Supervisor → visualizer → Supervisor → summarizer
```

The supervisor intelligently routes each stage based on the current state and available data, ensuring no direct agent communication occurs.
