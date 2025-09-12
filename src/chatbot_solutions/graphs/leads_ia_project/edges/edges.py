from ..crew_ai_agents.agents_schema import AgentState

def should_continue(state: AgentState)-> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"

