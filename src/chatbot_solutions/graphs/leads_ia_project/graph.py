from langgraph.graph import StateGraph, END
from .crew_ai_agents.agents_schema import AgentState
from .nodes.nodes import call_model, tool_executor
from .edges.edges import should_continue

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_executor)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")
leads_ia_graph = workflow.compile()
