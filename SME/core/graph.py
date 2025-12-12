from langgraph.graph import StateGraph, END
from . import nodes
from .state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("contextualize", nodes.contextualize_node) 
workflow.add_node("planner", nodes.planner_node)
workflow.add_node("executor", nodes.execute_tools_node)
workflow.add_node("final_response", nodes.final_response_node)

workflow.set_entry_point("contextualize")


workflow.add_edge("contextualize", "planner")
workflow.add_edge("planner", "executor")


workflow.add_conditional_edges(
    "executor",
    nodes.router_node,
    {
        "continue": "executor",
        "final_response": "final_response"
    }
)

workflow.add_edge("final_response", END)


app = workflow.compile(checkpointer=None, name="SMEAgentWorkflow")