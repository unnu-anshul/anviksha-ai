from langgraph.graph import END, StateGraph

from agent import nodes
from agent.schemas import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze", nodes.analyze)
    g.add_node("classify", nodes.classify)
    g.add_node("evidence", nodes.find_evidence)
    g.add_node("plan", nodes.plan)
    g.add_node("execute", nodes.execute)
    g.add_node("critic", nodes.critic)
    g.add_node("report", nodes.report)

    g.set_entry_point("analyze")
    g.add_edge("analyze", "classify")
    g.add_edge("classify", "evidence")
    g.add_edge("evidence", "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "critic")
    g.add_conditional_edges("critic", nodes.route_after_critic,
                            {"plan": "plan", "report": "report"})
    g.add_edge("report", END)
    return g.compile()
