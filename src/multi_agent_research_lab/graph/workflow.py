"""LangGraph workflow skeleton."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Create a LangGraph graph."""
        from langgraph.graph import StateGraph, END
        from multi_agent_research_lab.agents.supervisor import SupervisorAgent
        from multi_agent_research_lab.agents.researcher import ResearcherAgent
        from multi_agent_research_lab.agents.analyst import AnalystAgent
        from multi_agent_research_lab.agents.writer import WriterAgent

        from multi_agent_research_lab.agents.critic import CriticAgent

        workflow = StateGraph(ResearchState)

        # Nodes
        workflow.add_node("supervisor", SupervisorAgent().run)
        workflow.add_node("researcher", ResearcherAgent().run)
        workflow.add_node("analyst", AnalystAgent().run)
        workflow.add_node("writer", WriterAgent().run)
        workflow.add_node("critic", CriticAgent().run)

        # Edges: Workers always return to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        # Supervisor decides where to go
        def route(state: ResearchState) -> str:
            if not state.route_history:
                return "supervisor"
            last_route = state.route_history[-1]
            if last_route == "done":
                return END
            return last_route

        workflow.add_conditional_edges(
            "supervisor",
            route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END
            }
        )

        workflow.set_entry_point("supervisor")
        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        app = self.build()
        final_state = app.invoke(state)
        return ResearchState(**final_state) if isinstance(final_state, dict) else final_state
