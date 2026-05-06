"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        import logging
        from multi_agent_research_lab.core.schemas import AgentResult
        from multi_agent_research_lab.services.llm_client import LLMClient
        from multi_agent_research_lab.services.search_client import SearchClient
        
        search_client = SearchClient()
        llm = LLMClient()
        
        # 1. Search
        sources = search_client.search(state.request.query)
        logging.info(
            f"Researcher found {len(sources)} sources.",
            extra={"payload": {"sources": [s.dict() for s in sources]}, "agent": "researcher"}
        )
        state.sources.extend(sources)
        
        # 2. Extract notes
        context = "\n\n".join([f"Source: {s.title} ({s.url})\nContent: {s.snippet}" for s in sources])
        system_prompt = "You are a Research Assistant. Extract key facts and information from the search results. Use bullet points."
        user_prompt = f"Query: {state.request.query}\n\nSearch Results:\n{context}"
        
        response = llm.complete(system_prompt, user_prompt)
        state.agent_results.append(AgentResult(
            agent="researcher", 
            content=response.content, 
            metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        ))
        state.research_notes = (state.research_notes or "") + "\n" + response.content
        return state
