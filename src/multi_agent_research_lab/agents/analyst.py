"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        import logging
        from multi_agent_research_lab.core.schemas import AgentResult
        from multi_agent_research_lab.services.llm_client import LLMClient
        
        if not state.research_notes:
            return state

        llm = LLMClient()
        system_prompt = """You are a Research Analyst. 
Review the research notes and:
1. Synthesize the key claims.
2. Identify any conflicting information.
3. Highlight gaps that might need more research.
4. Flag any potential biases in the sources."""
        
        user_prompt = f"Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}"
        
        response = llm.complete(system_prompt, user_prompt)
        logging.info(
            "Analyst synthesized research notes.",
            extra={"payload": {"analysis_notes": response.content}, "agent": "analyst"}
        )
        state.agent_results.append(AgentResult(
            agent="analyst", 
            content=response.content, 
            metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        ))
        state.analysis_notes = response.content
        return state
