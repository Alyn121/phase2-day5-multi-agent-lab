"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        from multi_agent_research_lab.core.schemas import AgentResult
        from multi_agent_research_lab.services.llm_client import LLMClient
        
        if not state.analysis_notes:
            return state

        llm = LLMClient()
        system_prompt = """You are a Technical Writer.
Create a comprehensive summary based on the provided research and analysis.
Requirements:
1. Clear headings.
2. Direct citations to sources (e.g., [Title](URL)).
3. Professional and objective tone.
4. Summary of key findings and any remaining uncertainties."""

        user_prompt = f"Query: {state.request.query}\n\nAnalysis Notes:\n{state.analysis_notes}\n\nSources:\n" + \
                     "\n".join([f"- {s.title}: {s.url}" for s in state.sources])
        
        response = llm.complete(system_prompt, user_prompt)
        import logging
        logging.info(
            "Writer generated final answer.",
            extra={"payload": {"final_answer": response.content}, "agent": "writer"}
        )
        state.agent_results.append(AgentResult(
            agent="writer", 
            content=response.content, 
            metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        ))
        state.final_answer = response.content
        return state
