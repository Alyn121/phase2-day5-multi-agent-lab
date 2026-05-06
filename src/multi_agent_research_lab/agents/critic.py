"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        import logging
        from multi_agent_research_lab.core.schemas import AgentResult
        from multi_agent_research_lab.services.llm_client import LLMClient
        
        if not state.final_answer:
            return state
            
        llm = LLMClient()
        system_prompt = """You are a Quality Critic. 
Evaluate the final answer against the research notes.
If it is accurate and has citations, respond with 'APPROVED'.
If it has errors or missing info, respond with 'REJECTED' and provide specific feedback."""

        user_prompt = f"Final Answer:\n{state.final_answer}\n\nResearch Notes:\n{state.research_notes}"
        
        response = llm.complete(system_prompt, user_prompt)
        logging.info(
            "Critic provided feedback.",
            extra={"payload": {"feedback": response.content}, "agent": "critic"}
        )
        state.agent_results.append(AgentResult(
            agent="critic", 
            content=response.content, 
            metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        ))
        
        # Store feedback in state (using analyst_notes as a temporary synthesis place or a new field)
        # For simplicity in this lab, we'll append to errors or use a trace event
        state.add_trace_event("critic_feedback", {"feedback": response.content})
        
        # We can also add a field to ResearchState if we want to be more formal
        # But let's assume Supervisor will check the trace for approval
        return state
