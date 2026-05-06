"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        import logging
        from multi_agent_research_lab.core.schemas import AgentResult
        from multi_agent_research_lab.services.llm_client import LLMClient
        
        # Guardrail: stop after 7 iterations (increased for critic loop)
        if state.iteration >= 7:
            state.record_route("done")
            return state

        llm = LLMClient()
        system_prompt = """You are the Research Supervisor. Decide the next step.
Available agents:
- researcher: Gathers facts.
- analyst: Synthesizes facts.
- writer: Writes the final answer.
- critic: Validates the writer's work for accuracy and citations.
- done: Finish only if the critic has APPROVED or max iterations reached.

Routing Rules (Evaluate in order, stop at the first matching rule):
1. FIRST check: If research_notes is empty -> 'researcher'.
2. SECOND check: If analysis_notes is empty -> 'analyst'.
3. THIRD check: If final_answer is empty -> 'writer'.
4. FOURTH check: If final_answer exists but hasn't been checked by critic -> 'critic'.
5. FIFTH check: If critic REJECTED -> 'writer' (to fix).
6. SIXTH check: If critic APPROVED -> 'done'.

You must respond with EXACTLY ONE WORD from the list of agents, based on the FIRST rule that matches."""

        # Check for critic feedback in trace
        critic_approved = False
        critic_rejected = False
        for event in reversed(state.trace):
            if event["name"] == "critic_feedback":
                feedback = event["payload"]["feedback"]
                if "APPROVED" in feedback.upper():
                    critic_approved = True
                if "REJECTED" in feedback.upper():
                    critic_rejected = True
                break

        user_prompt = f"""Query: {state.request.query}
History: {state.route_history}

State:
- research_notes is empty: {not bool(state.research_notes)}
- analysis_notes is empty: {not bool(state.analysis_notes)}
- final_answer is empty: {not bool(state.final_answer)}
- critic status: {"APPROVED" if critic_approved else "REJECTED" if critic_rejected else "NOT_CHECKED"}

What is the next route?"""

        response = llm.complete(system_prompt, user_prompt)
        state.agent_results.append(AgentResult(
            agent="supervisor", 
            content=response.content, 
            metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        ))
        raw_route = response.content.strip().lower()
        print(f"\n[DEBUG Supervisor] Raw Response: {raw_route}")
        
        # 1. Smart fallback based on current state (ensures progress even if LLM fails)
        if not state.research_notes:
            next_route = "researcher"
        elif not state.analysis_notes:
            next_route = "analyst"
        elif not state.final_answer:
            next_route = "writer"
        elif not critic_approved and not critic_rejected:
            next_route = "critic"
        elif critic_rejected:
            next_route = "writer"
        else:
            next_route = "done"

        # 2. Let the LLM override if it correctly chose a valid agent
        # Split by non-alphanumeric to get clean words
        import re
        words = re.findall(r'\b\w+\b', raw_route)
        
        # Check from highest priority to lowest priority
        for valid_route in ["done", "critic", "writer", "analyst", "researcher"]:
            if valid_route in words:
                next_route = valid_route
                break

        print(f"[DEBUG Supervisor] Decided Route: {next_route}\n")
        logging.info(
            f"Supervisor decided next route: {next_route}",
            extra={"payload": {"decided_route": next_route, "raw_llm": raw_route, "history": state.route_history}, "agent": "supervisor"}
        )
        state.record_route(next_route)
        return state
