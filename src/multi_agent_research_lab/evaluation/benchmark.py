"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return benchmark metrics."""
    from multi_agent_research_lab.services.llm_client import LLMClient

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    # 1. LLM-as-a-judge for Quality Score
    quality_score = 0.0
    judge_reasoning = ""
    if state.final_answer:
        llm = LLMClient()
        judge_system = """You are an Expert AI Judge evaluating a research answer.
Evaluate the answer based on these criteria (score 0-10):
1. accuracy: Is the information factually correct based on the query?
2. citations (CRITICAL): Does the answer explicitly include URLs or specific source names? Give a maximum of 4.0 if there are NO explicit URLs or citations. Give 9-10 if URLs are provided and well-integrated.
3. depth: Does it cover the topic thoroughly with synthesis from multiple angles?
4. presentation: Is the structure and tone professional?

You MUST respond with a valid JSON object EXACTLY like this:
{
  "accuracy": 8.5,
  "citations": 7.0,
  "depth": 9.0,
  "presentation": 8.0,
  "reasoning": "Brief explanation of the scores."
}"""
        judge_user = f"Query: {query}\n\nAnswer:\n{state.final_answer}"
        try:
            score_res = llm.complete(judge_system, judge_user)
            import json
            import re
            
            # Extract JSON block if surrounded by markdown
            content = score_res.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                
            data = json.loads(content)
            
            # Calculate average score
            scores = [data.get(k, 5.0) for k in ["accuracy", "citations", "depth", "presentation"]]
            quality_score = sum(scores) / len(scores)
            
            # Format reasoning for the report
            judge_reasoning = f"**Judge Reasoning:** {data.get('reasoning', '')} " \
                              f"(Acc: {data.get('accuracy')}, Cit: {data.get('citations')}, " \
                              f"Dep: {data.get('depth')}, Pre: {data.get('presentation')})"
                              
        except Exception as e:
            quality_score = 5.0 # Default if parsing fails
            judge_reasoning = f"Judge failed to parse score: {str(e)}"

    # 2. Extract detailed metrics from state
    input_tokens = state.total_input_tokens
    output_tokens = state.total_output_tokens
    source_count = len(state.sources)
    
    # 3. Estimate cost
    # Cerebras is currently treated as free in this calculation
    cerebras_cost = 0.0 
    
    # OpenAI gpt-4o-mini equivalent cost
    openai_cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
    
    # 4. Agent Breakdown
    agent_breakdown = {}
    for res in state.agent_results:
        agent_name = res.agent.value if hasattr(res.agent, "value") else str(res.agent)
        if agent_name not in agent_breakdown:
            agent_breakdown[agent_name] = {"calls": 0, "input": 0, "output": 0}
        agent_breakdown[agent_name]["calls"] += 1
        agent_breakdown[agent_name]["input"] += res.metadata.get("input_tokens", 0)
        agent_breakdown[agent_name]["output"] += res.metadata.get("output_tokens", 0)

    metrics = BenchmarkMetrics(
        run_name=run_name, 
        latency_seconds=latency,
        estimated_cost_usd=cerebras_cost,
        openai_equivalent_cost_usd=openai_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        source_count=source_count,
        quality_score=quality_score,
        agent_breakdown=agent_breakdown,
        notes=judge_reasoning
    )
    return state, metrics
