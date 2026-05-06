"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics], query: str) -> str:
    """Render benchmark metrics to markdown."""
    lines = [
        f"# Benchmark Report: {query}",
        "",
        "| Run Name | Latency | Tokens (In/Out) | Cerebras Cost | OpenAI Est. | Quality | Sources |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        cerebras_cost = f"${item.estimated_cost_usd:.4f}" if item.estimated_cost_usd is not None else "Free"
        openai_cost = f"${item.openai_equivalent_cost_usd:.4f}" if item.openai_equivalent_cost_usd is not None else "N/A"
        quality = f"{item.quality_score:.1f}/10" if item.quality_score is not None else "N/A"
        tokens = f"{item.input_tokens}/{item.output_tokens}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f}s | {tokens} | {cerebras_cost} | {openai_cost} | {quality} | {item.source_count} |")
    
    lines.append("\n## Analysis")
    if len(metrics) >= 2:
        m1, m2 = metrics[0], metrics[1]
        cost_saving = (m2.openai_equivalent_cost_usd or 0) - (m2.estimated_cost_usd or 0)
        lines.append(f"- **Cost Analysis:** Multi-agent with Cerebras saved approximately ${cost_saving:.4f} compared to OpenAI GPT-4o-mini rates.")
        lines.append(f"- **Quality Gain:** Multi-agent achieved a +{(m2.quality_score or 0) - (m1.quality_score or 0):.1f} quality improvement.")

    # Agent Breakdown Section
    for m in metrics:
        if m.agent_breakdown:
            lines.append(f"\n### Agent Breakdown: {m.run_name}")
            lines.append("| Agent | Calls | Tokens | % of Total |")
            lines.append("|---|---:|---:|---:|")
            total_tokens = m.input_tokens + m.output_tokens
            for agent, stats in m.agent_breakdown.items():
                agent_tokens = stats.get("input", 0) + stats.get("output", 0)
                pct = (agent_tokens / total_tokens * 100) if total_tokens > 0 else 0
                lines.append(f"| {agent} | {stats.get('calls', 0)} | {agent_tokens} | {pct:.1f}% |")
                
    # Judge Reasoning Section
    lines.append("\n## Judge Feedback")
    for m in metrics:
        if m.notes:
            lines.append(f"\n### {m.run_name.capitalize()}")
            lines.append(f"{m.notes}")
    
    return "\n".join(lines) + "\n"
