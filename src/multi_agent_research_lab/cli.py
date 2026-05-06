"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    console.print(f"[dim]Log Level: {settings.log_level}[/dim]")


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline."""

    _init()
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    
    def runner(q: str) -> ResearchState:
        llm = LLMClient()
        state = ResearchState(request=ResearchQuery(query=q))
        response = llm.complete("You are a Research Assistant. Answer the query.", q)
        from multi_agent_research_lab.core.schemas import AgentResult
        state.agent_results.append(AgentResult(
            agent="researcher", 
            content=response.content, 
            metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        ))
        state.final_answer = response.content
        return state

    state, metrics = run_benchmark("baseline", query, runner)
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline Result"))
    console.print(f"Latency: {metrics.latency_seconds:.2f}s")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    
    def runner(q: str) -> ResearchState:
        workflow = MultiAgentWorkflow()
        state = ResearchState(request=ResearchQuery(query=q))
        return workflow.run(state)

    try:
        state, metrics = run_benchmark("multi-agent", query, runner)
        console.print(Panel.fit(state.final_answer or "No answer.", title="Multi-Agent Result"))
        console.print(f"Latency: {metrics.latency_seconds:.2f}s")
        console.print(f"Iterations: {state.iteration}")
        console.print(f"Sources: {len(state.sources)}")
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc


@app.command()
def compare(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[str, typer.Option("--output", "-o", help="Report file path")] = "reports/benchmark_report.md",
) -> None:
    """Run both baseline and multi-agent and generate a comparison report."""
    _init()
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report
    import os

    all_metrics = []
    
    # 1. Baseline
    console.print("[bold blue]Running Baseline...[/bold blue]")
    def baseline_runner(q: str) -> ResearchState:
        from multi_agent_research_lab.services.llm_client import LLMClient
        llm = LLMClient()
        state = ResearchState(request=ResearchQuery(query=q))
        response = llm.complete("You are a Research Assistant. Answer concisely.", q)
        from multi_agent_research_lab.core.schemas import AgentResult
        state.agent_results.append(AgentResult(
            agent="researcher", 
            content=response.content, 
            metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        ))
        state.final_answer = response.content
        return state
    _, m_baseline = run_benchmark("baseline", query, baseline_runner)
    all_metrics.append(m_baseline)

    # 2. Multi-Agent
    console.print("[bold green]Running Multi-Agent...[/bold green]")
    def multi_runner(q: str) -> ResearchState:
        workflow = MultiAgentWorkflow()
        state = ResearchState(request=ResearchQuery(query=q))
        return workflow.run(state)
    state_multi, m_multi = run_benchmark("multi-agent", query, multi_runner)
    all_metrics.append(m_multi)

    # 3. Render and Save
    from rich.markdown import Markdown
    from rich.table import Table

    report_md = render_markdown_report(all_metrics, query)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report_md)
    
    # BEAUTIFUL TERMINAL DISPLAY
    console.print("\n" + "="*50)
    console.print("[bold cyan]RESEARCH REPORT[/bold cyan]")
    console.print("="*50 + "\n")
    
    # Display the multi-agent final answer beautifully
    if state_multi.final_answer:
        console.print(Panel(Markdown(state_multi.final_answer), title="Final Answer (Markdown Rendering)", border_style="green"))
    
    # Display comparison table
    table = Table(title="Benchmark Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Baseline", justify="right")
    table.add_column("Multi-Agent", justify="right")
    table.add_column("Improvement", justify="center")

    table.add_row(
        "Latency", 
        f"{m_baseline.latency_seconds:.2f}s", 
        f"{m_multi.latency_seconds:.2f}s",
        f"[red]+{m_multi.latency_seconds - m_baseline.latency_seconds:.1f}s[/red]"
    )
    table.add_row(
        "Quality (0-10)", 
        f"{m_baseline.quality_score:.1f}", 
        f"{m_multi.quality_score:.1f}",
        f"[green]+{m_multi.quality_score - m_baseline.quality_score:.1f}[/green]"
    )
    table.add_row(
        "Tokens (In/Out)", 
        f"{m_baseline.input_tokens}/{m_baseline.output_tokens}", 
        f"{m_multi.input_tokens}/{m_multi.output_tokens}",
        ""
    )
    table.add_row(
        "OpenAI Est. Cost", 
        f"${m_baseline.openai_equivalent_cost_usd:.4f}", 
        f"${m_multi.openai_equivalent_cost_usd:.4f}",
        f"[yellow]Saved vs OpenAI[/yellow]"
    )
    
    console.print(table)

    # Agent breakdown table
    if m_multi.agent_breakdown:
        agent_table = Table(title="Agent Token Distribution (%)", show_header=True, header_style="bold blue")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Calls", justify="center")
        agent_table.add_column("Tokens", justify="right")
        agent_table.add_column("Share (%)", justify="right")

        total_tokens = m_multi.input_tokens + m_multi.output_tokens
        for agent, stats in m_multi.agent_breakdown.items():
            agent_tokens = stats.get("input", 0) + stats.get("output", 0)
            pct = (agent_tokens / total_tokens * 100) if total_tokens > 0 else 0
            agent_table.add_row(
                agent.capitalize(),
                str(stats.get("calls", 0)),
                str(agent_tokens),
                f"{pct:.1f}%"
            )
        console.print(agent_table)

    if m_multi.notes:
        console.print("\n[bold yellow]Multi-Agent Quality Feedback:[/bold yellow]")
        console.print(m_multi.notes)

    console.print(f"\n[bold green]✔ Report saved to {output}[/bold green]")
    console.print("[bold blue]✔ JSON Trace saved to logs/execution_trace.json[/bold blue]\n")


if __name__ == "__main__":
    app()
