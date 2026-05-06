"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query using Tavily."""
        from multi_agent_research_lab.core.config import get_settings
        from langchain_community.tools.tavily_search import TavilySearchResults

        settings = get_settings()
        search_tool = TavilySearchResults(
            max_results=max_results,
            tavily_api_key=settings.tavily_api_key
        )
        results = search_tool.invoke({"query": query})
        print(f"\n[DEBUG] Query: {query}")
        print(f"[DEBUG] Raw Results: {results}\n")
        
        # Robust parsing for list or JSON string
        if isinstance(results, str):
            import json
            try:
                results = json.loads(results)
            except Exception:
                results = []

        documents = []
        if isinstance(results, list):
            for res in results:
                documents.append(SourceDocument(
                    title=res.get("title", "Search Result"),
                    url=res.get("url", ""),
                    snippet=res.get("content", res.get("snippet", ""))
                ))
        return documents
