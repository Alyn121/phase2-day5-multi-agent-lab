"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.errors import StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion using Cerebras API via ChatOpenAI."""
        from multi_agent_research_lab.core.config import get_settings
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        settings = get_settings()

        # Cerebras is OpenAI-compatible
        llm = ChatOpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=settings.cerebras_api_key,
            model="llama3.1-8b", 
            temperature=0
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        response = llm.invoke(messages)
        
        # Extract token usage
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        
        # Cerebras is currently very cheap/free for many users, but we can keep a placeholder
        cost = 0.0 

        return LLMResponse(
            content=str(response.content),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost
        )
