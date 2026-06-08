"""ML: LLM provider.

The LLM is used in THREE distinct roles in this project — answer generation,
query rewriting, and as an evaluation judge. All three go through one `LLM`
interface. The mock implementation is rule-based and deterministic so you can
run, test, and reason about the entire agent loop with no API key.
"""

from __future__ import annotations

from typing import Protocol

from config.settings import Settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class MockLLM:
    """Deterministic stand-in.

    It branches on cues in the system prompt so each role returns something
    sensible: rewrites echo the question, the judge returns a parseable verdict,
    and generation stitches together the provided context.
    """

    def complete(self, system: str, user: str) -> str:
        s = system.lower()
        if "rewrite" in s:
            # Pretend to expand the query; deterministic so tests are stable.
            return user.strip().rstrip("?") + " definition requirements details"
        if "judge" in s or "grade" in s:
            # The harness asks yes/no questions (faithfulness, relevance,
            # context sufficiency). Return YES when there is real content to
            # judge, NO when the payload is empty/trivial. Deterministic.
            return "YES" if len(user.strip()) > 40 else "NO"
        # Generation: extractive summary of the first context block.
        marker = "context:"
        ctx = user.lower().split(marker, 1)[-1] if marker in user.lower() else user
        snippet = " ".join(ctx.split()[:60])
        return f"Based on the retrieved sources: {snippet.strip()}"


class OpenAILLM:
    def __init__(self, model: str, api_key: str | None) -> None:
        from openai import OpenAI  # lazy import

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# def build_llm(settings: Settings) -> LLM:
#     if settings.llm_provider == "openai":
#         return OpenAILLM(settings.llm_model, settings.openai_api_key)
#     return MockLLM()

from openai import OpenAI

class OllamaLLM:
    def __init__(self, model: str = "llama3"):
        from openai import OpenAI
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # با این خط، نام مدل را به زور روی llama3 قفل می‌کنیم!
        self.model = "llama3"

    def complete(self, system: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

def build_llm(settings):
    if settings.llm_provider == "mock":
        return MockLLM()

    if settings.llm_provider == "ollama":
        return OllamaLLM(model=settings.llm_model)

    if settings.llm_provider == "openai":
        return OpenAILLM(
            model=settings.llm_model,
            api_key=settings.openai_api_key
        )