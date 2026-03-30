"""LLM client — thin wrapper around LiteLLM + Instructor for structured output.

Provides a single `LLMClient` that routes to any provider (Anthropic, OpenAI,
Google, Ollama, etc.) and returns validated Pydantic models when requested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

import instructor
import litellm
from pydantic import BaseModel

# SECURITY: Pin litellm to <=1.82.6 in pyproject.toml.
# Versions 1.82.7 and 1.82.8 were compromised in a supply chain attack
# (March 2026). See https://docs.litellm.ai/blog/security-update-march-2026

# Suppress LiteLLM's noisy logging by default
litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class CompletionResult:
    """Result from an LLM call — contains either raw text or structured output."""

    content: str = ""
    structured: BaseModel | None = None
    usage: Usage = field(default_factory=Usage)


# ── Provider routing ─────────────────────────────────────────────────────────

# Maps user-friendly provider names to LiteLLM model prefixes.
# The user picks a provider + model in the UI; we combine them into
# the LiteLLM format: "{prefix}{model}" (e.g. "anthropic/claude-sonnet-4-6").
PROVIDER_PREFIXES: dict[str, str] = {
    "anthropic": "anthropic/",
    "openai": "",           # OpenAI models have no prefix in LiteLLM
    "google": "gemini/",
    "ollama": "ollama/",
    "openrouter": "openrouter/",
    "mock": "",             # testing — no API calls
}

# Maps provider names to the env var that holds the API key.
PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    # ollama doesn't need a key
}

# Default models per provider (design / generate).
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {"design": "claude-opus-4-6", "generate": "claude-sonnet-4-6"},
    "openai": {"design": "gpt-5.4", "generate": "gpt-5.4"},
    "google": {"design": "gemini-2.5-pro", "generate": "gemini-2.5-flash"},
    "ollama": {"design": "llama3", "generate": "llama3"},
    "openrouter": {"design": "anthropic/claude-opus-4-6", "generate": "anthropic/claude-sonnet-4-6"},
    "mock": {"design": "mock", "generate": "mock"},
}


class QuotaExhaustedError(Exception):
    """Raised when the API returns a billing/quota error — no point retrying."""
    pass


def resolve_model(provider: str, model: str) -> str:
    """Turn a provider + short model name into a LiteLLM model string."""
    prefix = PROVIDER_PREFIXES.get(provider, "")
    # If the model already contains the prefix or a slash, use as-is
    if "/" in model and model.startswith(prefix):
        return model
    return f"{prefix}{model}"


# ── LLM Client ───────────────────────────────────────────────────────────────


class LLMClient:
    """Unified LLM client backed by LiteLLM + Instructor.

    Usage:
        client = LLMClient(provider="anthropic", api_key="sk-...")

        # Raw completion
        result = await client.complete(messages=[...], model="claude-sonnet-4-6")

        # Structured output (returns validated Pydantic model)
        analysis = await client.complete(
            messages=[...],
            model="claude-opus-4-6",
            response_model=Analysis,
        )
    """

    def __init__(self, provider: str = "anthropic", api_key: str | None = None):
        self.provider = provider
        self._setup_api_key(api_key)

        # Pick Instructor mode based on provider — newer OpenAI models
        # don't support tool_choice.function, use JSON_SCHEMA instead
        mode = {
            "anthropic": instructor.Mode.TOOLS,
            "openai": instructor.Mode.JSON_SCHEMA,
            "google": instructor.Mode.JSON,
            "ollama": instructor.Mode.JSON,
            "openrouter": instructor.Mode.TOOLS,
        }.get(provider, instructor.Mode.JSON)

        self._instructor = instructor.from_litellm(
            litellm.acompletion, mode=mode
        )

        # Cumulative usage tracking across all calls
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_calls: int = 0

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r})"

    def _setup_api_key(self, api_key: str | None) -> None:
        """Set the API key in the environment for LiteLLM to pick up.

        The key is NOT stored on the instance — only in os.environ where
        LiteLLM reads it. This prevents accidental exposure via repr/logging.
        """
        if not api_key:
            return
        env_var = PROVIDER_ENV_VARS.get(self.provider)
        if env_var:
            os.environ[env_var] = api_key

    async def complete(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        response_model: type[T] | None = None,
        max_tokens: int = 16384,
        max_retries: int = 2,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Make a single LLM call, optionally with structured output.

        If response_model is provided, Instructor handles:
          - Sending the JSON schema to the LLM
          - Parsing the response into a Pydantic model
          - Retrying on validation failure (feeds error back to model)

        Returns a CompletionResult with either .content (raw) or
        .structured (validated Pydantic model).
        """
        full_model = resolve_model(self.provider, model)

        # Build kwargs common to both paths
        kwargs: dict = {
            "model": full_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Anthropic uses a top-level system param; others use a system message.
        # LiteLLM handles this translation, but we need to pass it correctly.
        if system:
            if self.provider == "anthropic":
                # LiteLLM passes this through to the Anthropic API
                kwargs["system"] = system
            else:
                # Prepend as a system message for OpenAI-compatible APIs
                kwargs["messages"] = [
                    {"role": "system", "content": system},
                    *messages,
                ]

        try:
            if response_model is not None:
                return await self._structured_call(kwargs, response_model, max_retries)
            else:
                return await self._raw_call(kwargs)
        except Exception as e:
            self._check_quota_error(e)
            raise

    @staticmethod
    def _check_quota_error(exc: Exception) -> None:
        """Detect billing/quota errors and raise QuotaExhaustedError.

        These should never be retried — the wallet is empty.
        """
        err_str = str(exc).lower()
        if any(phrase in err_str for phrase in [
            "exceeded your current quota",
            "insufficient_quota",
            "billing",
            "payment required",
        ]):
            raise QuotaExhaustedError(
                "API quota exhausted — check your billing at the provider's dashboard."
            ) from exc

    async def _raw_call(self, kwargs: dict) -> CompletionResult:
        """Plain text completion."""
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content or ""
        usage = self._extract_usage(response, kwargs.get("model", ""))
        self._track(usage)
        return CompletionResult(content=content, usage=usage)

    async def _structured_call(
        self,
        kwargs: dict,
        response_model: type[T],
        max_retries: int,
    ) -> CompletionResult:
        """Structured output via Instructor — returns validated Pydantic model."""
        try:
            result, raw_response = await self._instructor.create_with_completion(
                response_model=response_model,
                max_retries=max_retries,
                **kwargs,
            )
        except Exception as e:
            self._check_quota_error(e)
            raise
        # Extract usage from the raw LiteLLM response
        usage = self._extract_usage(raw_response, kwargs.get("model", ""))
        self._track(usage)
        return CompletionResult(
            content=result.model_dump_json(indent=2),
            structured=result,
            usage=usage,
        )

    def _track(self, usage: Usage) -> None:
        """Accumulate usage from a single call into running totals."""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        if usage.cost_usd is not None:
            self.total_cost_usd += usage.cost_usd
        self.total_calls += 1

    def get_totals(self) -> dict:
        """Return cumulative usage stats for the entire session."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost_usd": self.total_cost_usd,
            "api_calls": self.total_calls,
        }

    def _extract_usage(self, response: Any, model: str) -> Usage:
        """Pull token counts and cost from a LiteLLM response."""
        u = getattr(response, "usage", None)
        if u is None:
            return Usage()

        input_tokens = getattr(u, "prompt_tokens", 0) or 0
        output_tokens = getattr(u, "completion_tokens", 0) or 0

        # LiteLLM can estimate cost
        cost = None
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
