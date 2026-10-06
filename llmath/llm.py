"""One call interface for local (Ollama) and cloud (LiteLLM) models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from .config import OLLAMA_URL, ModelSpec


@dataclass
class LLMResponse:
    text: str
    reasoning: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    cost_usd: float | None


class LLMError(RuntimeError):
    pass


def chat(spec: ModelSpec, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
    if spec.provider == "ollama":
        return _ollama_chat(spec, messages, temperature)
    if spec.provider == "litellm":
        return _litellm_chat(spec, messages, temperature)
    raise ValueError(f"bilinmeyen sağlayıcı: {spec.provider}")


def _ollama_chat(spec: ModelSpec, messages: list[dict[str, str]], temperature: float) -> LLMResponse:
    payload: dict = {
        "model": spec.model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": spec.max_tokens},
    }
    if spec.think is not None:
        payload["think"] = spec.think
    start = time.perf_counter()
    try:
        response = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMError(f"Ollama isteği başarısız: {exc}") from exc
    data = response.json()
    message = data.get("message") or {}
    return LLMResponse(
        text=message.get("content") or "",
        reasoning=message.get("thinking") or None,
        input_tokens=data.get("prompt_eval_count"),
        output_tokens=data.get("eval_count"),
        latency_ms=int((time.perf_counter() - start) * 1000),
        cost_usd=0.0,
    )


def _litellm_chat(spec: ModelSpec, messages: list[dict[str, str]], temperature: float) -> LLMResponse:
    import litellm  # imported lazily: slow to load and only needed for cloud models

    litellm.drop_params = True  # some models reject parameters such as temperature
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)  # per-call deprecation notices
    start = time.perf_counter()
    try:
        response = litellm.completion(
            model=spec.model, messages=messages, temperature=temperature,
            max_tokens=spec.max_tokens, num_retries=3, timeout=300,
        )
    except Exception as exc:  # LiteLLM raises provider-specific exception types
        raise LLMError(f"{spec.model} isteği başarısız: {exc}") from exc
    message = response.choices[0].message
    usage = getattr(response, "usage", None)
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:  # unknown pricing for very new models
        cost = None
    return LLMResponse(
        text=message.content or "",
        reasoning=getattr(message, "reasoning_content", None),
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        latency_ms=int((time.perf_counter() - start) * 1000),
        cost_usd=cost,
    )


def ollama_models() -> set[str]:
    """Names of models pulled into the local Ollama server (empty if it is down)."""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError:
        return set()
    return {m["name"] for m in response.json().get("models", [])}
