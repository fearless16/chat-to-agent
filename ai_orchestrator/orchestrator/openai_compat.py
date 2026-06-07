"""OpenAI-compatible API — ``/v1/chat/completions`` and ``/v1/models``.

This router exposes every registered provider adapter as an OpenAI-compatible
model.  Tools like OpenCode, Cursor, Continue, or any OpenAI SDK client can
point their ``base_url`` at ``http://localhost:8000/v1`` and use browser-based
chat UIs (ChatGPT, DeepSeek, Qwen, Kimi …) as inference backends.

Streaming emits ``reasoning_content`` delta chunks (when the provider supplies
reasoning) followed by ``content`` delta chunks, matching the OpenAI streaming
spec for reasoning models (o1, o3, DeepSeek-R1, etc.).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["OpenAI-compatible"])

# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | None = ""
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    user: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str = ""
    reasoning_content: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "browser"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Helpers — import build_adapter + provider map from main at call time
# to avoid circular imports.
# ---------------------------------------------------------------------------


def _get_provider_map() -> dict[str, type[ProviderAdapter]]:
    from ai_orchestrator.orchestrator.main import _PROVIDER_CLASS_MAP
    return {k: v for k, v in _PROVIDER_CLASS_MAP.items() if k != "local_llm"}


def _build_adapter(provider: str, mock_mode: bool) -> ProviderAdapter:
    from ai_orchestrator.orchestrator.main import _build_adapter as builder
    return builder(provider, mock_mode)


def _gen_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List all available models (providers) in OpenAI format."""
    provider_map = _get_provider_map()
    data = []
    for name in sorted(provider_map):
        data.append(ModelInfo(id=name, owned_by="browser"))
    return ModelListResponse(data=data)


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
    """OpenAI-compatible chat completions endpoint.

    - ``model`` maps to a provider name (chatgpt_ui, local_llm, etc.)
    - ``stream=true`` returns Server-Sent Events with delta chunks
    - ``stream=false`` (default) returns a single JSON response
    - ``reasoning_content`` is included when the provider supplies it
    """
    provider_map = _get_provider_map()
    if req.model not in provider_map:
        raise HTTPException(
            status_code=400,
            detail=f"Model {req.model!r} not found. Available: {sorted(provider_map)}",
        )

    # Extract prompt: last user message. Pass full history as context.
    prompt = ""
    context: list[dict] = []
    for msg in req.messages:
        context.append({"role": msg.role, "content": msg.content or ""})
        if msg.role == "user":
            prompt = msg.content or ""

    if not prompt:
        raise HTTPException(status_code=400, detail="No user message found in messages")

    # Default to real mode (mock_mode=False) — this is an inference provider.
    # Allow override via query param ?mock=true for testing.
    mock_mode = request.query_params.get("mock", "false").lower() in ("true", "1", "yes")

    adapter = _build_adapter(req.model, mock_mode)

    if req.stream:
        return StreamingResponse(
            _stream_response(adapter, prompt, context, req.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming path
    try:
        resp = await adapter.send(prompt, context)
    except Exception as exc:
        resp = ProviderResponse(success=False, error=str(exc), model=req.model)
    finally:
        try:
            await adapter.close()
        except Exception:
            pass

    if not resp.success:
        raise HTTPException(status_code=502, detail=resp.error or "Provider error")

    completion_id = _gen_id()
    created = int(time.time())

    content = resp.content or ""
    # Rough token estimation (1 token ≈ 4 chars)
    prompt_tokens = sum(len(m.content or "") for m in req.messages) // 4
    completion_tokens = len(content) // 4
    if resp.reasoning_content:
        completion_tokens += len(resp.reasoning_content) // 4

    message = ChoiceMessage(content=content)
    if resp.reasoning_content:
        message.reasoning_content = resp.reasoning_content

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=req.model,
        choices=[
            Choice(
                message=message,
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# SSE streaming generator
# ---------------------------------------------------------------------------


async def _stream_response(
    adapter: ProviderAdapter,
    prompt: str,
    context: list[dict],
    model: str,
) -> Any:
    """Yield SSE chunks in OpenAI streaming format.

    Reasoning content (if any) is emitted first as ``reasoning_content``
    delta chunks, followed by regular ``content`` delta chunks.  This
    matches the OpenAI streaming spec for reasoning models.

    Failing adapters raise ``HTTPException(502)`` instead of streaming
    error text as content.
    """
    completion_id = _gen_id()
    created = int(time.time())

    # Get the full response from the adapter BEFORE emitting any SSE
    try:
        resp = await adapter.send(prompt, context)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        if adapter:
            try:
                await adapter.close()
            except Exception:
                pass

    if not resp.success:
        raise HTTPException(status_code=502, detail=resp.error or "Provider error")

    # First chunk: role declaration
    yield _sse_chunk(completion_id, created, model, delta={"role": "assistant"})

    # Phase 1: emit reasoning_content deltas (if any)
    reasoning = resp.reasoning_content or ""
    if reasoning:
        reasoning_words = reasoning.split(" ")
        for i, word in enumerate(reasoning_words):
            token = f" {word}" if i > 0 else word
            yield _sse_chunk(
                completion_id, created, model,
                delta={"reasoning_content": token},
            )
            await asyncio.sleep(0.02)

    # Phase 2: emit content deltas
    content = resp.content or ""
    words = content.split(" ")

    for i, word in enumerate(words):
        token = f" {word}" if i > 0 else word
        yield _sse_chunk(
            completion_id, created, model,
            delta={"content": token},
        )
        await asyncio.sleep(0.02)

    # Final chunk with finish_reason
    yield _sse_chunk(
        completion_id, created, model,
        delta={}, finish_reason="stop",
    )
    yield "data: [DONE]\n\n"


def _sse_chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"
