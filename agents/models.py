"""Optional advisory language-model adapter.

Core decisions and numeric measurements never depend on an LLM. When a model
is configured, this module sends only short aggregate diagnostic summaries to
an OpenAI-compatible endpoint and treats every provider failure as a missing
note. Raw waveforms, API credentials and internal stack traces are never sent.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

AGENT_IDS = tuple(f"agent{i}" for i in range(1, 8)) + (
    "azhva_perception",
    "azhva_reasoning",
    "azhva_remediation",
)


def configured_models() -> dict[str, str]:
    """Return configured model IDs for each known agent without exposing keys."""
    return {agent_id: settings.nvidia_models.get(agent_id, "") for agent_id in AGENT_IDS}


def model_configured(agent_id: str) -> bool:
    """Whether an optional LLM call can actually be attempted for this agent."""
    if agent_id.startswith("azhva"):
        return bool(os.getenv("GROQ_API_KEY", "").strip())
    return bool(settings.nvidia_api_key and settings.nvidia_models.get(agent_id, ""))


def _content_text(content: Any) -> str | None:
    """Normalize chat-completion content while rejecting unexpected payloads."""
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
        text = "".join(str(chunk) for chunk in chunks).strip()
        return text or None
    return None


async def ask_model(
    agent_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 256,
    image_base64: str | None = None,
) -> str | None:
    """Request an advisory note, routing Azhva to Groq and Agents 1-7 to NVIDIA.

    Falls back cleanly to None if credentials or endpoints are unavailable.
    """
    if agent_id not in AGENT_IDS:
        logger.warning("Ignoring model request for unknown agent %s", agent_id)
        return None

    # --- AZHVA PIPELINE ROUTED TO REAL GROQ API ---
    if agent_id.startswith("azhva"):
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if not groq_key:
            env_candidates = [
                settings.project_root / ".env",
                settings.project_root.parent / ".env",
            ]
            for ep in env_candidates:
                if ep.exists():
                    with open(ep, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("GROQ_API_KEY="):
                                groq_key = line.split("=", 1)[1].strip()
                                break
                if groq_key:
                    break

        if groq_key and groq_key != "gsk_":
            groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
            models_to_try = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
            safe_max_tokens = min(max(int(max_tokens), 1), 1024)

            for mid in models_to_try:
                payload = {
                    "model": mid,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": safe_max_tokens,
                    "temperature": 0.2,
                }
                headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                }
                try:
                    async with httpx.AsyncClient(timeout=25.0) as client:
                        resp = await client.post(f"{groq_base_url}/chat/completions", headers=headers, json=payload)
                        if resp.status_code == 200:
                            body = resp.json()
                            choices = body.get("choices", [])
                            if choices and isinstance(choices[0], dict):
                                content = choices[0].get("message", {}).get("content")
                                if content:
                                    logger.info("Azhva request successfully answered by Groq model '%s'", mid)
                                    return content.strip()
                except Exception as exc:
                    logger.warning("Groq call to model %s failed (%s); trying fallback...", mid, type(exc).__name__)
                    continue

        logger.info("Groq API not reachable or key absent; using deterministic copilot knowledge base.")
        return None

    # --- NVIDIA AGENTS PIPELINE (Agents 1-7) ---
    model_id = settings.nvidia_models.get(agent_id, "")
    if not settings.nvidia_api_key or not model_id:
        return None

    safe_max_tokens = min(max(int(max_tokens), 1), 1024)
    user_content: Any = user_prompt
    if image_base64:
        user_content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": safe_max_tokens,
        "temperature": 0.1,
    }
    url = f"{settings.nvidia_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.nvidia_timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if not choices or not isinstance(choices[0], dict):
            logger.warning("Advisory model returned no choices for %s", agent_id)
            return None
        message = choices[0].get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        return _content_text(content)
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Advisory model request failed for %s (%s)", agent_id, type(exc).__name__)
        return None


async def close_model_client() -> None:
    """Kept as a lifecycle hook; clients are scoped to calls and need no cleanup."""
    return None
