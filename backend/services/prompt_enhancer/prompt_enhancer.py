"""Local prompt enhancement via OpenAI-compatible API (e.g. LM Studio)."""

from __future__ import annotations

import logging

from services.http_client.http_client import HTTPClient, HttpTimeoutError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a video generation prompt enhancer. Given a short user prompt, rewrite it \
into a detailed, vivid description suitable for an AI video generator. Add specific \
visual details, lighting, camera movement, atmosphere, colors, textures, and motion \
cues. Keep the enhanced prompt to 2-3 sentences. Do NOT add any preamble, \
explanation, or formatting — output ONLY the enhanced prompt text."""


def enhance_prompt(
    http: HTTPClient,
    endpoint: str,
    model: str,
    prompt: str,
) -> str | None:
    """Call a local OpenAI-compatible endpoint to enhance a prompt.

    Returns the enhanced prompt string, or None on failure (caller should
    fall back to the original prompt).
    """
    if not endpoint or not model:
        return None

    url = f"{endpoint.rstrip('/')}/chat/completions"

    try:
        response = http.post(
            url,
            headers={"Content-Type": "application/json"},
            json_payload={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.7,
            },
            timeout=30,
        )

        if response.status_code != 200:
            logger.warning("Prompt enhancer API error %s: %s", response.status_code, response.text)
            return None

        data = response.json()
        if not isinstance(data, dict):
            return None

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            return None

        message = choices[0].get("message", {})
        content = message.get("content", "")
        enhanced = content.strip()

        if enhanced:
            logger.info("Prompt enhanced: %r -> %r", prompt[:60], enhanced[:60])
            return enhanced
        return None

    except HttpTimeoutError:
        logger.warning("Prompt enhancer timed out")
        return None
    except Exception as exc:
        logger.warning("Prompt enhancer failed: %s", exc, exc_info=True)
        return None
