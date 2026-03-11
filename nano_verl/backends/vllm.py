"""vLLM rollout backends."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from time import perf_counter
from urllib import error, request

from nano_verl.backends.base import RolloutBackend, RolloutRequest
from nano_verl.types import RolloutCandidate, RolloutSample


@dataclass(slots=True)
class VLLMServerConfig:
    """Connection settings for a vLLM OpenAI-compatible server."""

    base_url: str
    api_key: str = "EMPTY"


class VLLMServerRolloutBackend(RolloutBackend):
    """Rollout backend that talks to vLLM's OpenAI-compatible chat endpoint."""

    backend_name = "vllm-server"

    def __init__(self, config: VLLMServerConfig) -> None:
        self.config = config

    async def generate(self, request_payload: RolloutRequest) -> RolloutSample:
        return await asyncio.to_thread(self._generate_sync, request_payload)

    def _generate_sync(self, request_payload: RolloutRequest) -> RolloutSample:
        if not request_payload.model_name:
            raise ValueError("vLLM rollout requires a model name.")

        started_at = perf_counter()
        body = {
            "model": request_payload.model_name,
            "messages": [{"role": "user", "content": request_payload.prompt.prompt}],
            "n": request_payload.num_samples,
            "temperature": request_payload.temperature,
            "top_p": request_payload.top_p,
            "max_tokens": request_payload.max_tokens,
            "stream": False,
        }

        if request_payload.seed is not None:
            body["seed"] = request_payload.seed

        raw_response = self._post_json(
            url=f"{self.config.base_url.rstrip('/')}/v1/chat/completions",
            payload=body,
            timeout_s=request_payload.timeout_s,
        )
        elapsed_ms = round((perf_counter() - started_at) * 1000.0, 2)
        choices = raw_response.get("choices", [])

        candidates: list[RolloutCandidate] = []
        for sample_index, choice in enumerate(choices):
            text = _extract_choice_text(choice)
            token_count = _extract_completion_tokens(choice, text)
            candidates.append(
                RolloutCandidate(
                    sample_id=f"{request_payload.prompt.prompt_id}-cand-{sample_index:02d}",
                    prompt_id=request_payload.prompt.prompt_id,
                    text=text,
                    token_count=token_count,
                    latency_ms=elapsed_ms,
                    metadata={
                        "backend": self.backend_name,
                        "finish_reason": choice.get("finish_reason"),
                        "index": choice.get("index", sample_index),
                    },
                )
            )

        return RolloutSample(prompt=request_payload.prompt, candidates=candidates)

    def _post_json(self, url: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM server request failed with status {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach vLLM server at {url}: {exc.reason}") from exc


def _extract_choice_text(choice: dict[str, object]) -> str:
    message = choice.get("message", {})
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item.get("content", ""))))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
    text = choice.get("text", "")
    return str(text)


def _extract_completion_tokens(choice: dict[str, object], text: str) -> int:
    usage = choice.get("usage", {})
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        return max(1, int(usage["completion_tokens"]))
    return max(1, len(text.split()))

