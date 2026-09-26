"""OpenAI-compatible JSON calls for online play and offline reflection."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ChatJSONClient:
    def __init__(self, *, model: str, api_key_env: str = "OPENAI_API_KEY",
                 base_url: str | None = None, json_mode: bool = True):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"set {api_key_env} before making model calls")
        if not model:
            raise ValueError("a chat model name is required")
        self.model = model
        self.json_mode = json_mode
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def complete_json(self, *, stage: str, system: str, prompt: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system + "\nReturn one valid JSON object."},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(3):
            body: dict[str, Any] = {"model": self.model, "messages": messages}
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            request = Request(
                self.base_url + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": "Bearer " + self.api_key,
                         "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=120) as response:
                    reply = json.load(response)
            except HTTPError as error:
                raise RuntimeError(f"{stage}: chat endpoint returned HTTP {error.code}") from None
            try:
                choice = reply["choices"][0]
                content = choice["message"]["content"]
                complete = choice.get("finish_reason") == "stop"
            except (KeyError, IndexError, TypeError):
                content, complete = None, False
            if complete and isinstance(content, str):
                try:
                    result = json.loads(content)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
            if attempt < 2:
                messages.append({"role": "user", "content":
                                 "The answer was invalid or incomplete. Return one complete JSON object."})
        raise ValueError(f"{stage} did not return a complete JSON object")
