from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


class OllamaClient:
    def __init__(self, model, base_url: Optional[str] = None):
        backend_dir = Path(__file__).resolve().parents[3]
        load_dotenv(backend_dir / ".env")
        resolved_base_url = base_url if base_url is not None else os.getenv("OLLAMA_BASE_URL")
        if resolved_base_url is None:
            resolved_base_url = "http://localhost:11434"
        self.url = f"{resolved_base_url.rstrip('/')}/api/chat"
        self.model = model

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = requests.post(self.url, json=payload, timeout=120)
        r.raise_for_status()
        return str(r.json()["message"]["content"])


_ollama_instances: dict[str, OllamaClient] = {}


def get_llm_client(*, model: Optional[str] = None) -> OllamaClient:
    backend_dir = Path(__file__).resolve().parents[3]
    load_dotenv(backend_dir / ".env")

    resolved_model = model or os.getenv("OLLAMA_MODEL") or "qwen2.5:3b-instruct"

    if resolved_model not in _ollama_instances:
        _ollama_instances[resolved_model] = OllamaClient(model=resolved_model)

    return _ollama_instances[resolved_model]
