from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class HuggingFaceClient:
    """Cliente que carga un modelo de HuggingFace localmente con transformers.

    Variables de entorno:
        HF_MODEL      - model ID en HuggingFace Hub (default: Qwen/Qwen2.5-3B-Instruct)
        HF_CACHE_DIR  - directorio de caché para modelos descargados (opcional)
    """

    _instances: dict[str, "HuggingFaceClient"] = {}

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.model_id = model_id
        self.quantization = os.getenv("HF_QUANTIZATION", "none").lower()
        cache_dir = os.getenv("HF_CACHE_DIR") or None

        tokenizer_kwargs: dict[str, Any] = {"cache_dir": cache_dir}

        if "mistral" in model_id.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True

        model_kwargs: dict[str, Any] = {
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }

        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16

        if self.quantization == "bnb4":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        import torch

        with torch.inference_mode():
            outputs = self.pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.empty_cache()

        return str(outputs[0]["generated_text"])


_ollama_instances: dict[str, OllamaClient] = {}


def get_llm_client(*, model: Optional[str] = None) -> OllamaClient | HuggingFaceClient:
    """Devuelve un cliente LLM según la variable de entorno LLM_BACKEND.

    LLM_BACKEND=ollama      → OllamaClient (default local)
    LLM_BACKEND=huggingface → HuggingFaceClient con modelo de HF Hub
    """
    backend_dir = Path(__file__).resolve().parents[3]
    load_dotenv(backend_dir / ".env")

    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "huggingface":
        model_id = model or os.getenv("HF_MODEL") or "Qwen/Qwen2.5-3B-Instruct"
        if model_id not in HuggingFaceClient._instances:
            HuggingFaceClient._instances[model_id] = HuggingFaceClient(model_id)
        return HuggingFaceClient._instances[model_id]

    # Default: Ollama
    resolved_model = model or os.getenv("OLLAMA_MODEL") or "qwen2.5:3b-instruct"
    if resolved_model not in _ollama_instances:
        _ollama_instances[resolved_model] = OllamaClient(model=resolved_model)
    return _ollama_instances[resolved_model]
