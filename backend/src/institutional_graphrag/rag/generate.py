# src/rag/generate.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import requests  # type: ignore
import torch
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoModelForCausalLM, AutoTokenizer

from institutional_graphrag.ingest.embedder import E5Embedder
from institutional_graphrag.retrieval.vector_store import VectorStore

BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_EMBEDDINGS_DIR = BASE_DIR / Path("data/embeddings")

_llm_instances: dict[str, LLMClient] = {}


@dataclass
class RAGChunk:
    id: str
    semantic_id: str
    text: str
    score: float
    source: str


@dataclass
class RAGResult:
    answer: str
    contexts: List[RAGChunk]


class RAG:
    def __init__(
        self,
        *,
        top_k: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 512,
        llm_model: Optional[str] = None,
        llm_provider: str,
    ):

        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        pass

    def build_context(self, chunks: List[RAGChunk], max_chars: int = 12000) -> str:
        parts: List[str] = []
        total = 0

        for ch in chunks:
            header = f"[[{ch.source or 'unknown'}|{ch.semantic_id}]] "

            block = (
                f"Inicio del chunk"
                f"Extracto de Documento de investigacion con nombre de documento: {header}\n"
                f"Texto a utilizar: {ch.text.strip()}\n"
                f"Fin del chunk"
            )

            if total + len(block) > max_chars:
                break

            parts.append(block)
            total += len(block)

        return "\n...\n\n".join(parts).strip()

    def build_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        system = (
            "Mantener una conversación sobre extractos de documentos relacionados con temas académicos.\n"
            "→ Dar respuestas cortas, concretas y precisas.\n"
            "→ Utilizar únicamente la información de los siguientes extractos para construir la respuesta.\n"
            "===\n"
            "Extracto de Documento de investigación con nombre de documento: {Nombre del documento del primer chunk}\n"
            "→ Chunk: {ID del primer chunk}\n"
            "→ {Texto del primer chunk}\n"
            "...\n"
            "Extracto de Documento de investigación con nombre de documento: {Nombre del documento del último chunk}\n"
            "→ Chunk: {ID del último chunk}\n"
            "→ {Texto del último chunk}\n"
            "===\n"
            "→ Si los extractos no tienen una relación evidente con la pregunta, ignorarlos y responder que no se cuenta con información para responder la pregunta.\n"
            "→ Evitar las suposiciones.\n"
            "→ La conversación debe ser en español. Recordá ser concreto.\n"
            "→ Si la respuesta se construye utilizando información de más de un chunk, indicá claramente qué información proviene de cada chunk.\n"
            "→ Para cada afirmación relevante, mencioná el chunk correspondiente.\n"
            "→ No mezclar información de distintos chunks sin aclarar su origen.\n"
            "Siempre agregar al final de la respuesta los documentos y chunks utilizados bajo el título:\n"
            "**Referencias:**\n"
        )

        user = f"Pregunta: {question}\n\n" f"CONTEXTO:\n{context}\n\n" "===\n" "Respuesta:"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, query: str) -> RAGResult:

        if not query or not query.strip():
            raise ValueError("question is empty")
        # Generar embedding de la query
        embedder = get_embedder()
        query_embedding_array = embedder.embed_query(query)
        query_embedding = query_embedding_array[0].tolist()
        # 1) Retrieve
        store = VectorStore(collection_name="demo_collection", embedding_dim=1024)  # E5-large-v2
        raw = store.search(query_embedding, top_k=self.top_k)
        chunks = []
        for doc_id, score, meta in raw:
            rag_chunk = RAGChunk(
                id=doc_id,
                semantic_id=meta.get("semantic_id", "N/A"),
                text=meta.get("page_content", "N/A"),
                score=float(score),
                source=meta.get("__npy__", "N/A"),
            )
            chunks.append(rag_chunk)

        # 2) Build context
        context = self.build_context(chunks)

        # 3) Generate
        messages = self.build_messages(query, context)

        llm = get_llm_client(provider=self.llm_provider, model=self.llm_model)

        answer = llm.generate(
            messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        # 4) Return
        return RAGResult(answer=answer, contexts=chunks)


# LLMS Provider


class LLMClient(Protocol):
    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str: ...


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = E5Embedder()
    return _embedder


class GroqClient:
    def __init__(self, model):
        backend_dir = Path(__file__).resolve().parents[3]
        load_dotenv(backend_dir / ".env")
        key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=key)
        self.model = model

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return str(resp.choices[0].message.content)


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


class HFLocalLLM:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name

        # ===== elegir hardware =====
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
            self.device_map = "auto"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
            self.device_map = None
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            self.device_map = None

        print(f" LLM cargando en {self.device} ({self.dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )

        if self.device_map is None:
            self.model.to(self.device)

        self.model.eval()

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        top_p = (0.9,)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if self.device_map is None:
            input_ids = input_ids.to(self.device)

        do_sample = temperature > 0

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0, input_ids.shape[1] :]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


def get_llm_client(provider: str, *, model: Optional[str] = None) -> LLMClient:
    if provider not in {"groq", "ollama", "local"}:
        raise ValueError(f"LLM provider no soportado: {provider}")

    cached = _llm_instances.get(provider)
    if cached is not None:
        return cached

    if provider == "groq":
        client: LLMClient = GroqClient(model=model or "llama-3.1-8b-instant")
    elif provider == "ollama":
        client = OllamaClient(model or "qwen2.5:3b-instruct")
    else:  # provider == "local"
        client = HFLocalLLM(model_name=model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    _llm_instances[provider] = client
    return client
