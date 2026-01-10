"""Tests para la VectorStore."""

import pytest

from institutional_graphrag.retrieval import VectorStore


@pytest.fixture
def vector_store():
    """Vector store para testing."""
    store = VectorStore(
        collection_name="test_collection",
        embedding_dim=128,  # para testing
    )
    # Limpiar colección antes de cada test
    store.clear_collection()
    yield store
    store.close()


@pytest.fixture
def sample_data():
    """Sample data para testing."""
    texts = [
        "El proyecto de investigación busca mejorar la calidad educativa.",
        "Los resultados del estudio muestran avances significativos.",
        "La metodología aplicada fue validada por expertos.",
    ]
    # mockear embeddings
    embeddings = [
        [0.1] * 128,
        [0.2] * 128,
        [0.3] * 128,
    ]
    metadata = [
        {"year": 2021, "category": "education"},
        {"year": 2022, "category": "research"},
        {"year": 2023, "category": "methodology"},
    ]
    return texts, embeddings, metadata


def test_vector_store_initialization(vector_store):
    """La vector store inicia correctamente"""
    assert vector_store.collection_name == "test_collection"
    assert vector_store.embedding_dim == 128
    assert vector_store.count_documents() == 0


def test_add_documents(vector_store, sample_data):
    """Agregar documentos al vector store."""
    texts, embeddings, metadata = sample_data

    ids = vector_store.add_documents(embeddings, metadata)

    assert len(ids) == len(embeddings)
    assert vector_store.count_documents() == len(embeddings)


def test_search(vector_store, sample_data):
    """Buscar documentos similares."""
    texts, embeddings, metadata = sample_data
    vector_store.add_documents(embeddings, metadata)

    # Buscar con una consulta similar al primer documento
    query_embedding = [0.1] * 128
    results = vector_store.search(query_embedding, top_k=2)

    assert len(results) <= 2
    assert all(len(result) == 3 for result in results)  # (id, score, metadata)

    # Verificar estructura de resultados
    doc_id, score, meta = results[0]
    assert isinstance(doc_id, str)
    assert isinstance(score, float)
    assert isinstance(meta, dict)


def test_search_with_score_threshold(vector_store, sample_data):
    """Buscar documentos con score."""
    texts, embeddings, metadata = sample_data
    vector_store.add_documents(embeddings, metadata)

    query_embedding = [0.1] * 128
    results = vector_store.search(query_embedding, top_k=10, score_threshold=0.99)

    # Con un umbral alto, deberíamos obtener menos resultados
    assert all(score >= 0.99 for _, score, _ in results)


def test_count_documents(vector_store, sample_data):
    """Test contar documentos en la colección."""
    texts, embeddings, metadata = sample_data

    assert vector_store.count_documents() == 0

    vector_store.add_documents(embeddings[:2], metadata[:2])
    assert vector_store.count_documents() == 2

    vector_store.add_documents(embeddings[2:], metadata[2:])
    assert vector_store.count_documents() == 3
