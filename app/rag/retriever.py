from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


VECTORSTORE_DIR = Path("app/vectorstore")

INDEX_PATH = VECTORSTORE_DIR / "medicine_index.faiss"
DOCUMENTS_PATH = VECTORSTORE_DIR / "documents.pkl"

# IMPORTANT:
# This MUST be the exact same model used in embeddings.py during
# build_index.py, otherwise query vectors and document vectors will
# live in different embedding spaces and retrieval will silently
# return garbage (no error, just wrong/irrelevant matches).
#
# If embeddings.py uses a different model name, update this constant
# to match it exactly.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# FAISS always returns the k nearest neighbors, even if none of them
# are actually relevant to the query. With normalized vectors +
# IndexFlatIP, the score is cosine similarity (-1 to 1). Below this
# threshold, treat the match as "not found" rather than a real hit,
# so dynamic resolution actually triggers for medicines that aren't
# meaningfully represented in the index yet.
MIN_RELEVANCE_SCORE = 0.45


# ---------------------------------------------------------
# Lazy-loaded singletons (avoid reloading FAISS/model per call)
# ---------------------------------------------------------

_index: Optional[faiss.Index] = None
_documents: Optional[list] = None
_model: Optional[SentenceTransformer] = None


def _load_index() -> faiss.Index:
    global _index
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run build_index.py first."
            )
        _index = faiss.read_index(str(INDEX_PATH))
    return _index


def _load_documents() -> list:
    global _documents
    if _documents is None:
        if not DOCUMENTS_PATH.exists():
            raise FileNotFoundError(
                f"documents.pkl not found at {DOCUMENTS_PATH}. "
                "Run build_index.py first."
            )
        with open(DOCUMENTS_PATH, "rb") as file:
            _documents = pickle.load(file)
    return _documents


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _embed_query(query: str) -> np.ndarray:
    model = _load_model()
    vector = model.encode(
        [query],
        normalize_embeddings=True,
    )
    return np.asarray(vector, dtype="float32")


# ---------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------

def retrieve_medicine(medicine_name: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top_k most relevant medicine knowledge chunks for a
    given medicine name via semantic search over the FAISS index
    built from DailyMed data.

    Returns a list of dicts ready to drop into the Gemini counseling
    prompt (this is what counseling.py already expects back from
    `retrieve_medicine`).
    """

    if not medicine_name or not medicine_name.strip():
        return []

    index = _load_index()
    documents = _load_documents()

    if index.ntotal == 0 or not documents:
        return []

    query = f"What is {medicine_name.strip()}?"
    query_vector = _embed_query(query)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, k)

    results = []

    for score, doc_index in zip(scores[0], indices[0]):

        if doc_index < 0 or doc_index >= len(documents):
            continue

        document = documents[doc_index]

        results.append({
            "medicine_name": getattr(document, "medicine_name", medicine_name),
            "section": getattr(document, "section", None),
            "content": getattr(document, "content", ""),
            "source_id": getattr(document, "source_id", None),
            "rxcui": getattr(document, "rxcui", None),
            "relevance_score": float(score),
        })

    return results


# ---------------------------------------------------------
# Dynamic resolution for medicines not yet in the index
# ---------------------------------------------------------

def resolve_and_cache_medicine(medicine_name: str) -> list[dict]:
    """
    Fallback for medicines OCR detects that aren't in the current
    FAISS index. Resolves them live through RxNorm + DailyMed (same
    pipeline as build_index.py), embeds the new documents, appends
    them to the in-memory + on-disk index, and returns fresh
    retrieval results.

    This is what turns your 6-medicine seed list into an index that
    grows over time instead of saying "not found" forever.
    """

    from app.rag.build_index import retrieve_medicine_documents
    from app.rag.embeddings import generate_embeddings

    new_docs = retrieve_medicine_documents(medicine_name)

    if not new_docs:
        return []

    embedded_docs, vectors = generate_embeddings(new_docs)

    index = _load_index()
    documents = _load_documents()

    index.add(vectors)
    documents.extend(embedded_docs)

    faiss.write_index(index, str(INDEX_PATH))

    with open(DOCUMENTS_PATH, "wb") as file:
        pickle.dump(documents, file)

    global _index, _documents
    _index = index
    _documents = documents

    return retrieve_medicine(medicine_name, top_k=3)


def get_medicine_knowledge(medicine_name: str, top_k: int = 3) -> list[dict]:
    """
    Preferred entry point for the counseling pipeline.

    Tries the existing index first; if nothing relevant comes back
    (medicine not yet indexed), resolves it dynamically via RxNorm/
    DailyMed and caches it for next time, instead of returning
    "not found".

    Swap this in for `retrieve_medicine` inside counseling.py once
    you're ready to make retrieval dynamic.
    """

    results = retrieve_medicine(medicine_name, top_k=top_k)

    if results:
        return results

    print(f"'{medicine_name}' not found in index. Resolving dynamically...")

    return resolve_and_cache_medicine(medicine_name)