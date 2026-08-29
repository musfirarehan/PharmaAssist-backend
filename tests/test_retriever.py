import types
import unittest

import numpy as np

from app.rag import retriever


class FakeIndex:
    def __init__(self, scores, indices, ntotal):
        self._scores = scores
        self._indices = indices
        self.ntotal = ntotal

    def search(self, query_vector, k):
        return self._scores, self._indices


def _doc(name, rxcui=None, section="indications", content="content"):
    return types.SimpleNamespace(
        medicine_name=name,
        rxcui=rxcui,
        section=section,
        content=content,
        source_id=f"{name}-source",
    )


class RetrieverThresholdTests(unittest.TestCase):
    def setUp(self):
        retriever._index = None
        retriever._documents = None
        retriever._model = None

    def test_retrieve_medicine_keeps_only_qualifying_results(self):
        fake_index = FakeIndex(
            np.array([[0.91, 0.72, 0.41]]),
            np.array([[0, 1, 2]]),
            ntotal=3,
        )
        docs = [
            _doc("Amoxicillin", "123"),
            _doc("Amoxicillin", "123"),
            _doc("Metformin", "456"),
        ]

        retriever._load_index = lambda: fake_index
        retriever._load_documents = lambda: docs
        retriever._embed_query = lambda query: np.array([[0.1]], dtype="float32")

        results = retriever.retrieve_medicine("Amoxicillin", top_k=3)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["relevance_score"] >= retriever.MIN_RELEVANCE_SCORE for item in results))
        self.assertTrue(all(item["medicine_name"] == "Amoxicillin" for item in results))

    def test_get_medicine_knowledge_uses_fallback_when_all_scores_are_low(self):
        fake_index = FakeIndex(
            np.array([[0.30, 0.20]]),
            np.array([[0, 1]]),
            ntotal=2,
        )
        docs = [
            _doc("Metformin", "456"),
            _doc("Pantoprazole", "789"),
        ]

        retriever._load_index = lambda: fake_index
        retriever._load_documents = lambda: docs
        retriever._embed_query = lambda query: np.array([[0.1]], dtype="float32")

        fallback = [{"medicine_name": "Amoxicillin", "rxcui": "123", "relevance_score": 0.9}]
        retriever.resolve_and_cache_medicine = lambda medicine_name: fallback

        result = retriever.get_medicine_knowledge("Amoxicillin", top_k=3)

        self.assertEqual(result, fallback)

    def test_retrieve_medicine_rejects_identity_conflict_even_with_high_score(self):
        fake_index = FakeIndex(
            np.array([[0.91]]),
            np.array([[0]]),
            ntotal=1,
        )
        docs = [_doc("Metformin", "456")]

        retriever._load_index = lambda: fake_index
        retriever._load_documents = lambda: docs
        retriever._embed_query = lambda query: np.array([[0.1]], dtype="float32")

        result = retriever.retrieve_medicine("Amoxicillin", top_k=3)

        self.assertEqual(result, [])

    def test_retrieve_medicine_accepts_boundary_score(self):
        fake_index = FakeIndex(
            np.array([[0.45]]),
            np.array([[0]]),
            ntotal=1,
        )
        docs = [_doc("Amoxicillin", "123")]

        retriever._load_index = lambda: fake_index
        retriever._load_documents = lambda: docs
        retriever._embed_query = lambda query: np.array([[0.1]], dtype="float32")

        result = retriever.retrieve_medicine("Amoxicillin", top_k=3)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["relevance_score"], 0.45)

    def test_get_medicine_knowledge_rejects_unrelated_identity_conflicts(self):
        fake_index = FakeIndex(
            np.array([[0.71]]),
            np.array([[0]]),
            ntotal=1,
        )
        docs = [_doc("Ibuprofen", "111")]

        retriever._load_index = lambda: fake_index
        retriever._load_documents = lambda: docs
        retriever._embed_query = lambda query: np.array([[0.1]], dtype="float32")

        retriever.resolve_and_cache_medicine = lambda medicine_name: []

        result = retriever.get_medicine_knowledge("Amoxicillin", top_k=3)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
