from sentence_transformers import SentenceTransformer

from app.rag.models import MedicineDocument


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(
    documents: list[MedicineDocument],
):
    """
    Convert authoritative medicine documents into
    vector embeddings.

    Returns:
        documents: original MedicineDocument objects
        vectors: numpy array of embeddings
    """

    if not documents:
        raise ValueError(
            "No medicine documents were provided."
        )

    texts = [
        document.content
        for document in documents
    ]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return documents, vectors