import pickle
from pathlib import Path

import faiss

from app.rag.embeddings import generate_embeddings
from app.rag.models import MedicineDocument
from app.rag.parser import documents_to_text
from app.rag.sources.dailymed import (
    search_dailymed_by_rxcui,
    search_dailymed_by_name,
    get_spl,
)
from app.rag.sources.rxnorm import (
    identify_medicine,
)


VECTORSTORE_DIR = Path(
    "app/vectorstore"
)

INDEX_PATH = (
    VECTORSTORE_DIR /
    "medicine_index.faiss"
)

DOCUMENTS_PATH = (
    VECTORSTORE_DIR /
    "documents.pkl"
)


# Medicines to bootstrap the knowledge base.
#
# IMPORTANT:
# These are NOT a manually-created medical database.
# They are simply medicine names for which we want
# to retrieve authoritative DailyMed information.
SEED_MEDICINES = [
    "Augmentin",
    "Amoxicillin",
    "Metformin",
    "Atorvastatin",
    "Ibuprofen",
    "Pantoprazole",
]


def retrieve_medicine_documents(
    medicine_name: str,
) -> list[MedicineDocument]:

    print(
        f"\nProcessing medicine: "
        f"{medicine_name}"
    )

    # ---------------------------------------------------------
    # 1. Normalize medicine using RxNorm
    # ---------------------------------------------------------

    identity = identify_medicine(
        medicine_name
    )

    if identity:

        print(
            f"RxNorm: "
            f"{identity.normalized_name}"
        )

        print(
            f"RxCUI: "
            f"{identity.rxcui}"
        )

    else:

        print(
            "RxNorm could not identify "
            f"{medicine_name}"
        )

    # ---------------------------------------------------------
    # 2. Search DailyMed
    # ---------------------------------------------------------

    results = []

    if identity and identity.rxcui:

        results = search_dailymed_by_rxcui(
            identity.rxcui
        )

    # Fallback to name search.
    if not results:

        print(
            "RxCUI search returned no "
            "DailyMed records. Trying name..."
        )

        results = search_dailymed_by_name(
            medicine_name
        )

    if not results:

        print(
            f"No DailyMed records found "
            f"for {medicine_name}"
        )

        return []

    documents = []

    # ---------------------------------------------------------
    # 3. Retrieve SPL records
    # ---------------------------------------------------------

    # Start conservatively.
    # We don't want hundreds of duplicate labels.
    for result in results[:3]:

        set_id = (
            result.get("setid")
            or result.get("setId")
        )

        if not set_id:
            continue

        print(
            f"Retrieving SPL: {set_id}"
        )

        spl = get_spl(
            set_id
        )

        if not spl:
            continue

        parsed = (
            __import__(
                "app.rag.parser",
                fromlist=[
                    "parse_dailymed_spl"
                ],
            )
            .parse_dailymed_spl(
                spl,
                medicine_name=(
                    identity.normalized_name
                    if identity
                    else medicine_name
                ),
                rxcui=(
                    identity.rxcui
                    if identity
                    else None
                ),
            )
        )

        documents.extend(
            parsed
        )

    return documents


def build_index():

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_documents = []

    for medicine_name in SEED_MEDICINES:

        try:

            documents = (
                retrieve_medicine_documents(
                    medicine_name
                )
            )

            all_documents.extend(
                documents
            )

        except Exception as exc:

            print(
                f"Failed to process "
                f"{medicine_name}: {exc}"
            )

    if not all_documents:

        raise RuntimeError(
            "No medicine documents were "
            "retrieved. Index was not created."
        )

    # ---------------------------------------------------------
    # Remove duplicate documents
    # ---------------------------------------------------------

    unique_documents = []

    seen = set()

    for document in all_documents:

        key = (
            document.source_id,
            document.section,
            document.content,
        )

        if key in seen:
            continue

        seen.add(key)

        unique_documents.append(
            document
        )

    all_documents = unique_documents

    print(
        f"\nTotal documents: "
        f"{len(all_documents)}"
    )

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------

    documents, vectors = (
        generate_embeddings(
            all_documents
        )
    )

    dimension = vectors.shape[1]

    # ---------------------------------------------------------
    # Build FAISS index
    # ---------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        vectors
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    with open(
        DOCUMENTS_PATH,
        "wb",
    ) as file:

        pickle.dump(
            documents,
            file,
        )

    print(
        "\n================================"
    )

    print(
        "Medicine RAG index created"
    )

    print(
        f"Documents: "
        f"{len(documents)}"
    )

    print(
        f"Index: "
        f"{INDEX_PATH}"
    )

    print(
        f"Documents: "
        f"{DOCUMENTS_PATH}"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    build_index()