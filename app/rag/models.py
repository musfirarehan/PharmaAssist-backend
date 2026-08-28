from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PrescriptionMedicine:
    """
    Facts extracted directly from the patient's prescription.
    These must never be generated or modified by the LLM.
    """

    name: str
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""
    confidence: Optional[float] = None


@dataclass
class MedicineIdentity:
    """
    Normalized medicine identity obtained from RxNorm.
    """

    original_name: str
    normalized_name: str = ""
    rxcui: Optional[str] = None
    generic_name: str = ""
    match_type: str = ""
    match_score: Optional[float] = None


@dataclass
class MedicineDocument:
    """
    A searchable RAG document derived from an authoritative
    medicine source such as DailyMed.
    """

    medicine_name: str
    content: str
    source: str
    source_id: str = ""
    rxcui: Optional[str] = None
    section: str = ""
    last_updated: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    """
    A document returned by the RAG retriever.
    """

    content: str
    source: str
    source_id: str = ""
    medicine_name: str = ""
    section: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)