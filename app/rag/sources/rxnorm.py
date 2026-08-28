import re
from typing import Optional

import requests

from app.rag.models import MedicineIdentity


RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

REQUEST_TIMEOUT = 15


def _clean_name(name: str) -> str:
    """
    Clean OCR medicine names before sending them to RxNorm.
    """

    if not name:
        return ""

    name = name.strip()

    # Remove common prescription dosage information.
    name = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|mg/ml|mcg/ml)\b",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Remove common tablet/capsule abbreviations.
    name = re.sub(
        r"\b(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules)\b",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def _request_rxcui(
    name: str,
    search_mode: int,
) -> list[str]:

    url = f"{RXNORM_BASE_URL}/rxcui.json"

    response = requests.get(
        url,
        params={
            "name": name,
            "search": search_mode,
            "allsrc": 0,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    id_group = data.get("idGroup", {})

    rxcuis = id_group.get("rxnormId", [])

    if isinstance(rxcuis, str):
        rxcuis = [rxcuis]

    return rxcuis


def _get_concept_properties(rxcui: str) -> dict:

    url = f"{RXNORM_BASE_URL}/rxcui/{rxcui}/properties.json"

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("properties", {})


def _approximate_match(name: str) -> Optional[dict]:

    url = f"{RXNORM_BASE_URL}/approximateTerm.json"

    response = requests.get(
        url,
        params={
            "term": name,
            "maxEntries": 5,
            "option": 1,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    candidates = (
        data
        .get("approximateGroup", {})
        .get("candidate", [])
    )

    if not candidates:
        return None

    if isinstance(candidates, dict):
        candidates = [candidates]

    # RxNorm returns candidates ordered by match quality.
    return candidates[0]


def identify_medicine(
    medicine_name: str,
) -> Optional[MedicineIdentity]:
    """
    Resolve an OCR medicine name to an RxNorm concept.

    Strategy:

    1. Exact match
    2. Normalized match
    3. Approximate match

    Returns None when the medicine cannot be identified
    confidently enough.
    """

    original_name = medicine_name.strip()

    if not original_name:
        return None

    cleaned_name = _clean_name(original_name)

    if not cleaned_name:
        return None

    # ---------------------------------------------------------
    # 1. Exact match
    # ---------------------------------------------------------

    try:

        rxcuis = _request_rxcui(
            cleaned_name,
            search_mode=0,
        )

        if rxcuis:

            rxcui = rxcuis[0]

            properties = _get_concept_properties(
                rxcui
            )

            return MedicineIdentity(
                original_name=original_name,
                normalized_name=properties.get(
                    "name",
                    cleaned_name,
                ),
                rxcui=rxcui,
                generic_name=properties.get(
                    "name",
                    "",
                ),
                match_type="exact",
                match_score=1.0,
            )

    except requests.RequestException:
        pass

    # ---------------------------------------------------------
    # 2. Normalized match
    # ---------------------------------------------------------

    try:

        rxcuis = _request_rxcui(
            cleaned_name,
            search_mode=1,
        )

        if rxcuis:

            rxcui = rxcuis[0]

            properties = _get_concept_properties(
                rxcui
            )

            return MedicineIdentity(
                original_name=original_name,
                normalized_name=properties.get(
                    "name",
                    cleaned_name,
                ),
                rxcui=rxcui,
                generic_name=properties.get(
                    "name",
                    "",
                ),
                match_type="normalized",
                match_score=0.95,
            )

    except requests.RequestException:
        pass

    # ---------------------------------------------------------
    # 3. Approximate match
    # ---------------------------------------------------------

    try:

        candidate = _approximate_match(
            cleaned_name
        )

        if candidate:

            rxcui = candidate.get("rxcui")

            if not rxcui:
                return None

            score = candidate.get("score")

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None

            properties = _get_concept_properties(
                rxcui
            )

            return MedicineIdentity(
                original_name=original_name,
                normalized_name=properties.get(
                    "name",
                    candidate.get(
                        "name",
                        cleaned_name,
                    ),
                ),
                rxcui=rxcui,
                generic_name=properties.get(
                    "name",
                    candidate.get(
                        "name",
                        "",
                    ),
                ),
                match_type="approximate",
                match_score=score,
            )

    except requests.RequestException:
        pass

    return None