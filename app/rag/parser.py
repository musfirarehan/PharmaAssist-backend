from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from app.rag.models import MedicineDocument


# -------------------------------------------------------------------
# DailyMed/SPL sections useful for medicine counseling
# -------------------------------------------------------------------

IMPORTANT_SECTIONS = {
    "INDICATIONS AND USAGE": "purpose",
    "DOSAGE AND ADMINISTRATION": "dosage",
    "CONTRAINDICATIONS": "contraindications",
    "WARNINGS": "warnings",
    "WARNINGS AND PRECAUTIONS": "warnings",
    "ADVERSE REACTIONS": "side_effects",
    "DRUG INTERACTIONS": "interactions",
    "USE IN SPECIFIC POPULATIONS": "special_populations",
    "PREGNANCY": "pregnancy",
    "LACTATION": "breastfeeding",
    "HOW SUPPLIED": "availability",
    "STORAGE": "storage",
    "PATIENT COUNSELING INFORMATION": "patient_counseling",
}


# -------------------------------------------------------------------
# Text utilities
# -------------------------------------------------------------------

def _clean_text(text: Any) -> str:
    """
    Normalize extracted SPL text.
    """

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Remove excessive whitespace.
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _element_text(element: ET.Element) -> str:
    """
    Extract all visible text contained inside an XML element.
    """

    text = " ".join(
        part.strip()
        for part in element.itertext()
        if part and part.strip()
    )

    return _clean_text(text)


def _normalize_heading(text: str) -> str:
    """
    Normalize section headings for comparison.
    """

    text = _clean_text(text).upper()

    # Remove numbering such as:
    # 1
    # 1.1
    # 5.3
    # 5.3.1
    text = re.sub(
        r"^\d+(?:\.\d+)*[\s\.\-:]*",
        "",
        text,
    )

    return text.strip()


# -------------------------------------------------------------------
# Section matching
# -------------------------------------------------------------------

def _section_name_from_heading(
    heading: str,
) -> str | None:

    normalized = _normalize_heading(
        heading
    )

    # Longest/most specific matches first.
    candidates = sorted(
        IMPORTANT_SECTIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for known_heading, section_name in candidates:

        if (
            normalized == known_heading
            or normalized.startswith(
                known_heading + " "
            )
            or known_heading in normalized
        ):
            return section_name

    return None


# -------------------------------------------------------------------
# XML section extraction
# -------------------------------------------------------------------

def _find_xml_sections(
    root: ET.Element,
) -> dict[str, str]:

    sections: dict[str, str] = {}

    # SPL uses HL7 CDA namespaces.
    #
    # Instead of relying on one exact namespace URI,
    # we inspect local XML tag names. This makes the
    # parser more tolerant of namespace variations.

    for element in root.iter():

        local_name = (
            element.tag.split("}")[-1]
            if "}" in element.tag
            else element.tag
        )

        if local_name.lower() != "section":
            continue

        heading_text = ""

        # Look for the section title/heading.
        for child in element:

            child_local = (
                child.tag.split("}")[-1]
                if "}" in child.tag
                else child.tag
            )

            if child_local.lower() == "title":

                heading_text = _element_text(
                    child
                )

                break

        if not heading_text:
            continue

        section_name = _section_name_from_heading(
            heading_text
        )

        if not section_name:
            continue

        content = ""

        # Prefer section/text content.
        for child in element:

            child_local = (
                child.tag.split("}")[-1]
                if "}" in child.tag
                else child.tag
            )

            if child_local.lower() == "text":

                content = _element_text(
                    child
                )

                break

        # Some SPL structures may not expose
        # everything through <text>.
        if not content:
            content = _element_text(
                element
            )

        # Remove the heading from the content
        # when it was included.
        if heading_text and content:

            content = content.replace(
                heading_text,
                "",
                1,
            ).strip()

        if not content:
            continue

        # Several nested sections may map to the
        # same category, e.g. WARNINGS and
        # WARNINGS AND PRECAUTIONS.
        if section_name in sections:

            sections[section_name] = (
                sections[section_name]
                + " "
                + content
            )

        else:

            sections[section_name] = content

    return {
        key: _clean_text(value)
        for key, value in sections.items()
        if value
    }


# -------------------------------------------------------------------
# Metadata extraction
# -------------------------------------------------------------------

def _find_xml_value(
    root: ET.Element,
    local_names: list[str],
) -> str:

    wanted = {
        name.lower()
        for name in local_names
    }

    for element in root.iter():

        local_name = (
            element.tag.split("}")[-1]
            if "}" in element.tag
            else element.tag
        )

        if local_name.lower() not in wanted:
            continue

        # First look at text.
        value = _element_text(element)

        if value:
            return value

        # Then common attributes.
        for attribute in (
            "value",
            "displayName",
            "code",
        ):

            if attribute in element.attrib:

                value = _clean_text(
                    element.attrib[attribute]
                )

                if value:
                    return value

    return ""


def _extract_xml_metadata(
    root: ET.Element,
) -> dict[str, str]:

    metadata: dict[str, str] = {}

    # Effective date.
    effective_time = ""

    for element in root.iter():

        local_name = (
            element.tag.split("}")[-1]
            if "}" in element.tag
            else element.tag
        )

        if local_name.lower() == "effectiveTime".lower():

            effective_time = (
                element.attrib.get(
                    "value",
                    "",
                )
            )

            if effective_time:
                break

    metadata["effective_time"] = (
        effective_time
    )

    title = _find_xml_value(
        root,
        [
            "title",
        ],
    )

    if title:
        metadata["title"] = title

    # SPL Set ID is usually represented by
    # an <id root="..."> element.
    for element in root.iter():

        local_name = (
            element.tag.split("}")[-1]
            if "}" in element.tag
            else element.tag
        )

        if local_name.lower() != "id":
            continue

        root_id = element.attrib.get(
            "root",
            "",
        )

        if root_id:
            metadata["set_id"] = root_id
            break

    return {
        key: value
        for key, value in metadata.items()
        if value
    }


# -------------------------------------------------------------------
# JSON fallback
# -------------------------------------------------------------------

def _find_json_sections(
    data: Any,
) -> dict[str, str]:

    """
    Keep support for JSON DailyMed responses.

    This is useful if DailyMed changes response
    formats or another source returns structured JSON.
    """

    sections: dict[str, str] = {}

    def walk(value: Any):

        if isinstance(value, dict):

            title = (
                value.get("title")
                or value.get("section")
                or value.get("heading")
                or value.get("name")
            )

            text = (
                value.get("text")
                or value.get("content")
                or value.get("value")
            )

            if (
                isinstance(title, str)
                and isinstance(text, str)
            ):

                section_name = (
                    _section_name_from_heading(
                        title
                    )
                )

                if section_name:

                    cleaned = _clean_text(
                        text
                    )

                    if cleaned:

                        if section_name in sections:

                            sections[
                                section_name
                            ] += (
                                " "
                                + cleaned
                            )

                        else:

                            sections[
                                section_name
                            ] = cleaned

            for child in value.values():
                walk(child)

        elif isinstance(value, list):

            for child in value:
                walk(child)

    walk(data)

    return sections


# -------------------------------------------------------------------
# Main parser
# -------------------------------------------------------------------

def parse_dailymed_spl(
    data: Any,
    medicine_name: str,
    rxcui: str | None = None,
) -> list[MedicineDocument]:

    """
    Convert an official DailyMed SPL into
    searchable MedicineDocument objects.

    Supported input:

    1. New get_spl() response:

        {
            "setid": "...",
            "format": "xml",
            "filename": "...",
            "content": "<ClinicalDocument>..."
        }

    2. Raw XML string.

    3. Legacy JSON response.
    """

    if not data:
        return []

    xml_content = None
    metadata: dict[str, str] = {}

    # ---------------------------------------------------------------
    # New DailyMed downloader response
    # ---------------------------------------------------------------

    if isinstance(data, dict):

        xml_content = data.get(
            "content"
        )

        if data.get("setid"):
            metadata["set_id"] = str(
                data["setid"]
            )

        if data.get("filename"):
            metadata["filename"] = str(
                data["filename"]
            )

    # ---------------------------------------------------------------
    # Raw XML string
    # ---------------------------------------------------------------

    elif isinstance(data, str):

        xml_content = data

    # ---------------------------------------------------------------
    # Parse XML
    # ---------------------------------------------------------------

    if xml_content:

        try:

            root = ET.fromstring(
                xml_content
            )

        except ET.ParseError as exc:

            print(
                "DailyMed SPL XML parsing failed: "
                f"{exc}"
            )

            return []

        xml_metadata = (
            _extract_xml_metadata(
                root
            )
        )

        metadata.update(
            xml_metadata
        )

        sections = (
            _find_xml_sections(
                root
            )
        )

    # ---------------------------------------------------------------
    # Legacy JSON fallback
    # ---------------------------------------------------------------

    elif isinstance(data, dict):

        sections = (
            _find_json_sections(
                data
            )
        )

        # Preserve common JSON metadata.
        for key in (
            "setId",
            "set_id",
            "publishedDate",
            "published_date",
            "effectiveTime",
            "effective_time",
            "title",
            "splTitle",
        ):

            if key not in data:
                continue

            value = data[key]

            if not value:
                continue

            normalized_key = {
                "setId": "set_id",
                "set_id": "set_id",
                "publishedDate": "published_date",
                "published_date": "published_date",
                "effectiveTime": "effective_time",
                "effective_time": "effective_time",
                "splTitle": "title",
                "title": "title",
            }.get(
                key,
                key,
            )

            metadata[
                normalized_key
            ] = str(value)

    else:

        return []

    # ---------------------------------------------------------------
    # Final metadata
    # ---------------------------------------------------------------

    set_id = metadata.get(
        "set_id",
        "",
    )

    # ---------------------------------------------------------------
    # Create RAG documents
    # ---------------------------------------------------------------

    documents: list[MedicineDocument] = []

    for section_name, content in (
        sections.items()
    ):

        if not content:
            continue

        document = MedicineDocument(
            medicine_name=medicine_name,
            content=(
                f"Medicine: {medicine_name}\n"
                f"Source: DailyMed\n"
                f"Section: {section_name}\n\n"
                f"{content}"
            ),
            source="DailyMed",
            source_id=set_id,
            rxcui=rxcui,
            section=section_name,
            last_updated=(
                metadata.get(
                    "effective_time",
                    "",
                )
            ),
            metadata=metadata,
        )

        documents.append(
            document
        )

    return documents


# -------------------------------------------------------------------
# Embedding helper
# -------------------------------------------------------------------

def documents_to_text(
    documents: list[MedicineDocument],
) -> list[str]:

    """
    Convert MedicineDocument objects into
    text suitable for embedding.
    """

    return [
        document.content
        for document in documents
    ]