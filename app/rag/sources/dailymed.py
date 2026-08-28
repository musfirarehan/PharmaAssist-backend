from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests


DAILYMED_BASE_URL = (
    "https://dailymed.nlm.nih.gov/dailymed/services/v2"
)

DAILYMED_DOWNLOAD_URL = (
    "https://dailymed.nlm.nih.gov/dailymed/downloadzipfile.cfm"
)

REQUEST_TIMEOUT = 30

CACHE_DIR = Path("app/data/raw/dailymed")
CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "PharmaAssist-AI/1.0 "
            "(educational research project)"
        ),
        "Accept": "*/*",
    }
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def _request(
    url: str,
    params: Optional[dict] = None,
    retries: int = 3,
) -> Optional[requests.Response]:

    for attempt in range(retries):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            return response

        except requests.RequestException as exc:

            print(
                f"DailyMed request failed "
                f"(attempt {attempt + 1}/{retries}): "
                f"{exc}"
            )

            if attempt < retries - 1:
                time.sleep(2)

    return None


# ---------------------------------------------------------
# Search by RxCUI
# ---------------------------------------------------------


def search_dailymed_by_rxcui(
    rxcui: str,
) -> list[dict]:

    """
    Find DailyMed SPL records associated
    with an RxNorm RxCUI.
    """

    if not rxcui:
        return []

    url = (
        f"{DAILYMED_BASE_URL}/spls.json"
    )

    response = _request(
        url,
        params={
            "rxcui": rxcui,
            "pagesize": 10,
        },
    )

    if response is None:
        return []

    try:

        data = response.json()

    except ValueError as exc:

        print(
            f"DailyMed returned invalid JSON: {exc}"
        )

        return []

    results = data.get(
        "data",
        [],
    )

    if not isinstance(results, list):
        return []

    return results


# ---------------------------------------------------------
# Search by medicine name
# ---------------------------------------------------------


def search_dailymed_by_name(
    medicine_name: str,
) -> list[dict]:

    """
    Search DailyMed directly by drug name.

    Used as a fallback when RxNorm
    identification/search does not return
    useful DailyMed records.
    """

    if not medicine_name:
        return []

    medicine_name = medicine_name.strip()

    url = (
        f"{DAILYMED_BASE_URL}/spls.json"
    )

    response = _request(
        url,
        params={
            "drug_name": medicine_name,
            "pagesize": 10,
        },
    )

    if response is None:
        return []

    try:

        data = response.json()

    except ValueError as exc:

        print(
            f"DailyMed returned invalid JSON: {exc}"
        )

        return []

    results = data.get(
        "data",
        [],
    )

    if not isinstance(results, list):
        return []

    return results


# ---------------------------------------------------------
# Download SPL
# ---------------------------------------------------------


def get_spl(
    set_id: str,
) -> Optional[dict]:

    """
    Download the official DailyMed SPL ZIP
    for a Set ID and extract the XML label.

    Returns a dictionary containing:

        {
            "setid": "...",
            "format": "xml",
            "filename": "...",
            "content": "..."
        }

    The XML is the actual Structured Product
    Label used by the downstream parser.
    """

    if not set_id:
        return None

    set_id = set_id.strip()

    cache_file = (
        CACHE_DIR /
        f"{set_id}.xml"
    )

    # -----------------------------------------------------
    # Use cached SPL if available
    # -----------------------------------------------------

    if cache_file.exists():

        try:

            content = cache_file.read_text(
                encoding="utf-8"
            )

            print(
                f"Using cached SPL: {set_id}"
            )

            return {
                "setid": set_id,
                "format": "xml",
                "filename": cache_file.name,
                "content": content,
            }

        except OSError as exc:

            print(
                f"Could not read cached SPL: {exc}"
            )

    # -----------------------------------------------------
    # Download official DailyMed ZIP
    # -----------------------------------------------------

    print(
        f"Downloading DailyMed SPL ZIP: "
        f"{set_id}"
    )

    response = _request(
        DAILYMED_DOWNLOAD_URL,
        params={
            "setId": set_id,
        },
    )

    if response is None:
        return None

    # -----------------------------------------------------
    # Validate ZIP
    # -----------------------------------------------------

    if not zipfile.is_zipfile(
        io.BytesIO(response.content)
    ):

        print(
            "DailyMed download was not a ZIP file."
        )

        print(
            f"Content-Type: "
            f"{response.headers.get('content-type')}"
        )

        return None

    # -----------------------------------------------------
    # Extract XML from ZIP
    # -----------------------------------------------------

    try:

        with zipfile.ZipFile(
            io.BytesIO(response.content)
        ) as archive:

            xml_files = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".xml")
            ]

            if not xml_files:

                print(
                    "No XML SPL found inside "
                    f"DailyMed ZIP: {set_id}"
                )

                return None

            # Prefer the largest XML file.
            #
            # SPL packages can contain supporting
            # XML files. The main label is normally
            # the largest XML document.

            xml_file = max(
                xml_files,
                key=lambda name: archive.getinfo(
                    name
                ).file_size,
            )

            content = archive.read(
                xml_file
            ).decode(
                "utf-8",
                errors="replace",
            )

    except (
        zipfile.BadZipFile,
        OSError,
        UnicodeDecodeError,
    ) as exc:

        print(
            f"Could not extract DailyMed SPL: "
            f"{exc}"
        )

        return None

    # -----------------------------------------------------
    # Cache XML
    # -----------------------------------------------------

    try:

        cache_file.write_text(
            content,
            encoding="utf-8",
        )

    except OSError as exc:

        print(
            f"Could not cache SPL: {exc}"
        )

    print(
        f"SPL retrieved successfully: "
        f"{xml_file}"
    )

    return {
        "setid": set_id,
        "format": "xml",
        "filename": xml_file,
        "content": content,
    }