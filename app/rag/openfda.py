import os
import json
import requests
from pathlib import Path

BASE_URL = "https://api.fda.gov/drug/label.json"

# Folder to cache API responses
CACHE_DIR = Path("app/data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def search_medicine(medicine_name: str):
    """
    Search OpenFDA by brand name first,
    then generic name if needed.
    """

    medicine_name = medicine_name.strip()

    cache_file = CACHE_DIR / f"{medicine_name.lower().replace(' ', '_')}.json"

    # Return cached data if available
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    queries = [
        f'openfda.brand_name:"{medicine_name}"',
        f'openfda.generic_name:"{medicine_name}"'
    ]

    for query in queries:

        try:

            response = requests.get(
                BASE_URL,
                params={
                    "search": query,
                    "limit": 1
                },
                timeout=20
            )

            if response.status_code != 200:
                continue

            data = response.json()

            if "results" not in data:
                continue

            # Save cache
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            return data

        except Exception as e:
            print(e)

    return None