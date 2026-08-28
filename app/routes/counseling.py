from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.counseling_service import generate_medicine_counseling
from app.utils.response import success_response


router = APIRouter()


class Medicine(BaseModel):
    name: str
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""


@router.post("/counsel")
def counsel(medicines: List[Medicine]):
    """
    Standalone counseling endpoint - e.g. to regenerate counseling
    for medicines the frontend already has (without re-uploading
    the prescription image).

    NOTE: original version took a single `medicine: dict`, but
    generate_medicine_counseling() iterates over `medicines` as a
    list, so a single dict would have broken on the first call.
    """

    medicine_data = [
        medicine.model_dump()
        for medicine in medicines
    ]

    result = generate_medicine_counseling(medicine_data)

    return success_response(
        result,
        "Medicine counseling generated successfully"
    )