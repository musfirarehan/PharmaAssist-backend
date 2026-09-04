from typing import List

from fastapi import APIRouter

from app.schemas import MedicineInput
from app.services.counseling_service import generate_medicine_counseling
from app.utils.response import error_response, success_response


router = APIRouter()


@router.post("/counsel")
def counsel(medicines: List[MedicineInput]):
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

    if isinstance(result, dict) and result.get("success") is False:
        return error_response(
            result.get("error"),
            "Counseling generation failed",
        )

    return success_response(
        result,
        "Medicine counseling generated successfully",
    )