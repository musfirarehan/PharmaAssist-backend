from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from app.services.reminder_service import generate_reminders
from app.utils.response import success_response


router = APIRouter()


class Medicine(BaseModel):
    name: str
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""


@router.post("/reminders")
def create_reminders(
    medicines: List[Medicine]
):

    medicine_data = [
        medicine.model_dump()
        for medicine in medicines
    ]


    result = generate_reminders(
        medicine_data
    )


    return success_response(
        result,
        "Medicine reminders generated successfully"
    )