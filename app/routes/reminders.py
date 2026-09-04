from typing import List

from fastapi import APIRouter, Header, HTTPException

from app.schemas import DoseActionRequest, MedicineInput, ReminderScheduleRequest
from app.services.auth_service import get_user_from_token
from app.services.reminder_service import (
    create_schedule,
    generate_reminders,
    get_adherence,
    get_pending_schedules,
    get_today_doses,
    update_dose,
    verify_schedule,
)
from app.utils.response import error_response, success_response


router = APIRouter()


@router.post("/reminders")
def create_reminders(
    medicines: List[MedicineInput],
):
    medicine_data = [
        medicine.model_dump()
        for medicine in medicines
    ]

    result = generate_reminders(medicine_data)

    return success_response(
        result,
        "Medicine reminders generated successfully",
    )


def _current_user(authorization: str | None):
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@router.post("/reminders/schedules")
def create_persisted_schedules(payload: ReminderScheduleRequest, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return success_response(create_schedule(user["id"], [medicine.model_dump() for medicine in payload.medicines], user["role"]), "Medication schedules created")


@router.get("/reminders/today")
def today_doses(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return success_response(get_today_doses(user["id"]), "Today’s medication doses")


@router.get("/reminders/adherence")
def adherence(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return success_response(get_adherence(user["id"]), "Medication adherence")


@router.patch("/reminders/doses/{dose_id}")
def dose_action(dose_id: int, payload: DoseActionRequest, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    try:
        return success_response(update_dose(user["id"], dose_id, payload.action), "Dose updated")
    except ValueError as exc:
        return error_response(str(exc), "Dose update failed")


@router.get("/reminders/pending")
def pending_schedules(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user["role"] != "pharmacist":
        raise HTTPException(status_code=403, detail="Pharmacist access required")
    return success_response(get_pending_schedules(), "Pending medication schedules")


@router.patch("/reminders/schedules/{schedule_id}/verify")
def verify_medication_schedule(schedule_id: int, approved: bool, edits: dict | None = None, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user["role"] != "pharmacist":
        raise HTTPException(status_code=403, detail="Pharmacist access required")
    try:
        return success_response(verify_schedule(schedule_id, approved, edits), "Medication schedule reviewed")
    except ValueError as exc:
        return error_response(str(exc), "Schedule review failed")