from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class MedicineInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., min_length=1)
    medicines: List[MedicineInput] = Field(default_factory=list)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=5, max_length=254)
    username: str = Field(..., min_length=3, max_length=40)
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field(..., pattern="^(patient|pharmacist)$")
    license_number: str = Field(default="", max_length=80)
    license_authority: str = Field(default="", max_length=120)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    username: str = Field(..., min_length=3, max_length=40)
    password: str = Field(..., min_length=8, max_length=128)


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recipient_id: int = Field(..., gt=0)
    body: str = Field(..., min_length=1, max_length=4000)


class ReminderScheduleMedicine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""
    scheduled_times: List[str] = Field(default_factory=list)
    food_instructions: str = ""
    confidence: float | int = 100


class ReminderScheduleRequest(BaseModel):
    medicines: List[ReminderScheduleMedicine] = Field(default_factory=list)


class DoseActionRequest(BaseModel):
    action: str = Field(..., pattern="^(taken|snooze|skip)$")


class OCRMedicine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""
    confidence: float | int = 0
    possible_names: List[str] = Field(default_factory=list)


class OCRPrescriptionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hospital: str = ""
    doctor_name: str = ""
    patient_name: str = ""
    date: str = ""
    diagnosis: str = ""
    advice: List[str] = Field(default_factory=list)
    medicines: List[OCRMedicine] = Field(default_factory=list)
