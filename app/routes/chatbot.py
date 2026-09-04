from fastapi import APIRouter

from app.schemas import ChatRequest
from app.services.chatbot_service import ask_pharmacist
from app.utils.response import error_response, success_response


router = APIRouter()


@router.post("/chat")
def chat(payload: ChatRequest):
    response = ask_pharmacist(
        payload.question,
        [medicine.model_dump() for medicine in payload.medicines],
    )

    if isinstance(response, dict) and response.get("success") is False:
        return error_response(
            response.get("error"),
            "Chatbot generation failed",
        )

    return success_response(
        response,
        "Chatbot response generated successfully",
    )