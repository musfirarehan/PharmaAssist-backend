from fastapi import APIRouter

from app.services.chatbot_service import ask_pharmacist
from app.utils.response import success_response


router = APIRouter()


@router.post("/chat")
def chat(data: dict):

    question = data.get(
        "question"
    )

    medicines = data.get(
        "medicines",
        []
    )


    response = ask_pharmacist(
        question,
        medicines
    )


    return success_response(
        response,
        "Chatbot response generated successfully"
    )