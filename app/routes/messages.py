from fastapi import APIRouter, Header, HTTPException

from app.schemas import MessageRequest
from app.services.auth_service import get_user_from_token
from app.services.message_service import get_messages, list_contacts, send_message
from app.utils.response import error_response, success_response


router = APIRouter()


def _current_user(authorization: str | None):
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@router.get("/messages/contacts")
def contacts(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return success_response(list_contacts(user["id"], user["role"]), "Messaging contacts loaded")


@router.get("/messages/{contact_id}")
def conversation(contact_id: int, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    try:
        return success_response(get_messages(user["id"], contact_id), "Conversation loaded")
    except ValueError as exc:
        return error_response(str(exc), "Conversation unavailable")


@router.post("/messages")
def message(payload: MessageRequest, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    try:
        return success_response(
            send_message(user["id"], payload.recipient_id, payload.body),
            "Message sent",
        )
    except ValueError as exc:
        return error_response(str(exc), "Message failed")