from fastapi import APIRouter, Header, HTTPException, Query

from app.rag.retriever import get_medicine_knowledge
from app.services.auth_service import get_user_from_token
from app.utils.response import success_response


router = APIRouter()


@router.get("/knowledge")
def search_knowledge(
    medicine: str = Query(..., min_length=1),
    authorization: str | None = Header(default=None),
):
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if user["role"] != "pharmacist":
        raise HTTPException(status_code=403, detail="Pharmacist access required")
    return success_response(
        {"medicine": medicine, "sources": get_medicine_knowledge(medicine, top_k=5)},
        "Medicine knowledge retrieved",
    )