import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.counseling_service import generate_medicine_counseling
from app.services.ocr_service import extract_prescription_text
from app.utils.response import error_response, success_response


router = APIRouter()


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@router.post("/upload-prescription")
async def upload_prescription(file: UploadFile = File(...)):
    """
    End-to-end pipeline:

        Upload -> OCR -> RAG retrieval (inside counseling_service)
        -> Gemini counseling -> structured JSON for the frontend

    Matches the frontend's PrescriptionUploader, which POSTs to
    /api/v1/upload-prescription and reads data.data.ocr_result.medicines
    from the response.
    """

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                "Please upload a JPEG or PNG."
            ),
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    temp_path = None

    try:

        # -----------------------------------------
        # Save upload to a temp file (OCR service
        # expects a file path, not raw bytes)
        # -----------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # -----------------------------------------
        # Step 1: OCR
        # -----------------------------------------

        ocr_result = extract_prescription_text(temp_path)

        if not ocr_result or ocr_result.get("success") is False:
            return error_response(
                error=(
                    ocr_result.get("error")
                    if ocr_result
                    else "OCR failed to read the prescription."
                ),
                message="Could not read prescription image",
            )

        medicines = ocr_result.get("medicines", [])

        if not medicines:
            return success_response(
                data={
                    "ocr_result": ocr_result,
                    "counseling": {"medicines": []},
                },
                message="No medicines were detected in the prescription.",
            )

        # -----------------------------------------
        # Step 2 + 3: RAG retrieval + Gemini counseling
        # (retrieval happens inside generate_medicine_counseling,
        # which calls the retriever per medicine)
        # -----------------------------------------

        counseling_result = generate_medicine_counseling(medicines)

        if (
            isinstance(counseling_result, dict)
            and counseling_result.get("success") is False
        ):
            return error_response(
                error=counseling_result.get("error"),
                message="Counseling generation failed",
            )

        # -----------------------------------------
        # Step 4: return combined result to frontend
        # -----------------------------------------

        return success_response(
            data={
                "ocr_result": ocr_result,
                "counseling": counseling_result,
            },
            message="Prescription analyzed successfully",
        )

    except HTTPException:
        raise

    except Exception as exc:
        return error_response(
            error=str(exc),
            message="Unexpected error processing prescription",
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)