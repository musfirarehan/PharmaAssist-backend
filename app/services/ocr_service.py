import json
import os
import tempfile

from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter

from app.schemas import OCRPrescriptionResult

load_dotenv()

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


def _get_client():
    if genai is None:
        raise RuntimeError(
            "The google-genai package is not installed. Install backend dependencies first."
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file before using AI features."
        )

    return genai.Client(api_key=api_key)


def preprocess_image(image_path: str):
    img = Image.open(image_path)
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(temp_file.name, quality=100)
    return temp_file.name


def clean_json_response(text: str):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_prescription_text(image_path: str):
    processed_image = None

    try:
        client = _get_client()
        processed_image = preprocess_image(image_path)

        with open(processed_image, "rb") as image_file:
            image_bytes = image_file.read()

        response = client.models.generate_content(
            model=os.getenv("GEMINI_OCR_MODEL", "gemini-3.5-flash-lite"),
            config={"response_mime_type": "application/json"},
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": """
You are an expert medical prescription OCR assistant.

Your ONLY task is to accurately extract information that is visibly written.

Rules:
1. Never invent medicine names.
2. Never autocorrect handwriting.
3. Never diagnose diseases.
4. Never provide treatment advice.
5. If handwriting is unclear:
   - Leave the medicine name empty.
   - Set confidence below 70.
   - Suggest possible_names only if you genuinely see possible alternatives.
6. Preserve the exact spelling if readable.
7. Convert frequencies:
   OD -> Once Daily
   BD -> Twice Daily
   TDS -> Three Times Daily
   QID -> Four Times Daily
8. Return ONLY valid JSON.

Return this structure exactly:

{
  "hospital": "",
  "doctor_name": "",
  "patient_name": "",
  "date": "",
  "diagnosis": "",
  "advice": [],
  "medicines": [
    {
      "name": "",
      "dosage": "",
      "frequency": "",
      "duration": "",
      "instructions": "",
      "confidence": 0,
      "possible_names": []
    }
  ]
}
"""
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_bytes,
                            }
                        },
                    ],
                }
            ],
        )

        result = clean_json_response(getattr(response, "text", ""))

        try:
            payload = json.loads(result)
            parsed = OCRPrescriptionResult.model_validate(payload)
            return parsed.model_dump()
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid JSON returned by Gemini.",
                "raw_output": result,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"OCR payload validation failed: {exc}",
                "raw_output": result,
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

    finally:
        if processed_image and os.path.exists(processed_image):
            os.remove(processed_image)