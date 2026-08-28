import os
import json
import tempfile
from dotenv import load_dotenv
from google import genai
from PIL import Image, ImageEnhance, ImageFilter

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def preprocess_image(image_path: str):
    """
    Enhance prescription image before OCR.
    """

    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert("L")

    # Increase contrast
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # Slight sharpening
    img = img.filter(ImageFilter.SHARPEN)

    # Save temporary processed image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(temp_file.name, quality=100)

    return temp_file.name


def clean_json_response(text: str):
    """
    Remove markdown formatting from Gemini output.
    """

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

        # Preprocess image
        processed_image = preprocess_image(image_path)

        with open(processed_image, "rb") as image_file:
            image_bytes = image_file.read()

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
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
                                "data": image_bytes
                            }
                        }
                    ]
                }
            ]
        )

        result = clean_json_response(response.text)

        try:
            return json.loads(result)

        except json.JSONDecodeError:

            return {
                "success": False,
                "error": "Invalid JSON returned by Gemini.",
                "raw_output": response.text
            }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

    finally:

        # Delete temporary processed image
        if processed_image and os.path.exists(processed_image):
            os.remove(processed_image)