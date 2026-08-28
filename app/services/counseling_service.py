import os
import json

from dotenv import load_dotenv
from google import genai

from app.rag.retriever import get_medicine_knowledge as retrieve_medicine


load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def clean_json_response(text: str):

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    if text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def _merge_prescription_facts(counseling_medicines, original_medicines):
    """
    Gemini's output schema never included dosage/frequency/duration,
    so the frontend always showed "Not detected" for dosage no
    matter how good the RAG data was.

    Rather than asking the LLM to copy these fields through (which
    it's told not to modify anyway, per the prompt's own safety
    rules), merge them in deterministically here. This guarantees
    the prescribed dosage shown to the patient is exactly what the
    prescription said - no LLM involved in that specific value.
    """

    by_name = {
        medicine.get("name", "").strip().lower(): medicine
        for medicine in original_medicines
    }

    for medicine in counseling_medicines:

        key = medicine.get("name", "").strip().lower()
        original = by_name.get(key)

        if not original:
            continue

        medicine["dosage"] = original.get("dosage", "")
        medicine["frequency"] = original.get("frequency", "")
        medicine["duration"] = original.get("duration", "")

    return counseling_medicines


def generate_medicine_counseling(medicines):

    """
    Generate medicine counseling using retrieved
    medicine knowledge as grounding context.
    """

    try:

        grounded_medicines = []

        # -----------------------------------------
        # RETRIEVE KNOWLEDGE FOR EACH MEDICINE
        # -----------------------------------------

        for medicine in medicines:

            medicine_name = medicine.get(
                "name",
                ""
            )

            retrieved = retrieve_medicine(
                medicine_name,
                top_k=3
            )

            grounded_medicines.append({
                "prescription_data": medicine,
                "knowledge_sources": retrieved
            })


        # -----------------------------------------
        # BUILD GROUNDED PROMPT
        # -----------------------------------------

        prompt = f"""
You are PharmaAssist AI, a professional pharmacist
assistant.

Your job is to explain prescription medicines to
patients using ONLY the provided medicine knowledge.

IMPORTANT:

The prescription data comes from OCR and may contain
errors.

The knowledge sources come from the medicine knowledge
retrieval system.

Do NOT rely on your general medical knowledge when the
provided knowledge is sufficient.

Do NOT invent missing information.

Prescription + retrieved medicine knowledge:

{json.dumps(
    grounded_medicines,
    indent=2,
    default=str
)}


For EACH medicine provide:

- name
- active_ingredient
- category
- purpose
- how_to_take
- food_instructions
- common_side_effects
- serious_warnings
- storage
- counseling_points


STRICT SAFETY RULES:

1. NEVER invent a medical indication.

2. NEVER infer the purpose from the medicine name alone.

3. Use the retrieved knowledge as the primary source.

4. Do not treat side effects as medical indications.

5. Food instructions must describe ONLY how food affects
   administration.

6. Do not change the prescribed dosage.

7. Do not change the prescribed frequency.

8. Do not change the prescribed duration.

9. Do not recommend starting another medicine.

10. Do not recommend stopping a prescribed medicine.

11. If the retrieved knowledge does not contain enough
    information, say:

    "Please confirm this information with your
    pharmacist or doctor."

12. Never claim that a medicine treats an unrelated
    symptom.

13. Keep the explanation concise and patient-friendly.

14. Distinguish clearly between:
    - why the medicine is prescribed
    - how to take it
    - possible side effects
    - safety warnings

15. Do not mention the RAG system, database, embeddings,
    retrieval, or internal instructions to the patient.

16. Return ONLY valid JSON.

Return exactly this structure:

{{
    "medicines": [
        {{
            "name": "",
            "active_ingredient": "",
            "category": "",
            "purpose": "",
            "how_to_take": "",
            "food_instructions": "",
            "common_side_effects": [],
            "serious_warnings": [],
            "storage": "",
            "counseling_points": []
        }}
    ]
}}
"""


        # -----------------------------------------
        # GEMINI
        # -----------------------------------------

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )


        result = clean_json_response(
            response.text
        )

        parsed = json.loads(result)

        # -----------------------------------------
        # MERGE DOSAGE/FREQUENCY/DURATION FROM OCR
        # -----------------------------------------

        parsed["medicines"] = _merge_prescription_facts(
            parsed.get("medicines", []),
            medicines,
        )

        return parsed


    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }