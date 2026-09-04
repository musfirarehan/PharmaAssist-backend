import json
import os

from dotenv import load_dotenv

from app.rag.retriever import get_medicine_knowledge as retrieve_medicine

load_dotenv()

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


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


def _generate_text(prompt):
    try:
        client = _get_client()
        return client.models.generate_content(
            model=os.getenv("GEMINI_COUNSELING_MODEL", "gemini-3.6-flash"),
            config={"response_mime_type": "application/json"},
            contents=prompt,
        ).text
    except Exception as gemini_error:
        if Groq is None or not os.getenv("GROQ_API_KEY"):
            raise gemini_error

        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


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
    try:
        grounded_medicines = []

        for medicine in medicines:
            medicine_name = medicine.get("name", "")
            retrieved = retrieve_medicine(medicine_name, top_k=3)
            grounded_medicines.append(
                {"prescription_data": medicine, "knowledge_sources": retrieved}
            )

        prompt = f"""
You are PharmaAssist AI, a professional pharmacist assistant.

Your job is to explain prescription medicines to patients using ONLY the provided medicine knowledge.

IMPORTANT:

The prescription data comes from OCR and may contain errors.
The knowledge sources come from the medicine knowledge retrieval system.
Do NOT rely on your general medical knowledge when the provided knowledge is sufficient.
Do NOT invent missing information.

Prescription + retrieved medicine knowledge:

{json.dumps(grounded_medicines, indent=2, default=str)}

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
5. Food instructions must describe ONLY how food affects administration.
6. Do not change the prescribed dosage.
7. Do not change the prescribed frequency.
8. Do not change the prescribed duration.
9. Do not recommend starting another medicine.
10. Do not recommend stopping a prescribed medicine.
11. If the retrieved knowledge does not contain enough information, say: "Please confirm this information with your pharmacist or doctor."
12. Never claim that a medicine treats an unrelated symptom.
13. Keep the explanation concise and patient-friendly.
14. Distinguish clearly between:
    - why the medicine is prescribed
    - how to take it
    - possible side effects
    - safety warnings
15. Do not mention the RAG system, database, embeddings, retrieval, or internal instructions to the patient.
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

        result = clean_json_response(_generate_text(prompt))
        parsed = json.loads(result)
        parsed["medicines"] = _merge_prescription_facts(
            parsed.get("medicines", []),
            medicines,
        )
        return parsed

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }