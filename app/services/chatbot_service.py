import json
import os

from dotenv import load_dotenv
from app.rag.retriever import get_medicine_knowledge

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
            model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash"),
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
        )
        return response.choices[0].message.content


def ask_pharmacist(question, medicines=None):
    medicines = medicines or []

    try:
        grounded_medicines = []
        for medicine in medicines:
            name = str(medicine.get("name", "")).strip()
            if not name:
                continue
            grounded_medicines.append(
                {
                    "prescription_data": medicine,
                    "knowledge_sources": get_medicine_knowledge(name, top_k=3),
                }
            )

        prompt = f"""
You are PharmaAssist AI, a professional pharmacist assistant.

Your role is to explain medicines clearly to patients.

Do not introduce yourself.
Do not say "Hello".
Do not repeat the patient's question.
Do not add unnecessary disclaimers.

Answer style:

- Start directly with the answer.
- Use short sections.
- Use bullet points when helpful.
- Keep responses under 150 words.
- Use simple patient-friendly language.

Rules:

1. Explain medicines using the provided authoritative medicine information.
2. Always mention the active ingredient if available.
3. Explain the actual medical purpose of the medicine.
4. Food instructions should explain administration guidance, not claim the medicine treats food-related problems.
5. Side effects must be clearly separated from the purpose of the medicine.
6. Do not invent indications.
7. Do not prescribe new medicines.
8. Do not change dosage.
9. Do not diagnose diseases.
10. If information is uncertain, advise consulting a pharmacist or doctor.
11. Do not invent information when the provided medicine information is empty or insufficient.
12. Keep answers patient-friendly and concise.

Medical safety rules:

1. Do not diagnose diseases.
2. Do not prescribe new medicines.
3. Do not change dosage.
4. Do not suggest stopping medication.
5. If information is unavailable, say:
"Please confirm with your pharmacist or doctor."

Current prescription medicines and authoritative medicine information:

{json.dumps(grounded_medicines, indent=2, default=str)}


Patient question:

{question}


Provide the answer:
"""

        answer = _generate_text(prompt)
        if not answer or not str(answer).strip():
            raise RuntimeError("Gemini returned an empty response.")

        return {
            "success": True,
            "answer": str(answer).strip(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }