import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_chat_route_accepts_typed_payload_and_returns_envelope(self):
        with patch("app.routes.chatbot.ask_pharmacist", return_value={"success": True, "answer": "Take with food."}):
            response = self.client.post(
                "/api/v1/chat",
                json={
                    "question": "What is this medicine for?",
                    "medicines": [{"name": "Amoxicillin", "dosage": "500mg"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["answer"], "Take with food.")

    def test_chat_route_propagates_service_error(self):
        with patch("app.routes.chatbot.ask_pharmacist", return_value={"success": False, "error": "service failure"}):
            response = self.client.post(
                "/api/v1/chat",
                json={
                    "question": "What is this medicine for?",
                    "medicines": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"], "service failure")

    def test_counsel_route_propagates_service_error(self):
        with patch(
            "app.routes.counseling.generate_medicine_counseling",
            return_value={"success": False, "error": "counsel failed"},
        ):
            response = self.client.post(
                "/api/v1/counsel",
                json=[{"name": "Amoxicillin", "dosage": "500mg"}],
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"], "counsel failed")

    def test_reminders_route_accepts_typed_payload(self):
        with patch(
            "app.routes.reminders.generate_reminders",
            return_value={"reminders": [{"medicine": "Amoxicillin", "times": ["08:00 AM"]}]},
        ):
            response = self.client.post(
                "/api/v1/reminders",
                json=[{"name": "Amoxicillin", "frequency": "Once Daily"}],
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["reminders"][0]["medicine"], "Amoxicillin")

    def test_ocr_service_validates_payload_structure(self):
        mock_response = MagicMock()
        mock_response.text = '{"hospital": "", "doctor_name": "", "patient_name": "", "date": "", "diagnosis": "", "advice": [], "medicines": [{"name": "Amoxicillin", "dosage": "500mg", "frequency": "Twice Daily", "duration": "5 days", "instructions": "", "confidence": 0.9, "possible_names": []}]}'

        with patch("app.services.ocr_service.preprocess_image", return_value="/tmp/mock.jpg"), \
             patch("builtins.open", unittest.mock.mock_open(read_data=b"fake"), create=True), \
             patch("app.services.ocr_service.client.models.generate_content", return_value=mock_response):
            result = __import__("app.services.ocr_service", fromlist=["extract_prescription_text"]).extract_prescription_text("/tmp/test.jpg")

        self.assertEqual(result["medicines"][0]["name"], "Amoxicillin")
        self.assertEqual(result["medicines"][0]["dosage"], "500mg")


if __name__ == "__main__":
    unittest.main()
