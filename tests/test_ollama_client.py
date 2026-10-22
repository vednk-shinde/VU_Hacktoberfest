import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from src.models.ollama_client import OllamaClient


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=-1):
        return self.payload


class TestOllamaClientConfigurationAndErrors(unittest.TestCase):
    def test_environment_configuration_is_used(self):
        with patch.dict(os.environ, {
            "OLLAMA_BASE_URL": "http://127.0.0.1:11435/",
            "OLLAMA_MODEL": "gemma3:1b",
            "OLLAMA_TIMEOUT_SECONDS": "45",
        }):
            client = OllamaClient()
        self.assertEqual(client.base_url, "http://127.0.0.1:11435")
        self.assertEqual(client.model_name, "gemma3:1b")
        self.assertEqual(client.timeout, 45)

    def test_explicit_configuration_overrides_environment(self):
        with patch.dict(os.environ, {
            "OLLAMA_BASE_URL": "http://invalid:11434",
            "OLLAMA_MODEL": "wrong",
            "OLLAMA_TIMEOUT_SECONDS": "3",
        }):
            client = OllamaClient("http://localhost:11434", "gemma3:4b", 15)
        self.assertEqual((client.base_url, client.model_name, client.timeout),
                         ("http://localhost:11434", "gemma3:4b", 15))

    def test_invalid_or_credential_bearing_url_is_rejected(self):
        for url in ("file:///etc/passwd", "http://user:password@localhost:11434",
                    "http://example.com:11434", "not-a-url", "http://localhost:99999"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaClient(base_url=url)

    def test_invalid_timeout_and_empty_model_are_rejected(self):
        with patch.dict(os.environ, {"OLLAMA_TIMEOUT_SECONDS": "not-a-number"}):
            with self.assertRaisesRegex(ValueError, "OLLAMA_TIMEOUT_SECONDS"):
                OllamaClient()
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            OllamaClient(timeout_seconds=0)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            OllamaClient(timeout_seconds=float("inf"))
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            OllamaClient(model_name="  ")

    def test_generate_reports_http_status_without_calling_it_unreachable(self):
        error = HTTPError("http://localhost:11434/api/generate", 404, "Not Found", {}, io.BytesIO(b"model missing"))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 404: model missing"):
                OllamaClient().generate("hello")

    def test_generate_rejects_non_object_response_envelope(self):
        payload = json.dumps(["invalid envelope"]).encode()
        with patch("urllib.request.urlopen", return_value=_Response(payload)):
            with self.assertRaisesRegex(ValueError, "expected a JSON object"):
                OllamaClient().generate("hello")

    def test_generate_reports_malformed_http_json_envelope(self):
        with patch("urllib.request.urlopen", return_value=_Response(b"not-json")):
            with self.assertRaisesRegex(ValueError, "malformed JSON response envelope"):
                OllamaClient().generate("hello")

    def test_generate_rejects_non_text_model_response(self):
        payload = json.dumps({"response": {"unexpected": True}}).encode()
        with patch("urllib.request.urlopen", return_value=_Response(payload)):
            with self.assertRaisesRegex(ValueError, "response.*must be text"):
                OllamaClient().generate("hello")


if __name__ == "__main__":
    unittest.main()
