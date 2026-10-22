import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class OllamaClient:
    """
    Standard-library HTTP client for the local Ollama inference service.
    Zero external dependencies; communicates strictly with the local endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "gemma3:4b",
        timeout_seconds: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout_seconds

    def is_healthy(self) -> bool:
        """Checks whether the local Ollama service is reachable."""
        url = f"{self.base_url}/api/version"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return "version" in data
        except Exception:
            return False
        return False

    def list_models(self) -> list:
        """Retrieves list of locally downloaded models."""
        url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("models", [])
        except Exception:
            return []
        return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        format_type: Optional[str] = "json",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Executes a local inference call with optional structured output constraints.
        Returns the parsed JSON response object or a raw response dictionary.
        """
        url = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        if format_type == "json":
            payload["format"] = "json"

        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP error status {resp.status}")
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))
        except urllib.error.URLError as err:
            raise RuntimeError(
                f"Local Ollama service unreachable at {self.base_url}. Error: {err}"
            ) from err
        except TimeoutError as err:
            raise TimeoutError(f"Ollama inference timed out after {self.timeout}s.") from err

        raw_response = data.get("response", "")

        # If format_type is json, parse inner response text to a Python dictionary
        if format_type == "json":
            try:
                parsed_json = json.loads(raw_response)
                return {
                    "parsed": parsed_json,
                    "raw": raw_response,
                    "eval_count": data.get("eval_count", 0),
                    "total_duration_ns": data.get("total_duration", 0),
                    "eval_duration_ns": data.get("eval_duration", 0),
                }
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Model returned invalid JSON syntax: {raw_response[:200]}"
                ) from err

        return {
            "parsed": None,
            "raw": raw_response,
            "eval_count": data.get("eval_count", 0),
            "total_duration_ns": data.get("total_duration", 0),
            "eval_duration_ns": data.get("eval_duration", 0),
        }
