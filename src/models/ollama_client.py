import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


class OllamaClient:
    """
    Standard-library HTTP client for the local Ollama inference service.
    Zero external dependencies; communicates strictly with the local endpoint.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        configured_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            parsed_url = urllib.parse.urlsplit(configured_url)
            port = parsed_url.port
        except (TypeError, ValueError) as err:
            raise ValueError("Ollama base URL is malformed.") from err
        if (parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc or
                parsed_url.username or parsed_url.password or
                parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("Ollama base URL must use a loopback HTTP(S) address without embedded credentials.")
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("Ollama base URL port must be between 1 and 65535.")

        configured_timeout = timeout_seconds
        if configured_timeout is None:
            timeout_value = os.environ.get("OLLAMA_TIMEOUT_SECONDS", "90")
            configured_timeout = timeout_value
        try:
            configured_timeout = float(configured_timeout)
        except (TypeError, ValueError) as err:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a positive finite number.") from err
        if not math.isfinite(configured_timeout) or configured_timeout <= 0:
            raise ValueError("Ollama timeout must be greater than zero.")

        configured_model = model_name or os.environ.get("OLLAMA_MODEL", "gemma3:4b")
        if not configured_model.strip():
            raise ValueError("Ollama model name cannot be empty.")

        self.base_url = configured_url.rstrip("/")
        self.model_name = configured_model.strip()
        self.timeout = configured_timeout

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
        except urllib.error.HTTPError as err:
            try:
                detail = err.read(500).decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"Ollama returned HTTP {err.code}{suffix}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(
                f"Local Ollama service unreachable at {self.base_url}. Error: {err}"
            ) from err
        except TimeoutError as err:
            raise TimeoutError(f"Ollama inference timed out after {self.timeout}s.") from err
        except json.JSONDecodeError as err:
            raise ValueError("Ollama returned a malformed JSON response envelope.") from err

        if not isinstance(data, dict):
            raise ValueError("Ollama returned an invalid response: expected a JSON object.")
        raw_response = data.get("response", "")
        if not isinstance(raw_response, str):
            raise ValueError("Ollama returned an invalid response: 'response' must be text.")

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
            if not isinstance(parsed_json, (dict, list)):
                raise ValueError("Model returned JSON, but the structured response must be an object or array.")

        return {
            "parsed": None,
            "raw": raw_response,
            "eval_count": data.get("eval_count", 0),
            "total_duration_ns": data.get("total_duration", 0),
            "eval_duration_ns": data.get("eval_duration", 0),
        }
