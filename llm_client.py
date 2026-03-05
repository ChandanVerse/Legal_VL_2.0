"""LLM client abstraction for Ollama (local) and Gemini (cloud)."""

import json
import os
import random
import time

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


class LLMError(Exception):
    """Raised when LLM calls fail after all retries."""


def call_ollama(
    prompt: str,
    expect_json: bool = True,
    model: str | None = None,
    base_url: str | None = None,
    max_retries: int = 3,
    timeout: int = 120,
) -> dict | str:
    """Call Ollama /api/generate endpoint.

    Args:
        prompt: The prompt to send.
        expect_json: If True, request JSON format and parse response.
        model: Override OLLAMA_MODEL env var.
        base_url: Override OLLAMA_BASE_URL env var.
        max_retries: Number of retry attempts.
        timeout: Request timeout in seconds.

    Returns:
        Parsed dict if expect_json, else raw string.

    Raises:
        LLMError: After all retries are exhausted.
    """
    model = model or OLLAMA_MODEL
    base_url = base_url or OLLAMA_BASE_URL
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if expect_json:
        payload["format"] = "json"

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")

            if expect_json:
                return json.loads(response_text)
            return response_text

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = (2**attempt) + random.uniform(0, 1)
                time.sleep(wait)

    raise LLMError(f"Ollama call failed after {max_retries} attempts: {last_error}")


def call_gemini(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    temperature: float = 0.2,
) -> str:
    """Call Gemini via google-genai SDK.

    Args:
        prompt: User prompt text.
        system_prompt: Optional system instruction.
        model: Override GEMINI_MODEL env var.
        api_key: Override GEMINI_API_KEY env var.
        max_retries: Number of retry attempts.
        temperature: Generation temperature (0.2 for factual).

    Returns:
        Generated text string.

    Raises:
        LLMError: After all retries are exhausted.
    """
    api_key = api_key or GEMINI_API_KEY
    model = model or GEMINI_MODEL

    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Add it to .env file.")

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        temperature=temperature,
    )
    if system_prompt:
        config.system_instruction = system_prompt

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            retryable = any(
                kw in error_str
                for kw in ["429", "503", "resource_exhausted", "rate", "overloaded"]
            )
            if retryable and attempt < max_retries - 1:
                wait = (2**attempt) * 2 + random.uniform(0, 1)
                time.sleep(wait)
            elif not retryable:
                raise LLMError(f"Gemini call failed (non-retryable): {last_error}")

    raise LLMError(f"Gemini call failed after {max_retries} attempts: {last_error}")
