"""LLM client abstraction for Ollama (local) and Gemini (cloud)."""

import json
import os
import random
import time

import threading

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "legal-mistral")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_gemini_client: genai.Client | None = None
_gemini_lock = threading.Lock()


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    with _gemini_lock:
        if _gemini_client is None:
            if not GEMINI_API_KEY:
                raise LLMError("GEMINI_API_KEY is not set. Add it to .env file.")
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


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

    Returns:
        Parsed dict if expect_json, else raw string.

    Raises:
        LLMError: After all retries are exhausted.
    """
    model = model or OLLAMA_MODEL
    base_url = base_url or OLLAMA_BASE_URL
    url = f"{base_url}/api/generate"

    payload = {"model": model, "prompt": prompt, "stream": False}
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
                time.sleep((2**attempt) + random.uniform(0, 1))

    raise LLMError(f"Ollama call failed after {max_retries} attempts: {last_error}")


def call_gemini(
    contents: list[types.Content],
    system_instruction: str | None = None,
    tools: list[types.Tool] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> types.GenerateContentResponse:
    """Call Gemini generate_content with optional tool definitions.

    Returns the full GenerateContentResponse.
    """
    client = _get_gemini_client()
    model = model or GEMINI_MODEL

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            retryable = any(
                kw in error_str for kw in ["429", "503", "rate", "overloaded", "resource"]
            )
            if retryable and attempt < max_retries - 1:
                time.sleep((2**attempt) * 2 + random.uniform(0, 1))
            elif not retryable:
                raise LLMError(f"Gemini call failed (non-retryable): {last_error}")

    raise LLMError(f"Gemini call failed after {max_retries} attempts: {last_error}")


def _response_text(response: types.GenerateContentResponse) -> str:
    """Safely extract text from a Gemini response (returns '' if blocked/empty)."""
    try:
        return response.text or ""
    except ValueError:
        return ""


def call_gemini_text(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
) -> str:
    """Simple Gemini text call: prompt in, text out."""
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    return _response_text(call_gemini(contents, system_instruction=system_prompt, temperature=temperature))


def call_gemini_chat(
    messages: list[dict],
    temperature: float = 0.2,
) -> str:
    """Gemini multi-turn chat from OpenAI-style messages list.

    Accepts {role, content} dicts with roles 'system', 'user', 'assistant'.
    """
    system_instruction = None
    contents: list[types.Content] = []

    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_instruction = text
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=gemini_role, parts=[types.Part(text=text)]))

    return _response_text(call_gemini(contents, system_instruction=system_instruction, temperature=temperature))
