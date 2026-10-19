import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CACHE_DIR = Path(".cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing. Put it in your .env file.")
        _client = genai.Client(api_key=key)
    return _client


def _parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    return json.loads(t)


def ask_json(prompt, system="", task="generic", retries=3, use_cache=True):
    # Same prompt -> same answer from disk cache: saves quota, makes demos repeatable.
    key = hashlib.sha256(f"{MODEL}|{system}|{prompt}".encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    from google.genai import types
    last_error = None
    for attempt in range(retries):
        try:
            resp = _get_client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            data = _parse_json(resp.text)
            cache_file.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception as e:  # network, quota, bad JSON
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed for task '{task}': {last_error}")
