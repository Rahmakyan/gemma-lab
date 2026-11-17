"""The teacher: Google's Gemini API over plain REST.

    decide(input, questions)  POST /v1beta/decisions with gemini-decision-1
    write(prompt)             generateContent with JSON output (GEMINI_WRITER)

Only fake data goes here. GEMINI_API_KEY comes from the environment or .env.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .paths import ROOT

BASE = "https://generativelanguage.googleapis.com/v1beta"
DECIDER = os.environ.get("GEMINI_DECIDER", "gemini-decision-1")
WRITER = os.environ.get("GEMINI_WRITER", "gemini-3.5-flash")


class Blocked(Exception):
    """Gemini refused the input (a safety filter): no answers."""


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set: put it in .env")
    return key


def post(path: str, body: dict, timeout: int = 180, retries: int = 5) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{BASE}/{path}", data, {"x-goog-api-key": api_key(), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            if e.code == 400 and any(w in text for w in ("safety", "prohibited_content", "blocklist", "content_blocked")):
                raise Blocked(text[:300]) from None
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(3 * 2 ** attempt)
                continue
            raise RuntimeError(f"Gemini HTTP {e.code} on {path}: {text[:300]}") from None
        except (TimeoutError, urllib.error.URLError):
            if attempt < retries:
                time.sleep(3 * 2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def decide(input: str, questions: dict) -> dict:
    """One call answers every question about one input."""
    return post("decisions", {"model": DECIDER, "input": input, "questions": questions})


def write(prompt: str, temperature: float = 1.0) -> object:
    """A JSON answer from the writer model."""
    out = post(f"models/{WRITER}:generateContent", {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json", "maxOutputTokens": 16384},
    }, timeout=300)
    parts = (out.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
