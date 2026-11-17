"""The local decider: the trained student behind POST /v1/decisions, in the
same shape as the Gemini Decisions API. It listens on 127.0.0.1 only and
keeps nothing it reads, so sensitive messages never leave this machine.

    LAB_STUDENT=adapters/lora-… uv run uvicorn lab.serve:app --port 8250
    curl localhost:8250/v1/decisions -d '{"input": "…", "questions": {"q": {"type": "noul", "instructions": "…"}}}'
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .answers import answer, check
from .paths import ADAPTERS, ROOT

app = FastAPI(title="local Gemma decider")
lock = threading.Lock()
loading = threading.Lock()  # one model loads at a time, and only once
student = None
source = None
reranker = None
transcriber = None


def chosen(env: str, pointer_name: str) -> Path | None:
    """The folder in the env variable, else the folder that the pointer file
    adapters/<pointer_name> names, if it has a student.json, else none."""
    if os.environ.get(env):
        p = Path(os.environ[env])
        return p if p.is_absolute() else ROOT / p
    pointer = ADAPTERS / pointer_name
    if not pointer.exists():
        return None
    folder = ADAPTERS / pointer.read_text().strip()
    if not (folder / "student.json").exists():
        print(f"adapters/{pointer_name} points at {folder.name}, which is missing", flush=True)
        return None
    return folder


def current() -> Path | None:
    """The student to serve: LAB_STUDENT, else adapters/current, else none
    (plain Gemma, untaught)."""
    return chosen("LAB_STUDENT", "current")


def current_reranker() -> Path | None:
    """LAB_RERANKER, else adapters/rerank-current, else none."""
    return chosen("LAB_RERANKER", "rerank-current")


def current_transcriber() -> Path | None:
    """LAB_TRANSCRIBER, else adapters/transcriber-current, else none."""
    return chosen("LAB_TRANSCRIBER", "transcriber-current")


def get_reranker():
    """Loaded at the first /v1/rerank call, not at the start."""
    global reranker
    with loading:
        if reranker is None:
            from .rerank import load

            folder = current_reranker()
            if folder is None:
                raise HTTPException(503, {"error": {"message": "no reranker trained yet: run `runit rerank bench`", "code": "unavailable"}})
            reranker = (load(folder), folder.name)
    return reranker


def get_transcriber():
    """Loaded at the first /v1/transcribe call, not at the start."""
    global transcriber
    with loading:
        if transcriber is None:
            from .transcriber import Transcriber

            folder = current_transcriber()
            if folder is None:
                raise HTTPException(503, {"error": {"message": "no transcriber trained yet: run `runit voice train`", "code": "unavailable"}})
            transcriber = (Transcriber.load(folder), folder.name)
    return transcriber


def warm_all():
    """Loads the reranker and the transcriber, so that the first demo call
    does not wait for them. A missing model is skipped."""
    for load in (get_reranker, get_transcriber):
        try:
            load()
        except HTTPException:
            pass


def get():
    global student, source
    if student is None:
        from .students import load, make

        folder = current()
        student, source = (load(folder), folder.name) if folder else (make("lora"), "plain Gemma (untaught)")
    return student


class Decide(BaseModel):
    model: str | None = None
    input: str
    questions: dict[str, dict]


@app.on_event("startup")
def warm():
    get()


@app.get("/health")
def health():
    s = get()
    return {"student": source, "kind": s.kind, "model": s.name, "device": s.device, "root": str(ROOT)}


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    """A page that shows the three local models, with fake test data."""
    from . import demo

    if reranker is None or transcriber is None:
        threading.Thread(target=warm_all, daemon=True).start()
    return demo.page()


@app.get("/demo/audio/{clip}.wav")
def demo_audio(clip: str):
    from . import demo

    path = demo.audio(clip)
    if path is None:
        raise HTTPException(404, {"error": {"message": "no such test voicemail", "code": "not_found"}})
    return FileResponse(path, media_type="audio/wav")


class Rerank(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = None


@app.post("/v1/rerank")
def rerank(body: Rerank):
    """The documents in order of how well they match the query, with
    P(match) as relevance_score (the shape of common rerank APIs)."""
    r, name = get_reranker()
    started = time.time()
    with lock:
        scores = r.score(body.query, body.documents) if body.documents else []
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[: body.top_n or len(scores)]
    return {"id": f"rr_{uuid.uuid4().hex[:12]}", "model": f"{r.kind}:{name}",
            "results": [{"index": i, "relevance_score": round(scores[i], 4), "document": body.documents[i]} for i in order],
            "took_ms": int((time.time() - started) * 1000)}


@app.post("/v1/transcribe")
async def transcribe(request: Request, phone: bool = False):
    """The audio file (WAV, FLAC or OGG) is the request body; ?phone=true
    first makes it sound like a phone line. Loaded at the first call."""
    data = await request.body()
    if not data:
        raise HTTPException(400, {"error": {"message": "send the audio file as the request body", "code": "invalid_request"}})
    # The model work runs in a thread, so that /health and the page still
    # answer while a long clip is heard.
    t, name = await asyncio.to_thread(get_transcriber)
    from .voice import load, phone as phone_line

    try:
        x = load(data)
    except Exception as e:
        raise HTTPException(400, {"error": {"message": f"cannot read the audio: {e}", "code": "invalid_request"}}) from None
    if phone:
        x = phone_line(x, 0)
    started = time.time()

    def hear():
        with lock:
            return t.transcribe([x])[0]

    res = await asyncio.to_thread(hear)
    return {"id": f"tr_{uuid.uuid4().hex[:12]}", "model": f"transcriber:{name}", "text": res["text"], "words": res["words"],
            "audio_seconds": round(len(x) / 16000, 2), "took_ms": int((time.time() - started) * 1000)}


@app.post("/v1/decisions")
def decisions(body: Decide):
    if wrong := check(body.questions):
        raise HTTPException(400, {"error": {"message": wrong, "code": "invalid_request"}})
    s = get()
    started = time.time()
    with lock:
        try:
            probs = s.predict([body.input], body.questions)[0]
            answers = {name: answer(q, probs[name]) for name, q in body.questions.items()}
        except (KeyError, ValueError) as e:
            raise HTTPException(400, {"error": {"message": str(e).strip("'\""), "code": "invalid_request"}}) from None
    return {"id": f"dec_{uuid.uuid4().hex[:12]}", "model": f"{s.kind}:{source}",
            "answers": answers,
            "took_ms": int((time.time() - started) * 1000)}
