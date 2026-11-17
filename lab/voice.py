"""Fake voicemails for the transcriber: the same fake scenarios, spoken.

1. Script: Gemini rewrites each voicemail scenario as the exact words a
   person says (numbers, dates and phone numbers as words). The script is
   the truth for the word error rate (WER).
2. Audio: gemini-3.8-flash-tts speaks the script with one of 30 voices,
   picked by a hash of the id.
3. Phone line: phone() makes a clip sound like a phone call (8 kHz, the
   300-3400 Hz band, mu-law, a little noise). Test clips always get it;
   training clips get it at random.
4. The bar: gemini-3.5-transcribe on the same test clips.

Splits are the scenarios' own splits, so a test voicemail is a test
message for the decision and rerank students too.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import gemini
from .paths import DATA, SCENARIOS, read_jsonl, write_jsonl

VOICE = DATA / "voice"
SCRIPTS = VOICE / "scripts.jsonl"
AUDIO = VOICE / "audio"
BAR = VOICE / "gemini-transcripts.jsonl"
TTS = "gemini-3.8-flash-tts"
TRANSCRIBER = "gemini-3.5-transcribe"
RATE = 16000
VOICES = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel",
          "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima",
          "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"]

FILLERS = {"uh", "um", "uhh", "umm", "uhm", "er", "erm", "ah", "ahh", "hmm", "mm", "mhm", "ugh", "eh", "oh"}
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def say(n: int) -> list[str]:
    """0-99 as words."""
    return [ONES[n]] if n < 20 else [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])


def spoken_number(tok: str) -> list[str]:
    """A digit token as most people say it: years and 1-2 digit numbers as
    numbers, phone-like groups digit by digit, 4th as fourth."""
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)?", tok)
    if not m:
        return [tok]
    d, suffix = m.group(1), m.group(2)
    if len(d) <= 2 and not (len(d) == 2 and d[0] == "0"):
        out = say(int(d))
    elif len(d) == 4 and d[:2] in ("19", "20") and d[2] != "0":
        out = say(int(d[:2])) + say(int(d[2:]))
    elif len(d) == 3 and d[0] != "0" and d[1:] == "00":
        out = [ONES[int(d[0])], "hundred"]
    else:
        out = [ONES[int(c)] for c in d]
    if suffix:
        last = out[-1]
        out[-1] = ORDINAL.get(last, (last[:-1] + "ieth") if last.endswith("y") else last + "th")
    return out


def words(text: str) -> list[str]:
    """The words for the WER and the vocabulary, the same for the truth and
    every transcript: lower case, no punctuation, no apostrophes, no fillers,
    digits as spoken words, spelled letters joined ("d o b" = "dob")."""
    text = text.lower().replace("’", "'").replace("'", "")
    toks = re.findall(r"[a-z]+|\d+(?:st|nd|rd|th)?", re.sub(r"[-/.,]", " ", text))
    out = []
    for t in toks:
        out += spoken_number(t) if t[0].isdigit() else [t]
    joined, run = [], []
    for w in out + [""]:
        if len(w) == 1 and w not in ("a", "i"):
            run.append(w)
            continue
        if run:
            joined.append("".join(run))
            run = []
        if w and w not in FILLERS:
            joined.append(w)
    return joined


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(edits, reference words): the word-level edit distance."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1], len(ref)


# --- 1. scripts ---------------------------------------------------------------

SPOKEN = """Here are voicemail transcripts (fake data for a clinic). Rewrite each one as the exact words the caller says, for a text-to-speech voice:
- write every number, date, time, phone number, dose and abbreviation as spoken words ("five five five, two one four", "March fourth, nineteen sixty-two", "ten milligrams", "B P")
- correct the typos and misspellings, and write texting short forms as the full words a person says ("thx" is "thanks", "u" is "you", "pls" is "please", "callin" is "calling")
- keep the caller's own words, word order, punctuation and capital letters; add nothing and remove nothing else
- no digits and no symbols in the result

{items}

Answer with a JSON list, one object per transcript, in the same order: [{{"id": "...", "spoken": "..."}}]"""


def write_scripts(batch: list[dict]) -> list[dict]:
    items = "\n\n".join(f"id {m['id']}:\n{m['message']}" for m in batch)
    try:
        out = gemini.write(SPOKEN.format(items=items))
    except Exception as e:
        print(f"  ✗ scripts {batch[0]['id']}: {e}")
        return []
    by_id = {str(o.get("id")): str(o.get("spoken", "")).strip() for o in out if isinstance(o, dict)}
    rows = []
    for m in batch:
        s = by_id.get(m["id"], "")
        if s and not re.search(r"\d", s):  # digits left: the truth would be unclear
            rows.append({"id": m["id"], "split": m["split"], "style": m["style"], "sender": m["sender"], "spoken": s})
    return rows


def scripts() -> list[dict]:
    have = read_jsonl(SCRIPTS)
    done = {r["id"] for r in have}
    todo = [m for m in read_jsonl(SCENARIOS) if m["channel"] == "voicemail transcript" and m["id"] not in done]
    if todo:
        with ThreadPoolExecutor(8) as pool:
            new = [r for rows in pool.map(write_scripts, [todo[i:i + 10] for i in range(0, len(todo), 10)]) for r in rows]
        have += new
        write_jsonl(SCRIPTS, have)
        print(f"  {len(new)} spoken scripts by {gemini.WRITER} ({len(todo) - len(new)} dropped)")
    else:
        print(f"  {len(have)} spoken scripts already written")
    return have


# --- 2. audio -----------------------------------------------------------------


def voice_of(sid: str) -> str:
    return VOICES[zlib.crc32(sid.encode()) % len(VOICES)]


def speak(row: dict) -> bool:
    path = AUDIO / f"{row['id']}.wav"
    if path.exists():
        return True
    # Only the words to say. An instruction in the text ("Read this voicemail
    # aloud…", "Say quickly:") was itself read aloud in about half of the
    # clips, and this model takes no system instruction; the 30 voices give
    # the variety.
    body = {"contents": [{"parts": [{"text": row["spoken"]}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_of(row["id"])}}}}}
    try:
        out = gemini.post(f"models/{TTS}:generateContent", body)
        data = base64.b64decode(out["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
    except Exception as e:
        print(f"  ✗ speech {row['id']}: {type(e).__name__}: {str(e)[:200]}")
        return False
    path.write_bytes(data if data[:4] == b"RIFF" else wav_bytes(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768, 24000))
    return True


def audio(limit: int | None = None) -> int:
    rows = scripts()[:limit]
    AUDIO.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (AUDIO / f"{r['id']}.wav").exists()]
    ok = 0
    with ThreadPoolExecutor(6) as pool:
        for fut in as_completed([pool.submit(speak, r) for r in todo]):
            ok += fut.result()
    print(f"  {ok} clips spoken by {TTS} ({len(todo) - ok} failed); {sum(1 for _ in AUDIO.glob('*.wav'))} clips in {AUDIO.relative_to(DATA.parent)}")
    return ok


# --- 3. sound -----------------------------------------------------------------


def wav_bytes(x: np.ndarray, rate: int = RATE) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, x, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def load(data: bytes | str, rate: int = RATE) -> np.ndarray:
    """Mono float32 at `rate` Hz from WAV/FLAC/OGG bytes or a path."""
    import soundfile as sf
    import soxr

    x, sr = sf.read(io.BytesIO(data) if isinstance(data, bytes) else data, dtype="float32", always_2d=True)
    x = x.mean(1)
    return soxr.resample(x, sr, rate).astype(np.float32) if sr != rate else x


def phone(x: np.ndarray, seed: int, rate: int = RATE) -> np.ndarray:
    """A phone line: down to 8 kHz, the 300-3400 Hz band, 8-bit mu-law, a
    little line noise, back to 16 kHz."""
    import soxr

    rng = np.random.default_rng(seed)
    y = soxr.resample(x, rate, 8000)
    f = np.fft.rfft(y)
    hz = np.fft.rfftfreq(len(y), 1 / 8000)
    f[(hz < 300) | (hz > 3400)] = 0
    y = np.fft.irfft(f, len(y))
    y = y / (np.abs(y).max() + 1e-6) * rng.uniform(0.3, 0.9)
    mu = 255
    q = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((q + 1) / 2 * mu) / mu * 2 - 1
    y = np.sign(q) * np.expm1(np.abs(q) * np.log1p(mu)) / mu
    y = y + rng.normal(0, rng.uniform(0.002, 0.01), len(y))
    return soxr.resample(y.astype(np.float32), 8000, rate).astype(np.float32)


def clips(split: str) -> list[dict]:
    """[{id, words, spoken, wav}] for one split; wav is the clean 16 kHz audio."""
    out = []
    for r in read_jsonl(SCRIPTS):
        path = AUDIO / f"{r['id']}.wav"
        if r["split"] == split and path.exists():
            out.append({"id": r["id"], "spoken": r["spoken"], "words": words(r["spoken"]), "wav": path.read_bytes()})
    return out


# --- 4. the bar ---------------------------------------------------------------


def gemini_transcribe(clip: dict) -> dict | None:
    x = phone(load(clip["wav"]), seed=zlib.crc32(clip["id"].encode()))
    body = {"contents": [{"parts": [{"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav_bytes(x)).decode()}},
                                    {"text": "Transcribe this voicemail word for word. Write every number as spoken words, not digits."}]}]}
    try:
        out = gemini.post(f"models/{TRANSCRIBER}:generateContent", body)
        part = out["candidates"][0]["content"]["parts"][0]
        text = part.get("audioTranscription", {}).get("text") or part.get("text", "")
    except Exception as e:
        print(f"  ✗ {TRANSCRIBER} {clip['id']}: {type(e).__name__}: {str(e)[:200]}")
        return None
    return {"id": clip["id"], "text": text}


def bar() -> dict:
    """gemini-3.5-transcribe's WER on the test voicemails (phone line)."""
    have = {r["id"]: r for r in read_jsonl(BAR)}
    test = clips("test")
    todo = [c for c in test if c["id"] not in have]
    if todo:
        # Only this thread writes the file: as_completed hands the results here.
        with ThreadPoolExecutor(6) as pool, BAR.open("a") as f:
            for fut in as_completed([pool.submit(gemini_transcribe, c) for c in todo]):
                if (row := fut.result()) is not None:
                    f.write(json.dumps(row) + "\n")
                    have[row["id"]] = row
    edits = total = 0
    for c in test:
        if c["id"] in have:
            e, n = wer(c["words"], words(have[c["id"]]["text"]))
            edits, total = edits + e, total + n
    result = {"model": TRANSCRIBER, "clips": sum(c["id"] in have for c in test), "wer": round(edits / max(1, total), 4)}
    print(f"  {TRANSCRIBER}: WER {result['wer']:.1%} on {result['clips']} test voicemails (phone line)")
    return result
