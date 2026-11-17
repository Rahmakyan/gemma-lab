"""Training on a Modal GPU, with the same code (lab/).

train_on_modal() starts the app for the length of the call (nothing stays
deployed), trains one student there and returns its report and its files
(a LoRA adapter of a few MB, or the regressions). Several students train at
the same time, each in its own container.

Only fake scenarios and Gemini's labels go to Modal. Model weights stay in
the Modal volume gemma-lab-hf, so only the first run downloads them.
LAB_MODAL_GPU picks the GPUs, the first free one is used (default
L4,A10G,L40S: one type alone can wait long for capacity).
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import modal

GPU = os.environ.get("LAB_MODAL_GPU", "L4,A10G,L40S").split(",")

app = modal.App("gemma-lab")
cache = modal.Volume.from_name("gemma-lab-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.6", "transformers>=5.20", "accelerate>=1.0", "peft>=0.17", "numpy>=2.0", "sentence-transformers>=5.1",
                    "soundfile>=0.12", "soxr>=0.5", "pyarrow>=15", "pillow", "torchvision")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("lab")
)


@app.function(image=image, gpu=GPU, volumes={"/cache": cache}, timeout=3600)
def train(kind: str, rows: list[dict], questions: dict, cfg: dict) -> tuple[dict, dict[str, bytes]]:
    from lab.train import train_and_judge

    out = Path(tempfile.mkdtemp())
    report = train_and_judge(kind, rows, questions, cfg, out, log=lambda s: print(s, flush=True))
    import torch

    report["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    cache.commit()
    files = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    return report, files


@app.function(image=image, gpu=GPU, volumes={"/cache": cache}, timeout=3600)
def rerank(kind: str, data: dict, cfg: dict) -> tuple[dict, dict[str, bytes]]:
    import torch

    from lab.rerank import train_and_judge

    out = Path(tempfile.mkdtemp())
    report = train_and_judge(kind, data, cfg, out, log=lambda s: print(s, flush=True))
    report["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    cache.commit()
    return report, {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}


@app.function(image=image, gpu=os.environ.get("LAB_EAR_GPU", "H100"), volumes={"/cache": cache}, timeout=4 * 3600, cpu=16, memory=196608)
def transcriber(texts: dict, cfg: dict, stamp: str) -> dict:
    """Trains the reader and the punctuation model; the files go to the
    volume (word vectors are 300 MB), under out/<stamp>/."""
    import torch

    from lab.transcriber import train_and_judge
    from lab.voice import words

    cache.reload()
    voice = {}
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in Path(f"/cache/voice/{split}.jsonl").read_text().splitlines() if l.strip()]
        voice[split] = [{"id": r["id"], "words": words(r["spoken"]), "wav": Path(f"/cache/voice/audio/{r['id']}.wav").read_bytes()} for r in rows]
    out = Path(f"/cache/out/{stamp}")
    report = train_and_judge(voice, texts, cfg, out, Path("/cache/ear"), log=lambda s: print(s, flush=True))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    cache.commit()
    return report


def upload_voice(splits: dict[str, list[dict]], audio: Path):
    """The voicemail scripts and clips to the volume. A clip is skipped only
    when the volume has a file of the same name and size (clips made again
    keep their names, so the name alone is not enough)."""
    have = {}
    try:
        have = {Path(e.path).name: e.size for e in cache.listdir("/voice/audio")}
    except Exception:
        pass
    with cache.batch_upload(force=True) as b:
        for split, rows in splits.items():
            b.put_file(io.BytesIO("".join(json.dumps(r) + "\n" for r in rows).encode()), f"/voice/{split}.jsonl")
            for r in rows:
                path = audio / f"{r['id']}.wav"
                if have.get(path.name) != path.stat().st_size:
                    b.put_file(str(path), f"/voice/audio/{path.name}")


def download(remote: str, local: Path):
    local.mkdir(parents=True, exist_ok=True)
    for e in cache.listdir(remote):
        with open(local / Path(e.path).name, "wb") as f:
            for chunk in cache.read_file(e.path):
                f.write(chunk)


@app.function(image=image, volumes={"/cache": cache}, timeout=3600, cpu=4, memory=16384)
def libri_text(splits: list[str]) -> bytes:
    """The cleaned words of LibriSpeech transcripts, one line per clip, zlib
    packed. Reads only the text column of the parquet files (no audio)."""
    import zlib

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from lab.transcriber import LIBRI
    from lab.voice import words

    files = HfApi().list_repo_files(LIBRI, repo_type="dataset")
    lines = []
    for split in splits:
        folder = split if "/" in split else f"clean/{split}"
        for f in sorted(x for x in files if x.startswith(f"{folder}/") and x.endswith(".parquet")):
            lines += [" ".join(words(t)) for t in pq.read_table(hf_hub_download(LIBRI, f, repo_type="dataset"), columns=["text"])["text"].to_pylist()]
    return zlib.compress("\n".join(lines).encode())


def libri_text_on_modal(splits: list[str]) -> list[str]:
    import zlib

    with modal.enable_output(), app.run():
        return zlib.decompress(libri_text.remote(splits)).decode().split("\n")


def transcriber_on_modal(splits: dict, audio: Path, texts: dict, cfg: dict, stamp: str, local: Path) -> dict:
    print("  uploading the voicemails to the Modal volume")
    upload_voice(splits, audio)
    with modal.enable_output(), app.run():
        report = transcriber.remote(texts, cfg, stamp)
    print(f"  downloading the transcriber to {local}")
    download(f"/out/{stamp}", local)
    return report


def collect(jobs: list, calls: list) -> list:
    out = []
    for job, c in zip(jobs, calls):
        try:
            out.append(c.get())
        except Exception as e:
            print(f"  ✗ {job[0]} failed on Modal: {type(e).__name__}: {str(e)[:300]}")
            out.append(None)
    return out


def rerank_on_modal(jobs: list[tuple[str, dict]], data: dict) -> list[tuple[dict, dict[str, bytes]] | None]:
    with modal.enable_output(), app.run():
        return collect(jobs, [rerank.spawn(kind, data, cfg) for kind, cfg in jobs])


def train_on_modal(jobs: list[tuple[str, dict]], rows: list[dict], questions: dict) -> list[tuple[dict, dict[str, bytes]] | None]:
    """Trains each (kind, cfg) at the same time; results in the same order,
    None for a job that failed (the others are kept)."""
    with modal.enable_output(), app.run():
        return collect(jobs, [train.spawn(kind, rows, questions, cfg) for kind, cfg in jobs])
