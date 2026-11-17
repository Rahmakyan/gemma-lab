from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ADAPTERS = ROOT / "adapters"
TUPLES = DATA / "tuples.jsonl"
SCENARIOS = DATA / "scenarios.jsonl"
LABELS = DATA / "labels.jsonl"
REPORTS = DATA / "reports"


def read_jsonl(path: Path) -> list[dict]:
    import json

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
