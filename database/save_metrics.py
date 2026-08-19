from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_metrics(company: str, year: int | None, metrics: dict[str, Any]) -> None:
    """Persist metrics in a JSON file under data/metrics."""
    output_dir = Path("data/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{company}_{year if year is not None else 'all'}.json"
    out_file = output_dir / file_name
    payload = {
        "company": company,
        "year": year,
        "metrics": metrics,
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metrics to {out_file}")
