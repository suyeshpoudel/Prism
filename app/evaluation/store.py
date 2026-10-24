"""File-backed storage for reproducible evaluation-lab runs.

Each run is a single JSON document written atomically under
``PRISM_EVALUATION_RESULTS_DIR``. Reports are owner-scoped: listing and fetching
only expose the caller's runs. Run ids are restricted to a safe character set
so a request can never escape the results directory.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class EvaluationRunStore:
    def __init__(self, results_dir: Path | str) -> None:
        self._results_dir = Path(results_dir)

    @property
    def results_dir(self) -> Path:
        return self._results_dir

    def build_record(self, report: dict, *, owner_id: str) -> dict:
        created = datetime.now(timezone.utc)
        run_id = f"eval-{created.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        return {
            **report,
            "id": run_id,
            "created_at": created.isoformat(),
            "owner_id": owner_id,
        }

    def save(self, record: dict) -> dict:
        run_id = str(record["id"])
        if not RUN_ID_RE.match(run_id):
            raise ValueError("invalid evaluation run id")
        self._results_dir.mkdir(parents=True, exist_ok=True)
        target = self._results_dir / f"{run_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(target)
        return record

    def list(self, *, owner_id: str) -> list[dict]:
        if not self._results_dir.is_dir():
            return []
        summaries: list[dict] = []
        for path in self._results_dir.glob("eval-*.json"):
            record = self._read(path)
            if record is None or record.get("owner_id") != owner_id:
                continue
            summaries.append(
                {
                    "id": record.get("id", path.stem),
                    "created_at": record.get("created_at"),
                    "type": record.get("type"),
                    "config": record.get("config"),
                    "aggregate": record.get("aggregate"),
                    "configurations": record.get("configurations"),
                    "comparison": record.get("comparison"),
                }
            )
        summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return summaries

    def get(self, run_id: str, *, owner_id: str) -> dict | None:
        if not RUN_ID_RE.match(run_id):
            return None
        record = self._read(self._results_dir / f"{run_id}.json")
        if record is None or record.get("owner_id") != owner_id:
            return None
        return record

    @staticmethod
    def _read(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None
