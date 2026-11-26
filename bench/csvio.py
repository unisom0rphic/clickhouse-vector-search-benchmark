from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class CsvAppender:
    """Append-only CSV writer with stable headers and resumable output."""

    def __init__(self, path: str | Path, fields: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields)
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as stream:
                existing = next(csv.reader(stream), [])
            if existing != self.fields:
                raise ValueError(
                    f"CSV schema mismatch for {self.path}: existing={existing}, expected={self.fields}"
                )
            self._write_header = False
        else:
            self._write_header = True

    def append(self, row: Mapping[str, object]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields, extrasaction="ignore")
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow({field: row.get(field, "") for field in self.fields})
            stream.flush()

    def append_many(self, rows: Iterable[Mapping[str, object]]) -> None:
        for row in rows:
            self.append(row)


def load_completed_measurements(path: str | Path) -> set[tuple[str, str, str, int, int]]:
    path = Path(path)
    if not path.exists():
        return set()
    completed: set[tuple[str, str, str, int, int]] = set()
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            note = row.get("Run_notes", "")
            if note == "__WARMUP__" or note.startswith("ERROR:") or note == "query_log_timeout":
                continue
            if not row.get("Latency_ms"):
                continue
            try:
                completed.add((
                    row["Config_ID"],
                    row["Query_ID"],
                    row.get("Filter_ID", ""),
                    int(row["k_requested"]),
                    int(row["Repeat"]),
                ))
            except (KeyError, ValueError):
                continue
    return completed


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
