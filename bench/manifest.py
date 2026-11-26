from __future__ import annotations

import hashlib
import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
            check=True, timeout=5,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _docker_image_id(container_name: str) -> str | None:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Image}}", container_name],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def write_run_manifest(config: dict[str, Any], clickhouse_version: str, command: str) -> Path:
    root = Path(config["_project_root"])
    output_dir = root / config["project"].get("output_dir", "results") / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config["_config_path"])
    ch = config["clickhouse"]
    prepared_manifest = root / "data" / "prepared" / "workload_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_commit": _git_commit(root),
        "config_file": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else config_path.name,
        "config_sha256": _sha256(config_path),
        "clickhouse": {
            "version": clickhouse_version,
            "host": ch.get("host"),
            "http_port": ch.get("http_port"),
            "database": ch.get("database"),
            "max_threads": ch.get("max_threads"),
            "max_limit_for_vector_search_queries": ch.get("max_limit_for_vector_search_queries"),
        },
        "docker": {
            "container_name": config.get("docker", {}).get("container_name", "chvecbench-clickhouse"),
            "image_tag": os.getenv("CLICKHOUSE_VERSION", "26.8.44.3"),
            "image_id": _docker_image_id(config.get("docker", {}).get("container_name", "chvecbench-clickhouse")),
            "memory_gb": config.get("docker", {}).get("memory_gb", 6),
            "cpus": config.get("docker", {}).get("cpus", 4),
        },
        "experiment": {
            key: value for key, value in config.items()
            if key not in {"_config_path", "_project_root", "clickhouse"}
        },
        "workload_manifest": json.loads(prepared_manifest.read_text(encoding="utf-8")) if prepared_manifest.exists() else None,
    }
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_dir = output_dir / "run_manifests"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / f"{command}_{stamp}.json"
    archived.write_text(manifest_json, encoding="utf-8")
    latest = output_dir / "run_manifest.json"
    latest.write_text(manifest_json, encoding="utf-8")
    return archived
