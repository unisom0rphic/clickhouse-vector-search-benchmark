from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any


def _parse_docker_memory(value: str) -> float | None:
    value = value.strip().split("/")[0].strip()
    units = {"B": 1 / (1024**2), "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
    for unit, factor in units.items():
        if value.endswith(unit):
            try:
                return float(value[: -len(unit)].strip()) * factor
            except ValueError:
                return None
    return None


def snapshot_container(config: dict[str, Any], config_id: str, phase: str, client=None) -> dict[str, Any]:
    container_name = config.get("docker", {}).get("container_name", "chvecbench-clickhouse")
    memory_mb = None
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        memory_mb = _parse_docker_memory(result.stdout)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    cache_bytes = ""
    if client is not None:
        try:
            from .clickhouse import get_vector_index_cache_bytes
            value = get_vector_index_cache_bytes(client)
            cache_bytes = value if value is not None else ""
        except Exception:
            cache_bytes = ""
    return {
        "Config_ID": config_id,
        "Timestamp": datetime.now(timezone.utc).isoformat(),
        "Phase": phase,
        "Container_RSS_MB": memory_mb if memory_mb is not None else "",
        "Vector_Index_Cache_Bytes": cache_bytes,
        "Container_Limit_MB": int(float(config.get("docker", {}).get("memory_gb", 6)) * 1024),
    }
