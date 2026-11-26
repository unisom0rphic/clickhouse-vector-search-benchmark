from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the benchmark configuration is incomplete or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(root)

    # Environment variables override secrets and connection details, not experiment design.
    ch = config.setdefault("clickhouse", {})
    ch["host"] = os.getenv("CLICKHOUSE_HOST", ch.get("host", "localhost"))
    ch["http_port"] = int(os.getenv("CLICKHOUSE_HTTP_PORT", ch.get("http_port", 8123)))
    ch["username"] = os.getenv("CLICKHOUSE_USER", ch.get("username", "default"))
    ch["password"] = os.getenv("CLICKHOUSE_PASSWORD", ch.get("password", ""))
    ch["database"] = os.getenv("CLICKHOUSE_DATABASE", ch.get("database", "default"))

    validate_config(config)
    return config


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ("project", "clickhouse", "source", "workload", "execution")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")

    dimension = int(config["source"].get("dimension", 0))
    if dimension <= 0:
        raise ConfigError("source.dimension must be a positive integer")

    k_values = config["workload"].get("k_values", [])
    if not k_values or any(int(k) <= 0 for k in k_values):
        raise ConfigError("workload.k_values must contain positive integers")

    sigma_values = config["workload"].get("sigma_targets", [])
    if any(not 0 < float(sigma) < 1 for sigma in sigma_values):
        raise ConfigError("workload.sigma_targets must be strictly between 0 and 1")

    if int(config["execution"].get("repeats", 0)) < 1:
        raise ConfigError("execution.repeats must be >= 1")

    max_k = max(int(k) for k in k_values)
    if int(config["workload"].get("max_ground_truth_k", max_k)) < max_k:
        raise ConfigError("workload.max_ground_truth_k must be >= max(workload.k_values)")

    if int(config["workload"].get("local_neighbors", 0)) < 1:
        raise ConfigError("workload.local_neighbors must be >= 1")

    cache_state = str(config["execution"].get("cache_state", "warm")).lower()
    if cache_state != "warm":
        raise ConfigError(
            "Only execution.cache_state='warm' is implemented. Cold-cache measurements "
            "require an explicit OS/page-cache control procedure and must not be emulated by relabeling."
        )
