from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


class ClickHouseUnavailable(RuntimeError):
    pass


def create_client(config: dict[str, Any]):
    """Create a clickhouse-connect client lazily so offline analysis needs no server client."""
    try:
        import clickhouse_connect
    except ImportError as exc:  # pragma: no cover - exercised on installations without the extra
        raise ClickHouseUnavailable(
            "clickhouse-connect is not installed. Run `pip install -e .` or install requirements.txt."
        ) from exc

    ch = config["clickhouse"]
    return clickhouse_connect.get_client(
        host=ch.get("host", "localhost"),
        port=int(ch.get("http_port", 8123)),
        username=ch.get("username", "default"),
        password=ch.get("password", ""),
        database=ch.get("database", "default"),
        connect_timeout=int(ch.get("connect_timeout", 10)),
        send_receive_timeout=int(ch.get("send_receive_timeout", 600)),
        client_name="chvecbench/0.1",
    )


def quote_identifier(value: str) -> str:
    if not value or not value.replace("_", "").isalnum() or value[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return f"`{value}`"


def qualified_table(database: str, table: str) -> str:
    return f"{quote_identifier(database)}.{quote_identifier(table)}"


def wait_for_query_log(
    client,
    query_id: str,
    timeout_seconds: float = 15.0,
    poll_interval: float = 0.25,
) -> dict[str, Any]:
    """Return the QueryFinish row corresponding to query_id, or an empty dict on timeout."""
    deadline = time.monotonic() + timeout_seconds
    query = """
        SELECT
            query_duration_ms,
            read_rows,
            read_bytes,
            memory_usage,
            type,
            exception,
            query
        FROM system.query_log
        WHERE query_id = {query_id:String}
          AND type = 'QueryFinish'
        ORDER BY event_time_microseconds DESC
        LIMIT 1
    """
    while time.monotonic() < deadline:
        try:
            client.command("SYSTEM FLUSH LOGS")
            result = client.query(query, parameters={"query_id": query_id})
            if result.result_rows:
                names = result.column_names
                return dict(zip(names, result.result_rows[0], strict=True))
        except Exception as exc:  # query log can become available shortly after query response
            LOG.debug("Waiting for query_log row %s: %s", query_id, exc)
        time.sleep(poll_interval)
    return {}


def get_setting(client, name: str) -> str:
    result = client.query(
        "SELECT value FROM system.settings WHERE name = {name:String} LIMIT 1",
        parameters={"name": name},
    )
    return str(result.result_rows[0][0]) if result.result_rows else ""


def get_parts_stats(client, database: str, table: str) -> tuple[int, int]:
    """Return active part count and total compressed data bytes."""
    result = client.query(
        """SELECT count(), coalesce(sum(bytes_on_disk), 0)
           FROM system.parts
           WHERE active AND database = {database:String} AND table = {table:String}""",
        parameters={"database": database, "table": table},
    )
    if not result.result_rows:
        return 0, 0
    parts, bytes_on_disk = result.result_rows[0]
    return int(parts), int(bytes_on_disk)



def drop_vector_similarity_index_cache(client) -> None:
    """Clear the resident HNSW index cache between method configurations."""
    client.command("SYSTEM DROP VECTOR SIMILARITY INDEX CACHE")

def get_vector_index_cache_bytes(client) -> int | None:
    """Metric name varies by ClickHouse version; return None if unavailable."""
    try:
        result = client.query(
            "SELECT value FROM system.metrics WHERE metric ILIKE '%VectorSimilarity%Cache%Bytes%' LIMIT 1"
        )
        return int(result.result_rows[0][0]) if result.result_rows else None
    except Exception:
        return None


def wait_until_healthy(config: dict[str, Any], timeout_seconds: int = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client = None
        try:
            client = create_client(config)
            client.query("SELECT 1")
            version = client.query("SELECT version()").result_rows[0][0]
            return {"version": str(version), "client": client}
        except Exception as exc:  # pragma: no cover - depends on external ClickHouse
            last_error = exc
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            time.sleep(1)
    raise ClickHouseUnavailable(f"ClickHouse did not become healthy: {last_error}")
