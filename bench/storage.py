from __future__ import annotations

import logging
import re
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .clickhouse import get_parts_stats, qualified_table
from .math_utils import float32_to_bfloat16_roundtrip

LOG = logging.getLogger(__name__)


def _dataset_table_prefix(config: dict[str, Any]) -> str:
    base = config["source"].get("table_name", "movies")
    manifest = Path(config["_project_root"]) / "data" / "prepared" / "workload_manifest.json"
    if manifest.exists():
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            signature = str(metadata.get("dataset_signature", ""))
            if signature:
                return f"{base}_{signature[:10]}"
        except (OSError, ValueError, TypeError):
            LOG.warning("Could not read dataset signature from %s", manifest)
    return str(base)


def _table_name(config: dict[str, Any], suffix: str) -> str:
    base = _dataset_table_prefix(config)
    name = f"{base}_{suffix}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe table name: {name}")
    return name


def table_names(config: dict[str, Any]) -> dict[str, str]:
    return {
        "exact": _table_name(config, "exact"),
        "qbit": _table_name(config, "qbit"),
        "hnsw_prefix": _table_name(config, "hnsw"),
    }


def hnsw_table_name(config: dict[str, Any], m: int, ef_build: int) -> str:
    hnsw = config.get("hnsw", {})
    quantization = str(hnsw.get("quantization", "bf16")).lower()
    granularity = int(hnsw.get("granularity", 100000000))
    dimension = int(config["source"].get("dimension", 0))
    return _table_name(
        config,
        f"hnsw_m{int(m)}_ef{int(ef_build)}_{quantization}_g{granularity}_d{dimension}",
    )


def _ensure_database(client, database: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", database):
        raise ValueError(f"Unsafe database name: {database}")
    client.command(f"CREATE DATABASE IF NOT EXISTS `{database}`")



def get_vector_index_size_bytes(client, database: str, table: str) -> int | None:
    """Return the on-disk compressed size of the vector similarity index."""
    try:
        result = client.query(
            """SELECT coalesce(sum(data_compressed_bytes), 0)
               FROM system.data_skipping_indices
               WHERE database = {database:String}
                 AND table = {table:String}
                 AND type = 'vector_similarity'""",
            parameters={"database": database, "table": table},
        )
        if result.result_rows:
            return int(result.result_rows[0][0])
    except Exception:
        pass
    # Compatibility fallback for versions where system.data_skipping_indices differs.
    try:
        result = client.query(
            """SELECT coalesce(sum(secondary_indices_compressed_bytes), 0)
               FROM system.parts
               WHERE active AND database = {database:String} AND table = {table:String}""",
            parameters={"database": database, "table": table},
        )
        return int(result.result_rows[0][0]) if result.result_rows else None
    except Exception:
        return None

def _create_exact_table(client, config: dict[str, Any]) -> None:
    src = config["source"]
    ch = config["clickhouse"]
    table = qualified_table(ch["database"], table_names(config)["exact"])
    id_col = src.get("columns", {}).get("id", "mid")
    attr_col = src.get("columns", {}).get("filter_attribute", "avgrating")
    vec_col = src.get("columns", {}).get("vector", "mvector")
    for col in (id_col, attr_col, vec_col):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", col):
            raise ValueError(f"Unsafe column name: {col}")
    dim = int(src["dimension"])
    client.command(
        f"""CREATE TABLE IF NOT EXISTS {table}
        (
            `{id_col}` String,
            `{attr_col}` Float32,
            `{vec_col}` Array(BFloat16)
        ) ENGINE = MergeTree ORDER BY `{id_col}`"""
    )


def _existing_rows(client, config: dict[str, Any]) -> int:
    table = qualified_table(config["clickhouse"]["database"], table_names(config)["exact"])
    return int(client.query(f"SELECT count() FROM {table}").result_rows[0][0])


def import_dataset(client, config: dict[str, Any], dataset=None) -> dict[str, Any]:
    """Import a BFloat16-rounded copy; original HDF5 files are never modified."""
    from .dataset import load_movie_dataset

    dataset = dataset or load_movie_dataset(config)
    ch = config["clickhouse"]
    src = config["source"]
    id_col = src.get("columns", {}).get("id", "mid")
    attr_col = src.get("columns", {}).get("filter_attribute", "avgrating")
    vec_col = src.get("columns", {}).get("vector", "mvector")
    _ensure_database(client, ch["database"])
    _create_exact_table(client, config)
    current = _existing_rows(client, config)
    if current:
        if current != dataset.size:
            raise RuntimeError(
                f"Table already contains {current} rows but current dataset has {dataset.size}. "
                "Use `reset-tables` only when you intend to delete benchmark tables."
            )
        LOG.info("Exact table already contains %d rows; skipping import", current)
        return {"rows": current, "import_seconds": 0.0, "skipped": True}

    batch_size = int(src.get("insert_batch_size", 512))
    start = time.perf_counter()
    table = qualified_table(ch["database"], table_names(config)["exact"])
    for start_idx in range(0, dataset.size, batch_size):
        end_idx = min(start_idx + batch_size, dataset.size)
        batch_vectors = float32_to_bfloat16_roundtrip(dataset.vectors[start_idx:end_idx])
        batch = [
            [
                str(dataset.ids[index]),
                float(dataset.attributes[attr_col][index]),
                batch_vectors[index - start_idx].tolist(),
            ]
            for index in range(start_idx, end_idx)
        ]
        client.insert(
            table_names(config)["exact"],
            batch,
            column_names=[id_col, attr_col, vec_col],
            database=ch["database"],
        )
        LOG.info("Imported %d/%d rows", end_idx, dataset.size)
    elapsed = time.perf_counter() - start
    return {"rows": dataset.size, "import_seconds": elapsed, "skipped": False}


def _create_qbit(client, config: dict[str, Any]) -> float:
    ch, src = config["clickhouse"], config["source"]
    names = table_names(config)
    exact = qualified_table(ch["database"], names["exact"])
    qbit = qualified_table(ch["database"], names["qbit"])
    id_col = src.get("columns", {}).get("id", "mid")
    attr_col = src.get("columns", {}).get("filter_attribute", "avgrating")
    vec_col = src.get("columns", {}).get("vector", "mvector")
    dim = int(src["dimension"])
    client.command("SET allow_experimental_qbit_type = 1")
    client.command(f"DROP TABLE IF EXISTS {qbit}")
    client.command(
        f"""CREATE TABLE {qbit}
        (
            `{id_col}` String,
            `{attr_col}` Float32,
            `{vec_col}` QBit(BFloat16, {dim})
        ) ENGINE = MergeTree ORDER BY `{id_col}`"""
    )
    started = time.perf_counter()
    # ClickHouse performs the Array(BFloat16) -> QBit conversion on insertion.
    client.command(f"INSERT INTO {qbit} SELECT `{id_col}`, `{attr_col}`, `{vec_col}` FROM {exact}")
    return time.perf_counter() - started


def _create_hnsw(client, config: dict[str, Any], m: int, ef_build: int) -> tuple[str, float]:
    ch, src = config["clickhouse"], config["source"]
    name = hnsw_table_name(config, m, ef_build)
    exact = qualified_table(ch["database"], table_names(config)["exact"])
    table = qualified_table(ch["database"], name)
    id_col = src.get("columns", {}).get("id", "mid")
    attr_col = src.get("columns", {}).get("filter_attribute", "avgrating")
    vec_col = src.get("columns", {}).get("vector", "mvector")
    dim = int(src["dimension"])
    granularity = int(config.get("hnsw", {}).get("granularity", 100000000))
    quantization = config.get("hnsw", {}).get("quantization", "bf16")
    client.command(f"DROP TABLE IF EXISTS {table}")
    client.command(
        f"""CREATE TABLE {table}
        (
            `{id_col}` String,
            `{attr_col}` Float32,
            `{vec_col}` Array(BFloat16),
            INDEX ann_idx `{vec_col}` TYPE vector_similarity(
                'hnsw', 'L2Distance', {dim}, '{quantization}', {int(m)}, {int(ef_build)}
            ) GRANULARITY {granularity}
        ) ENGINE = MergeTree ORDER BY `{id_col}`"""
    )
    started = time.perf_counter()
    client.command(
        f"INSERT INTO {table} (`{id_col}`, `{attr_col}`, `{vec_col}`) "
        f"SELECT `{id_col}`, `{attr_col}`, `{vec_col}` FROM {exact}"
    )
    return name, time.perf_counter() - started


def build_indexes(client, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build QBit and all requested HNSW physical index variants."""
    ch = config["clickhouse"]
    tables: list[dict[str, Any]] = []
    exact_name = table_names(config)["exact"]
    exact_parts, exact_size = get_parts_stats(client, ch["database"], exact_name)
    tables.append({"table": exact_name, "method": "Exact", "build_seconds": 0.0, "parts": exact_parts, "size_bytes": exact_size, "index_size_bytes": None})
    expected_rows = int(client.query(f"SELECT count() FROM {qualified_table(ch['database'], exact_name)}").result_rows[0][0])
    if config.get("qbit", {}).get("enabled", True):
        qbit_name = table_names(config)["qbit"]
        qbit_table = qualified_table(ch["database"], qbit_name)
        qbit_exists = client.query(
            "SELECT count() FROM system.tables WHERE database={db:String} AND name={name:String}",
            parameters={"db": ch["database"], "name": qbit_name},
        ).result_rows[0][0]
        qbit_rows = 0
        if qbit_exists:
            qbit_rows = int(client.query(f"SELECT count() FROM {qbit_table}").result_rows[0][0])
        if not qbit_exists or qbit_rows != expected_rows:
            seconds = _create_qbit(client, config)
        else:
            seconds = 0.0
        parts, size = get_parts_stats(client, ch["database"], qbit_name)
        tables.append({"table": qbit_name, "method": "QBit", "build_seconds": seconds, "parts": parts, "size_bytes": size, "index_size_bytes": None})

    hnsw_cfg = config.get("hnsw", {})
    if hnsw_cfg.get("enabled", True):
        for m in hnsw_cfg.get("m", [32]):
            for ef in hnsw_cfg.get("ef_construction", [128]):
                name = hnsw_table_name(config, int(m), int(ef))
                exists = client.query(
                    "SELECT count() FROM system.tables WHERE database={db:String} AND name={name:String}",
                    parameters={"db": ch["database"], "name": name},
                ).result_rows[0][0]
                hnsw_rows = 0
                if exists:
                    hnsw_rows = int(client.query(f"SELECT count() FROM {qualified_table(ch['database'], name)}").result_rows[0][0])
                if exists and hnsw_rows == expected_rows:
                    seconds = 0.0
                else:
                    name, seconds = _create_hnsw(client, config, int(m), int(ef))
                parts, size = get_parts_stats(client, ch["database"], name)
                index_size = get_vector_index_size_bytes(client, ch["database"], name)
                tables.append({"table": name, "method": "HNSW", "M": int(m), "ef_construction": int(ef), "build_seconds": seconds, "parts": parts, "size_bytes": size, "index_size_bytes": index_size})
    return tables


def reset_tables(client, config: dict[str, Any]) -> None:
    database = config["clickhouse"]["database"]
    names = table_names(config)
    # All tables owned by this benchmark start with this dedicated prefix,
    # including dataset-fingerprinted Exact/QBit/HNSW tables.
    prefix = _dataset_table_prefix(config) + "_"
    result = client.query(
        "SELECT name FROM system.tables WHERE database={db:String} AND (name={exact:String} OR name={qbit:String} OR startsWith(name, {prefix:String}))",
        parameters={"db": database, "exact": names["exact"], "qbit": names["qbit"], "prefix": prefix},
    )
    for (name,) in result.result_rows:
        client.command(f"DROP TABLE IF EXISTS {qualified_table(database, str(name))}")
