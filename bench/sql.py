from __future__ import annotations

import re
from typing import Any

from .clickhouse import qualified_table
from .storage import hnsw_table_name, table_names


def build_search_sql(
    config: dict[str, Any],
    method: str,
    filter_sql: str,
    k: int,
    query_index: int,
    *,
    m: int | None = None,
    ef_build: int | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    """Return SQL, bound params, query settings and physical table name."""
    ch, src = config["clickhouse"], config["source"]
    names = table_names(config)
    id_col = src.get("columns", {}).get("id", "mid")
    attr_col = src.get("columns", {}).get("filter_attribute", "avgrating")
    vec_col = src.get("columns", {}).get("vector", "mvector")
    for col in (id_col, attr_col, vec_col):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", col):
            raise ValueError(f"Unsafe SQL column: {col}")

    if method == "HNSW":
        if m is None or ef_build is None:
            raise ValueError("HNSW physical config needs M and ef_construction")
        table_name = hnsw_table_name(config, m, ef_build)
        distance = f"L2Distance(`{vec_col}`, CAST({{q:Array(Float32)}}, 'Array(BFloat16)'))"
        settings = {
            "hnsw_candidate_list_size_for_search": int(config["_active_ef_search"]),
            "vector_search_index_fetch_multiplier": float(config["_active_fetch_multiplier"]),
            "vector_search_filter_strategy": str(config["_active_filter_mode"]),
            "vector_search_with_rescoring": 1 if config.get("hnsw", {}).get("rescoring", False) else 0,
            "max_limit_for_vector_search_queries": int(ch.get("max_limit_for_vector_search_queries", 100000)),
            "max_threads": int(ch.get("max_threads", 4)),
        }
    elif method == "QBit":
        table_name = names["qbit"]
        distance = f"L2DistanceTransposed(`{vec_col}`, CAST({{q:Array(Float32)}}, 'Array(BFloat16)'), {{bits:UInt8}})"
        settings = {
            "allow_experimental_qbit_type": 1,
            "max_threads": int(ch.get("max_threads", 4)),
        }
    elif method == "Exact":
        table_name = names["exact"]
        distance = f"L2Distance(`{vec_col}`, CAST({{q:Array(Float32)}}, 'Array(BFloat16)'))"
        settings = {"max_threads": int(ch.get("max_threads", 4))}
    elif method == "ExactPrefilter":
        if m is None or ef_build is None:
            raise ValueError("ExactPrefilter needs the HNSW table coordinates (M and ef_construction)")
        table_name = hnsw_table_name(config, m, ef_build)
        distance = f"L2Distance(`{vec_col}`, CAST({{q:Array(Float32)}}, 'Array(BFloat16)'))"
        settings = {
            "vector_search_filter_strategy": "prefilter",
            "max_limit_for_vector_search_queries": int(ch.get("max_limit_for_vector_search_queries", 100000)),
            "max_threads": int(ch.get("max_threads", 4)),
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    table = qualified_table(ch["database"], table_name)
    where = f"WHERE ({filter_sql})" if filter_sql else ""
    sql = (
        f"SELECT `{id_col}` AS id, {distance} AS distance "
        f"FROM {table} {where} "
        f"ORDER BY distance ASC LIMIT {{k:UInt32}}"
    )
    if method in {"HNSW", "QBit", "ExactPrefilter"}:
        settings["max_limit_for_vector_search_queries"] = int(ch.get("max_limit_for_vector_search_queries", 100000))
    params: dict[str, Any] = {"k": int(k)}
    return sql, params, settings, table_name
