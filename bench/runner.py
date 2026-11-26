from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clickhouse import (
    drop_vector_similarity_index_cache,
    get_parts_stats,
    qualified_table,
    wait_for_query_log,
)
from .csvio import CsvAppender, load_completed_measurements, read_csv_dicts
from .resources import snapshot_container
from .schema import CONFIG_FIELDS, MEASUREMENT_FIELDS, RESOURCE_FIELDS
from .sql import build_search_sql
from .storage import hnsw_table_name, table_names
from .workload import load_ground_truth, load_prepared_queries

LOG = logging.getLogger(__name__)


def _stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value)


def _experiment_identity(config: dict[str, Any], ch_version: str) -> dict[str, Any]:
    source = config["source"]
    root = Path(config["_project_root"])
    workload_manifest = root / "data" / "prepared" / "workload_manifest.json"
    workload_signature = "unprepared"
    if workload_manifest.exists():
        try:
            manifest = json.loads(workload_manifest.read_text(encoding="utf-8"))
            workload_signature = str(manifest.get("workload_signature", "unavailable"))
        except (OSError, ValueError, TypeError):
            workload_signature = "unavailable"
    source_file = Path(str(source.get("data_file", "")))
    if not source_file.is_absolute():
        source_file = root / source_file
    stat = source_file.stat() if source_file.exists() else None
    return {
        "clickhouse_version": ch_version,
        "dataset": source.get("dataset_size", "unknown"),
        "data_file": str(source_file),
        "data_file_size": stat.st_size if stat else None,
        "data_file_mtime_ns": stat.st_mtime_ns if stat else None,
        "vectors_key": source.get("vectors_key"),
        "dimension": int(source.get("dimension", 0)),
        "stored_vector_type": source.get("stored_vector_type", "BFloat16"),
        "filter_attribute": source.get("columns", {}).get("filter_attribute", "avgrating"),
        "workload_signature": workload_signature,
        "dataset_signature": (
            json.loads(workload_manifest.read_text(encoding="utf-8")).get("dataset_signature")
            if workload_manifest.exists() else "unprepared"
        ),
    }


def _config_record(config: dict[str, Any], method: str, **values: Any) -> dict[str, Any]:
    return {
        "Method": method,
        "CH_version": "",
        "Dataset": f"MoReVec-{config['source'].get('dataset_size', 'unknown')}",
        "Rows_N": "",
        "Dim_D": int(config["source"]["dimension"]),
        "Vector_type": config["source"].get("stored_vector_type", "BFloat16"),
        "Distance": "L2",
        "Filter_mode": values.get("filter_mode", "scan_filter_first"),
        "HNSW_M": values.get("m", ""),
        "HNSW_ef_build": values.get("ef_build", ""),
        "HNSW_ef_search": values.get("ef_search", ""),
        "Fetch_multiplier": values.get("fetch_multiplier", ""),
        "Rescoring": values.get("rescoring", "n/a"),
        "QBit_p": values.get("precision", ""),
        "max_limit": int(config["clickhouse"].get("max_limit_for_vector_search_queries", 100000)),
        "max_threads": int(config["clickhouse"].get("max_threads", 4)),
        "Container_RAM_GB": float(config.get("docker", {}).get("memory_gb", 6)),
        "max_server_memory_GB": float(config.get("docker", {}).get("max_server_memory_gb", 4.5)),
        "Codec": "",
        "Parts": "",
        "Build_time_s": "",
        "Index_size_MB": "",
        "Config_notes": values.get("notes", ""),
    }


def generate_configurations(config: dict[str, Any], ch_version: str) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    identity = _experiment_identity(config, ch_version)
    if config.get("exact", {}).get("enabled", True):
        exact_payload = {**identity, "method": "Exact", "mode": "scan_filter_first"}
        row = _config_record(config, "Exact", filter_mode="scan_filter_first", notes=f"table={table_names(config)['exact']}")
        row["Config_ID"] = _stable_id(exact_payload)
        row["CH_version"] = ch_version
        generated.append({"record": row, "method": "Exact", "table": table_names(config)["exact"], "mode": "scan_filter_first"})

    if config.get("qbit", {}).get("enabled", True):
        for precision in config.get("qbit", {}).get("precision", [8, 12, 16]):
            payload = {**identity, "method": "QBit", "precision": int(precision)}
            row = _config_record(config, "QBit", precision=int(precision), notes=f"table={table_names(config)['qbit']}")
            row["Config_ID"] = _stable_id(payload)
            row["CH_version"] = ch_version
            generated.append({"record": row, "method": "QBit", "precision": int(precision), "table": table_names(config)["qbit"], "mode": "scan_filter_first"})

    hnsw_cfg = config.get("hnsw", {})
    hnsw_physical: list[tuple[int, int]] = []
    if hnsw_cfg.get("enabled", True):
        hnsw_physical = [
            (int(m), int(ef_build))
            for m in hnsw_cfg.get("m", [32])
            for ef_build in hnsw_cfg.get("ef_construction", [128])
        ]
        for m, ef_build in hnsw_physical:
            table_name = hnsw_table_name(config, m, ef_build)
            for ef_search in hnsw_cfg.get("ef_search", [256]):
                for fetch in hnsw_cfg.get("fetch_multiplier", [1.0]):
                    for mode in hnsw_cfg.get("filter_modes", ["postfilter", "auto"]):
                        payload = {
                            **identity,
                            "method": "HNSW", "m": m, "ef_build": ef_build,
                            "ef_search": int(ef_search), "fetch_multiplier": float(fetch),
                            "mode": str(mode), "quantization": hnsw_cfg.get("quantization", "bf16"),
                            "granularity": hnsw_cfg.get("granularity", 100000000),
                            "rescoring": bool(hnsw_cfg.get("rescoring", False)),
                        }
                        row = _config_record(
                            config, "HNSW", filter_mode=str(mode), m=m, ef_build=ef_build,
                            ef_search=int(ef_search), fetch_multiplier=float(fetch),
                            rescoring=hnsw_cfg.get("rescoring", False), notes=f"table={table_name}",
                        )
                        row["Config_ID"] = _stable_id(payload)
                        row["CH_version"] = ch_version
                        generated.append({
                            "record": row, "method": "HNSW", "m": m, "ef_build": ef_build,
                            "ef_search": int(ef_search), "fetch_multiplier": float(fetch),
                            "mode": str(mode), "table": table_name,
                        })

    # ClickHouse's vector-search prefilter strategy performs exact filtered scan,
    # not graph traversal on a filtered subgraph. Keep this as a distinct config.
    if hnsw_physical and hnsw_cfg.get("include_exact_prefilter", True):
        m, ef_build = hnsw_physical[0]
        table_name = hnsw_table_name(config, m, ef_build)
        payload = {**identity, "method": "ExactPrefilter", "mode": "exact_prefilter", "table": table_name}
        row = _config_record(config, "ExactPrefilter", filter_mode="exact_prefilter", notes=f"table={table_name}; vector_search_filter_strategy=prefilter")
        row["Config_ID"] = _stable_id(payload)
        row["CH_version"] = ch_version
        generated.append({"record": row, "method": "ExactPrefilter", "mode": "prefilter", "table": table_name})

    return generated


def _write_configurations(path: Path, configs: list[dict[str, Any]], build_info_path: Path | None = None) -> None:
    existing_rows = read_csv_dicts(path) if path.exists() else []
    existing = {row.get("Config_ID") for row in existing_rows}
    build_by_table: dict[str, dict[str, Any]] = {}
    if build_info_path and build_info_path.exists():
        try:
            payload = json.loads(build_info_path.read_text(encoding="utf-8"))
            for item in payload.get("indexes", []):
                build_by_table[str(item.get("table"))] = item
        except (OSError, ValueError, TypeError):
            LOG.warning("Could not parse build metadata at %s", build_info_path)
    writer = CsvAppender(path, CONFIG_FIELDS)
    for cfg in configs:
        record = cfg["record"]
        table_name = str(cfg.get("table", ""))
        build = build_by_table.get(table_name, {})
        if build:
            record["Parts"] = build.get("parts", "")
            record["Build_time_s"] = build.get("build_seconds", "")
            record["Index_size_MB"] = (
                float(build.get("index_size_bytes")) / (1024**2)
                if build.get("index_size_bytes") not in (None, "") else ""
            )
            record["Table_size_MB"] = (
                float(build.get("size_bytes", 0)) / (1024**2) if build.get("size_bytes") else ""
            )
        if record["Config_ID"] not in existing:
            writer.append(record)
            existing.add(record["Config_ID"])


def _run_one_query(
    client,
    config: dict[str, Any],
    item: dict[str, Any],
    workload_row: dict[str, Any],
    query_vector: np.ndarray,
    k: int,
    excluded_id: str,
    query_id: str,
    *,
    collect_explain: bool = True,
    explain_ref_override: str = "",
) -> tuple[list[str], dict[str, Any], str, str]:
    method = item["method"]
    mode = item.get("mode", "")
    filter_sql = _clean_text(workload_row.get("Filter_SQL"))
    run_config = dict(config)
    run_config["_active_ef_search"] = item.get("ef_search", 256)
    run_config["_active_fetch_multiplier"] = item.get("fetch_multiplier", 1.0)
    run_config["_active_filter_mode"] = mode if method == "HNSW" else "prefilter"

    sql_method = "Exact" if method == "ExactPrefilter" else method
    sql, params, settings, table_name = build_search_sql(
        run_config,
        sql_method,
        filter_sql,
        k,
        int(workload_row["Query_Index"]),
        m=item.get("m"),
        ef_build=item.get("ef_build"),
    )
    # ExactPrefilter deliberately points to the HNSW physical table while forcing
    # ClickHouse's documented prefilter strategy, which executes an exact scan.
    if method == "ExactPrefilter":
        table_name = item["table"]
        database = config["clickhouse"]["database"]
        id_col = config["source"].get("columns", {}).get("id", "mid")
        vec_col = config["source"].get("columns", {}).get("vector", "mvector")
        where = f"WHERE ({filter_sql})" if filter_sql else ""
        sql = (
            f"SELECT `{id_col}` AS id, L2Distance(`{vec_col}`, "
            f"CAST({{q:Array(Float32)}}, 'Array(BFloat16)')) AS distance "
            f"FROM {qualified_table(database, table_name)} {where} "
            f"ORDER BY distance ASC LIMIT {{k:UInt32}}"
        )
        settings = {
            "vector_search_filter_strategy": "prefilter",
            "max_limit_for_vector_search_queries": int(config["clickhouse"].get("max_limit_for_vector_search_queries", 100000)),
            "max_threads": int(config["clickhouse"].get("max_threads", 4)),
        }

    params["q"] = query_vector.astype(np.float32).tolist()
    if method == "QBit":
        params["bits"] = int(item["precision"])
    if excluded_id:
        # Exclude a query vector sampled from the indexed collection.
        id_col = config["source"].get("columns", {}).get("id", "mid")
        if "WHERE" in sql:
            sql = sql.replace("WHERE (", f"WHERE `{id_col}` != {{excluded_id:String}} AND (", 1)
        else:
            sql = sql.replace("ORDER BY distance", f"WHERE `{id_col}` != {{excluded_id:String}} ORDER BY distance")
        params["excluded_id"] = str(excluded_id)

    explain_ref = explain_ref_override
    if collect_explain and config.get("execution", {}).get("collect_explain", True):
        artifact_dir = Path(config["_project_root"]) / "artifacts" / "explain"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        safe_query = str(workload_row["Query_ID"]).replace("/", "_")
        safe_filter = str(workload_row.get("Filter_ID", "filter")).replace("/", "_")
        explain_path = artifact_dir / f"{item['record']['Config_ID']}_{safe_query}_{safe_filter}_{k}.txt"
        try:
            plan = client.query(
                f"EXPLAIN indexes = 1 {sql}",
                parameters=params,
                settings=settings,
            )
            explain_path.write_text("\n".join(str(row[0]) for row in plan.result_rows), encoding="utf-8")
            explain_ref = str(explain_path.relative_to(Path(config["_project_root"])))
        except Exception as exc:
            explain_path.write_text(f"EXPLAIN failed: {type(exc).__name__}: {exc}\n\nSQL:\n{sql}", encoding="utf-8")
            explain_ref = str(explain_path.relative_to(Path(config["_project_root"])))

    result = client.query(
        sql,
        parameters=params,
        settings=settings,
        query_id=query_id,
    )
    ids = [str(row[0]) for row in result.result_rows]
    log_row = wait_for_query_log(
        client,
        query_id,
        timeout_seconds=float(config["clickhouse"].get("query_log_wait_seconds", 15)),
        poll_interval=float(config["clickhouse"].get("query_log_poll_interval", 0.25)),
    )
    return ids, log_row, explain_ref, sql


def _query_workload_path(config: dict[str, Any]) -> Path:
    return Path(config["_project_root"]) / "data" / "prepared" / "query_workload.csv"


def run_benchmark(client, config: dict[str, Any], ch_version: str, *, resume: bool = False) -> None:
    project = Path(config["_project_root"])
    raw_dir = project / config["project"].get("output_dir", "results") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config_csv = raw_dir / "configurations.csv"
    measurement_csv = raw_dir / "measurements.csv"
    resource_csv = raw_dir / "resource_snapshots.csv"

    workload_path = _query_workload_path(config)
    if not workload_path.exists():
        raise FileNotFoundError("Prepared workload not found. Run `prepare` first.")
    workload = pd.read_csv(workload_path).to_dict(orient="records")
    queries, query_ids, excluded_ids = load_prepared_queries(config)
    if not workload:
        raise RuntimeError("Prepared workload is empty")

    all_configs = generate_configurations(config, ch_version)
    exact_table = qualified_table(config["clickhouse"]["database"], table_names(config)["exact"])
    dataset_rows = int(client.query(f"SELECT count() FROM {exact_table}").result_rows[0][0])
    for item in all_configs:
        item["record"]["Rows_N"] = dataset_rows
    _write_configurations(config_csv, all_configs, raw_dir / "build_info.json")
    measurements = CsvAppender(measurement_csv, MEASUREMENT_FIELDS)
    resources = CsvAppender(resource_csv, RESOURCE_FIELDS)
    completed = load_completed_measurements(measurement_csv) if resume else set()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    rng = random.Random(int(config["project"].get("seed", 42)))
    repeats = int(config["execution"].get("repeats", 1))
    k_values = [int(x) for x in config["workload"]["k_values"]]
    max_cases = config["execution"].get("max_cases")
    smoke_mode = bool(config["execution"].get("smoke_mode", False))
    smoke_pairs = int(config["execution"].get("smoke_pairs_per_config", 2))
    executed = 0

    for item in all_configs:
        record = item["record"]
        config_id = record["Config_ID"]
        method = item["method"]
        LOG.info("Configuration %s: %s (%s)", config_id, method, item.get("mode", ""))
        if config["execution"].get("clear_vector_index_cache_before_config", True):
            try:
                drop_vector_similarity_index_cache(client)
            except Exception as exc:
                raise RuntimeError(
                    "Could not clear ClickHouse vector similarity index cache. "
                    "Set execution.clear_vector_index_cache_before_config=false only "
                    "if you intentionally accept cross-configuration cache carry-over."
                ) from exc
        resources.append(snapshot_container(config, config_id, "before_config", client))
        workload_order = list(workload)
        if smoke_mode:
            filtered_rows = [row for row in workload_order if _clean_text(row.get("Filter_SQL"))]
            unfiltered_rows = [row for row in workload_order if not _clean_text(row.get("Filter_SQL"))]
            selected = []
            if filtered_rows:
                selected.append(filtered_rows[0])
            if unfiltered_rows and len(selected) < smoke_pairs:
                selected.append(unfiltered_rows[0])
            if len(selected) < smoke_pairs:
                for row in workload_order:
                    if row not in selected:
                        selected.append(row)
                    if len(selected) >= smoke_pairs:
                        break
            workload_order = selected[:smoke_pairs]
        config_explain_refs: dict[tuple[str, int], str] = {}
        if config["execution"].get("randomize_order", True) and not smoke_mode:
            rng.shuffle(workload_order)

        # Prefer an unfiltered warm-up so the vector column/index is touched consistently.
        warmup_count = int(config["execution"].get("warmup_queries_per_config", 0))
        warmup_rows = sorted(
            workload_order,
            key=lambda row: 0 if not _clean_text(row.get("Filter_SQL")) else 1,
        )[:warmup_count]
        for warm_row in warmup_rows:
            qidx = int(warm_row["Query_Index"])
            for k in k_values[:1]:
                try:
                    _run_one_query(
                        client, config, item, warm_row, queries[qidx], min(k, int(warm_row["Eligible_Rows"])),
                        str(warm_row.get("Excluded_ID", "")), uuid.uuid4().hex,
                        collect_explain=False,
                    )
                except Exception as exc:
                    LOG.warning("Warmup failed for %s: %s", config_id, exc)

        for row in workload_order:
            if item["method"] == "ExactPrefilter" and not _clean_text(row.get("Filter_SQL")):
                continue
            qidx = int(row["Query_Index"])
            query_base_id = str(row["Query_ID"])
            filter_id = str(row["Filter_ID"])
            excluded = _clean_text(row.get("Excluded_ID"))
            for k in k_values:
                k_eff = min(k, int(row["Eligible_Rows"]))
                if k_eff == 0:
                    continue
                key_query_id = query_base_id
                for repeat in range(1, repeats + 1):
                    key = (config_id, key_query_id, filter_id, int(k), repeat)
                    if key in completed:
                        continue
                    query_uid = uuid.uuid4().hex
                    try:
                        result_ids, qlog, explain_ref, _sql = _run_one_query(
                            client, config, item, row, queries[qidx], k_eff, excluded, query_uid,
                            collect_explain=((filter_id, int(k)) not in config_explain_refs),
                            explain_ref_override=config_explain_refs.get((filter_id, int(k)), ""),
                        )
                        if explain_ref and (filter_id, int(k)) not in config_explain_refs:
                            config_explain_refs[(filter_id, int(k))] = explain_ref
                        gt = load_ground_truth(config, qidx, filter_id)
                        gt_ids = set(str(x) for x in gt[:k_eff])
                        tp = len(set(result_ids) & gt_ids)
                        gt_count = len(gt_ids)
                        recall = tp / gt_count if gt_count else ""
                        fill = len(result_ids) / k_eff if k_eff else ""
                        latency = qlog.get("query_duration_ms", "")
                        read_rows = qlog.get("read_rows", "")
                        read_bytes = qlog.get("read_bytes", "")
                        memory_mb = (
                            float(qlog["memory_usage"]) / (1024**2)
                            if qlog.get("memory_usage") not in (None, "") else ""
                        )
                        note = "" if qlog else "query_log_timeout"
                        measurements.append({
                            "Config_ID": config_id,
                            "Run_ID": run_id,
                            "Query_ID": key_query_id,
                            "Filter_ID": filter_id,
                            "Repeat": repeat,
                            "Cache_state": config["execution"].get("cache_state", "warm"),
                            "Query_type": "topk",
                            "Filter_type": row.get("Filter_type", ""),
                            "Sigma_target": row.get("Sigma_target", ""),
                            "Sigma_actual": row.get("Sigma_actual", ""),
                            "GLS_target": "",  # Natural GLS is measured, not assigned as an experimental target.
                            "GLS_actual": row.get("GLS", ""),
                            "GLS_Group": row.get("GLS_Group", ""),
                            "GLS_kNN": row.get("Local_K", ""),
                            "k_requested": k,
                            "eligible_rows": row.get("Eligible_Rows", ""),
                            "k_effective": k_eff,
                            "GT_count": gt_count,
                            "TP": tp,
                            "returned_rows": len(result_ids),
                            "Recall_at_k": recall,
                            "Fill_rate": fill,
                            "Latency_ms": latency,
                            "read_rows": read_rows,
                            "read_bytes": read_bytes,
                            "peak_mem_MB": memory_mb,
                            "EXPLAIN_ref": explain_ref,
                            "Run_notes": note,
                        })
                        completed.add(key)
                        executed += 1
                        if executed % 25 == 0:
                            LOG.info("Saved %d measurements", executed)
                    except Exception as exc:
                        LOG.exception("Query failed: config=%s query=%s filter=%s k=%s", config_id, key_query_id, filter_id, k)
                        measurements.append({
                            "Config_ID": config_id, "Run_ID": run_id, "Query_ID": key_query_id,
                            "Filter_ID": filter_id, "Repeat": repeat,
                            "Cache_state": config["execution"].get("cache_state", "warm"),
                            "Query_type": "topk", "Filter_type": row.get("Filter_type", ""),
                            "Sigma_target": row.get("Sigma_target", ""), "Sigma_actual": row.get("Sigma_actual", ""),
                            "GLS_target": "", "GLS_actual": row.get("GLS", ""),
                            "GLS_Group": row.get("GLS_Group", ""),
                            "GLS_kNN": row.get("Local_K", ""), "k_requested": k,
                            "eligible_rows": row.get("Eligible_Rows", ""), "k_effective": k_eff,
                            "Run_notes": f"ERROR: {type(exc).__name__}: {exc}",
                        })
                        if config["execution"].get("stop_on_error", False):
                            raise
                if max_cases is not None and executed >= int(max_cases):
                    LOG.warning("Reached execution.max_cases=%s", max_cases)
                    return
        resources.append(snapshot_container(config, config_id, "after_config", client))

    LOG.info("Benchmark completed; new measurements saved: %d", executed)
