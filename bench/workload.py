from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import resolve_path
from .dataset import MovieDataset, load_movie_dataset
from .math_utils import (
    float32_to_bfloat16_inplace,
    float32_to_bfloat16_roundtrip,
    gls_correlation,
    gls_group,
    squared_l2_from_dot,
    topk_indices,
)
from .schema import WORKLOAD_FIELDS

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterSpec:
    filter_id: str
    filter_sql: str
    filter_type: str
    sigma_target: float | None
    sigma_actual: float
    threshold: float | None


def _safe_sql_identifier(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return identifier


def _build_filters(dataset: MovieDataset, config: dict[str, Any]) -> list[FilterSpec]:
    workload = config["workload"]
    source = config["source"]
    attribute = _safe_sql_identifier(source.get("columns", {}).get("filter_attribute", "avgrating"))
    values = np.asarray(dataset.attributes[attribute], dtype=np.float64)
    n = len(values)
    filters: list[FilterSpec] = []
    seen_masks: set[bytes] = set()

    if workload.get("include_unfiltered", True):
        filters.append(FilterSpec(
            filter_id="unfiltered",
            filter_sql="",
            filter_type="none",
            sigma_target=1.0,
            sigma_actual=1.0,
            threshold=None,
        ))
        seen_masks.add(b"unfiltered")

    for target in workload.get("sigma_targets", []):
        target = float(target)
        wanted = max(1, min(n, int(round(target * n))))
        sorted_values = np.sort(values)
        threshold = float(sorted_values[n - wanted])
        mask = values >= threshold
        mask_hash = hashlib.blake2b(np.packbits(mask).tobytes(), digest_size=16).digest()
        if mask_hash in seen_masks:
            LOG.warning(
                "Skipping duplicate filter mask for target sigma=%s (ties in %s)", target, attribute
            )
            continue
        seen_masks.add(mask_hash)
        actual = float(mask.mean())
        filter_id = f"{attribute}_ge_{threshold:.6f}_t{target:.4f}".replace(".", "p")
        filters.append(FilterSpec(
            filter_id=filter_id,
            filter_sql=f"{attribute} >= {threshold:.9g}",
            filter_type="numeric_range",
            sigma_target=target,
            sigma_actual=actual,
            threshold=threshold,
        ))

    if not filters:
        raise ValueError("No filters generated. Check workload.include_unfiltered and sigma_targets.")
    return filters


def _group_for_rho(rho: float, config: dict[str, Any]) -> str:
    low, high = map(float, config["workload"].get("gls_group_thresholds", [-0.3, 0.3]))
    return gls_group(rho, lower=low, upper=high)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def prepare_workload(config: dict[str, Any], output_dir: str | Path | None = None) -> MovieDataset:
    """Prepare deterministic query/filter workload, GLS, and exact ground truth.

    Source MoReVec HDF5 files are read-only. The only transformed copy used by the
    ClickHouse benchmark is rounded to BFloat16 to match the chosen stored-vector type.
    """
    LOG.info("Loading MoReVec dataset")
    dataset = load_movie_dataset(config)
    project_root = Path(config["_project_root"])
    out_root = Path(output_dir) if output_dir else project_root / "data" / "prepared"
    out_root.mkdir(parents=True, exist_ok=True)
    gt_dir = out_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Use the same BFloat16-rounded values for ground truth, HNSW, QBit and exact baseline.
    # Round in place to avoid a second 1.7-GB matrix for Movies Large.
    LOG.info("Preparing BFloat16 vectors, queries and filters")
    vectors = float32_to_bfloat16_inplace(dataset.vectors)
    dataset.vectors = vectors
    queries = float32_to_bfloat16_roundtrip(dataset.queries)
    matrix_norm_sq = np.einsum("ij,ij->i", vectors, vectors, optimize=True)
    id_to_row = {str(value): index for index, value in enumerate(dataset.ids)}

    filters = _build_filters(dataset, config)
    local_k = int(config["workload"].get("local_neighbors", 2048))
    sensitivity_ks = [int(x) for x in config["workload"].get("local_neighbors_sensitivity", [])]
    local_ks = sorted(set([local_k] + sensitivity_ks))
    requested_max_gt = int(config["workload"].get("max_ground_truth_k", max(config["workload"]["k_values"])))

    query_npz = out_root / "queries.npz"
    np.savez_compressed(
        query_npz,
        vectors=queries.astype(np.float32),
        query_ids=np.asarray(dataset.query_ids, dtype="U64"),
        excluded_ids=np.asarray(dataset.excluded_ids, dtype="U128"),
    )

    filters_json = []
    for spec in filters:
        item = asdict(spec)
        if spec.threshold is not None:
            item["threshold"] = float(spec.threshold)
        filters_json.append(item)
    _save_json(out_root / "filters.json", filters_json)

    rows: list[dict[str, Any]] = []
    query_meta: list[dict[str, Any]] = []
    query_inputs = zip(queries, dataset.query_ids, dataset.excluded_ids, strict=True)
    progress = tqdm(query_inputs, total=len(queries), desc="Preparing ground truth and GLS", unit="query")
    for query_index, (query, query_id, excluded_id) in enumerate(progress):
        excluded_index = id_to_row.get(str(excluded_id)) if excluded_id else None
        distances = squared_l2_from_dot(vectors, query, matrix_norm_sq)
        if excluded_index is not None:
            distances[excluded_index] = np.inf

        # Compute local-neighbor masks once per query and re-use them across all filters.
        local_orders: dict[int, np.ndarray] = {}
        for lk in local_ks:
            effective_local_k = min(lk, int(np.isfinite(distances).sum()))
            local_orders[lk] = topk_indices(distances, effective_local_k)

        query_meta.append({
            "Query_ID": str(query_id),
            "Query_Index": query_index,
            "Excluded_ID": str(excluded_id or ""),
            "Query_Source": dataset.query_source,
        })

        for spec in filters:
            if spec.filter_sql:
                assert spec.threshold is not None
                valid = np.asarray(dataset.attributes[
                    config["source"].get("columns", {}).get("filter_attribute", "avgrating")
                ], dtype=np.float64) >= spec.threshold
            else:
                valid = np.ones(dataset.size, dtype=bool)

            if excluded_index is not None:
                valid[excluded_index] = False
            eligible_rows = int(valid.sum())
            available_rows = dataset.size - (1 if excluded_index is not None else 0)
            sigma_query = eligible_rows / max(available_rows, 1)
            if eligible_rows == 0:
                LOG.warning("Query %s/filter %s has no eligible rows; skipping pair", query_id, spec.filter_id)
                continue

            local_order = local_orders[local_k]
            local_selectivity = float(valid[local_order].mean()) if len(local_order) else 0.0
            rho = 0.0 if not spec.filter_sql else gls_correlation(sigma_query, local_selectivity)
            group = "unfiltered" if not spec.filter_sql else _group_for_rho(rho, config)

            # Store sensitivity GLS for larger fixed neighborhoods without changing the primary metric.
            rho_sensitivity: dict[str, float] = {}
            for lk in sensitivity_ks:
                lk_eff = min(lk, len(local_orders[lk]))
                sig_l = float(valid[local_orders[lk]].mean()) if lk_eff else 0.0
                rho_sensitivity[str(lk)] = 0.0 if not spec.filter_sql else gls_correlation(sigma_query, sig_l)

            # Filter exact distances and keep only the largest requested ground-truth prefix.
            valid_indices = np.flatnonzero(valid)
            filtered_distances = distances[valid_indices]
            k_eff = min(requested_max_gt, len(valid_indices))
            if k_eff == len(valid_indices):
                top_local = np.argsort(filtered_distances, kind="stable")
            else:
                candidate = np.argpartition(filtered_distances, k_eff - 1)[:k_eff]
                top_local = candidate[np.argsort(filtered_distances[candidate], kind="stable")]
            top_global = valid_indices[top_local]
            gt_ids = np.asarray(dataset.ids[top_global]).astype(str)

            gt_name = f"{query_index:06d}__{spec.filter_id}.npy"
            np.save(gt_dir / gt_name, gt_ids, allow_pickle=False)

            rows.append({
                "Query_ID": str(query_id),
                "Filter_ID": spec.filter_id,
                "Filter_SQL": spec.filter_sql,
                "Filter_type": spec.filter_type,
                "Sigma_target": spec.sigma_target,
                "Sigma_actual": sigma_query,
                "Local_K": local_k,
                "Local_Selectivity": local_selectivity,
                "GLS": rho,
                "GLS_Group": group,
                "Eligible_Rows": eligible_rows,
                "Excluded_ID": str(excluded_id or ""),
                "Query_Index": query_index,
                "Ground_Truth_File": gt_name,
                "GLS_Sensitivity": json.dumps(rho_sensitivity, sort_keys=True),
            })

    LOG.info("Writing workload summaries and computing dataset signatures")
    workload_df = pd.DataFrame(rows)
    if workload_df.empty:
        raise RuntimeError("No query/filter pairs produced. Check dataset mapping and filter attribute.")
    workload_df.to_csv(out_root / "query_workload.csv", index=False)
    pd.DataFrame(query_meta).to_csv(out_root / "query_manifest.csv", index=False)

    coverage = (
        workload_df.groupby(["Sigma_target", "GLS_Group"], dropna=False)
        .agg(
            query_filter_pairs=("Query_ID", "count"),
            unique_queries=("Query_ID", "nunique"),
            sigma_mean=("Sigma_actual", "mean"),
            gls_mean=("GLS", "mean"),
            gls_min=("GLS", "min"),
            gls_max=("GLS", "max"),
        )
        .reset_index()
    )
    coverage.to_csv(out_root / "gls_coverage.csv", index=False)

    filter_attribute = config["source"].get("columns", {}).get("filter_attribute", "avgrating")
    dataset_signature_payload = {
        "vector_sha256": _array_sha256(vectors),
        "attribute_sha256": _array_sha256(np.asarray(dataset.attributes[filter_attribute])),
        "ids_sha256": _array_sha256(np.asarray(dataset.ids).astype(str)),
        "dimension": dataset.dimension,
        "rows": dataset.size,
        "stored_vector_type": config["source"].get("stored_vector_type", "BFloat16"),
    }
    dataset_signature = hashlib.sha256(
        json.dumps(dataset_signature_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    signature_payload = {
        **dataset_signature_payload,
        "dataset_signature": dataset_signature,
        "query_sha256": _array_sha256(queries),
        "query_ids": dataset.query_ids,
        "excluded_ids": dataset.excluded_ids,
        "filters": [asdict(spec) for spec in filters],
        "seed": int(config["project"].get("seed", 42)),
        "local_neighbors": local_k,
        "sensitivity_local_neighbors": sensitivity_ks,
        "ground_truth_k": requested_max_gt,
    }
    workload_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()

    metadata = {
        "dataset": source_name(config),
        "dataset_signature": dataset_signature,
        "workload_signature": workload_signature,
        "rows": dataset.size,
        "dimension": dataset.dimension,
        "query_count": len(dataset.queries),
        "query_source": dataset.query_source,
        "local_neighbors": local_k,
        "sensitivity_local_neighbors": sensitivity_ks,
        "filters": filters_json,
        "ground_truth_k": requested_max_gt,
        "vector_representation_for_ground_truth": "BFloat16-rounded values represented as float32",
        "source_files_read_only": True,
    }
    _save_json(out_root / "workload_manifest.json", metadata)
    LOG.info("Prepared %s query/filter pairs under %s", len(workload_df), out_root)
    LOG.info("GLS coverage written to %s", out_root / "gls_coverage.csv")
    return dataset


def source_name(config: dict[str, Any]) -> str:
    return f"MoReVec-{config['source'].get('dataset_size', 'unknown')}"


def load_prepared_queries(config: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    project_root = Path(config["_project_root"])
    path = project_root / "data" / "prepared" / "queries.npz"
    if not path.exists():
        raise FileNotFoundError("Prepared query cache not found. Run `python benchmark.py prepare` first.")
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["vectors"], dtype=np.float32),
            [str(x) for x in payload["query_ids"].tolist()],
            [str(x) for x in payload["excluded_ids"].tolist()],
        )


def load_ground_truth(config: dict[str, Any], query_index: int, filter_id: str) -> np.ndarray:
    project_root = Path(config["_project_root"])
    path = project_root / "data" / "prepared" / "ground_truth" / f"{query_index:06d}__{filter_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth missing: {path}. Re-run `prepare`.")
    return np.load(path, allow_pickle=False).astype(str)
