from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .hdf5 import read_hdf5_array


@dataclass
class MovieDataset:
    ids: np.ndarray
    vectors: np.ndarray
    attributes: dict[str, np.ndarray]
    queries: np.ndarray
    query_ids: list[str]
    excluded_ids: list[str]
    dimension: int
    query_source: str

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def filter_values(self) -> np.ndarray:
        return np.asarray(self.attributes["avgrating"], dtype=np.float32)


def _decode_strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([
            item.decode("utf-8", errors="replace") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in values
        ], dtype=object)
    return values


def _read_spec(config: dict[str, Any], base_file: Path, spec: Any) -> np.ndarray | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        key = spec
        file_path = base_file
    elif isinstance(spec, dict):
        key = spec.get("key")
        file_path = resolve_path(config, spec.get("file")) or base_file
    else:
        raise TypeError(f"Unsupported HDF5 key mapping: {spec!r}")
    if not key:
        return None
    return read_hdf5_array(file_path, str(key))


def load_movie_dataset(config: dict[str, Any]) -> MovieDataset:
    """Load original MoReVec HDF5 arrays without modifying the source files.

    The HDF5 key mapping is deliberately explicit because the public MoReVec package
    separates vector data, filters and query workloads into different files.
    Run `inspect-data` before configuring keys for a local copy.
    """
    source = config["source"]
    data_file = resolve_path(config, source.get("data_file"))
    if data_file is None or not data_file.exists():
        raise FileNotFoundError(
            "MoReVec vector HDF5 file not found. Download MoReVec and set source.data_file."
        )

    vectors_key = source.get("vectors_key")
    if not vectors_key:
        raise ValueError("Set source.vectors_key after running `inspect-data`.")
    vectors = np.asarray(read_hdf5_array(data_file, str(vectors_key)), dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors array, got shape={vectors.shape}")

    expected_dim = int(source.get("dimension", vectors.shape[1]))
    if vectors.shape[1] != expected_dim:
        raise ValueError(
            f"Vector dimension mismatch: HDF5 has {vectors.shape[1]}, config expects {expected_dim}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Vector matrix contains NaN or infinity values")

    ids_spec = source.get("ids_key")
    ids = _read_spec(config, data_file, ids_spec)
    if ids is None:
        ids = np.arange(vectors.shape[0], dtype=np.int64).astype(str)
    else:
        ids = _decode_strings(ids.reshape(-1))
    if len(ids) != len(vectors):
        raise ValueError(f"ID count {len(ids)} differs from vector count {len(vectors)}")

    attributes: dict[str, np.ndarray] = {}
    for name, spec in source.get("attributes", {}).items():
        array = _read_spec(config, data_file, spec)
        if array is not None:
            array = _decode_strings(array.reshape(-1))
            if len(array) != len(vectors):
                raise ValueError(
                    f"Attribute '{name}' has {len(array)} rows; expected {len(vectors)}"
                )
            attributes[name] = array

    filter_attribute = source.get("columns", {}).get("filter_attribute", "avgrating")
    if filter_attribute not in attributes:
        raise ValueError(
            f"Required filter attribute '{filter_attribute}' was not loaded. "
            "Run `inspect-data` for the vector and filter HDF5 files, then set "
            "source.attributes.<attribute>.file and .key in the YAML config."
        )
    attributes[filter_attribute] = np.asarray(attributes[filter_attribute], dtype=np.float32)
    if not np.isfinite(attributes[filter_attribute]).all():
        raise ValueError(f"Filter attribute '{filter_attribute}' contains NaN or infinity")

    # Optional fields used in later relational/analytical extensions.
    if "genre" in attributes:
        attributes["genre"] = _decode_strings(attributes["genre"])
    for name in ("year", "numvotes"):
        if name in attributes:
            attributes[name] = np.asarray(attributes[name])

    query_cfg = source.get("queries", {})
    query_file = resolve_path(config, query_cfg.get("file"))
    query_key = query_cfg.get("vectors_key")
    requested_queries = int(query_cfg.get("count", config["workload"].get("query_count", 100)))
    seed = int(config["project"].get("seed", 42))
    rng = np.random.default_rng(seed)

    if query_file and query_file.exists() and query_key:
        queries = np.asarray(read_hdf5_array(query_file, str(query_key)), dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != expected_dim:
            raise ValueError(f"Query vectors have invalid shape: {queries.shape}")
        query_id_spec = query_cfg.get("ids_key")
        query_ids_array = _read_spec(config, query_file, query_id_spec)
        if query_ids_array is None:
            query_ids = [f"external_{i:06d}" for i in range(len(queries))]
        else:
            query_ids = [str(item) for item in _decode_strings(query_ids_array.reshape(-1))]
        chosen = rng.choice(len(queries), size=min(requested_queries, len(queries)), replace=False)
        queries = queries[chosen]
        query_ids = [query_ids[int(i)] for i in chosen]
        excluded_ids = [""] * len(queries)
        query_source = str(query_file)
    else:
        count = min(requested_queries, len(vectors))
        chosen = rng.choice(len(vectors), size=count, replace=False)
        queries = vectors[chosen].copy()
        query_ids = [f"q_{i:06d}" for i in range(count)]
        excluded_ids = [str(ids[int(i)]) for i in chosen]
        query_source = "sampled_from_base_with_self_exclusion"

    if not np.isfinite(queries).all():
        raise ValueError("Query vectors contain NaN or infinity values")

    if source.get("normalized_vectors", False):
        norms = np.linalg.norm(vectors[: min(len(vectors), 10000)], axis=1)
        query_norms = np.linalg.norm(queries[: min(len(queries), 1000)], axis=1)
        if not np.allclose(norms, 1.0, atol=2e-2) or not np.allclose(query_norms, 1.0, atol=2e-2):
            raise ValueError(
                "Config expects L2-normalized vectors, but sampled norms differ from 1. "
                "Set source.normalized_vectors=false only after verifying the dataset semantics."
            )

    return MovieDataset(
        ids=np.asarray(ids),
        vectors=vectors,
        attributes=attributes,
        queries=queries,
        query_ids=query_ids,
        excluded_ids=excluded_ids,
        dimension=expected_dim,
        query_source=query_source,
    )
