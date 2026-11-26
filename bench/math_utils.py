from __future__ import annotations

import numpy as np


def float32_to_bfloat16_roundtrip(values: np.ndarray) -> np.ndarray:
    """Return a copy rounded to IEEE bfloat16 precision, represented as float32."""
    array = np.asarray(values, dtype=np.float32).copy()
    return float32_to_bfloat16_inplace(array)


def float32_to_bfloat16_inplace(values: np.ndarray, chunk_rows: int = 8192) -> np.ndarray:
    """Round a writable float32 array to bfloat16 precision without duplicating the matrix."""
    array = np.asarray(values, dtype=np.float32)
    if not array.flags.writeable:
        raise ValueError("In-place bfloat16 conversion requires a writable array")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    flat = array.reshape(-1)
    bits = flat.view(np.uint32)
    for start in range(0, len(bits), int(chunk_rows) * (array.shape[1] if array.ndim > 1 else 1)):
        end = min(start + int(chunk_rows) * (array.shape[1] if array.ndim > 1 else 1), len(bits))
        view = bits[start:end]
        bias = np.uint32(0x7FFF) + ((view >> np.uint32(16)) & np.uint32(1))
        view[:] = (view + bias) & np.uint32(0xFFFF0000)
    return array


def squared_l2_from_dot(
    matrix: np.ndarray,
    query: np.ndarray,
    matrix_norm_sq: np.ndarray | None = None,
) -> np.ndarray:
    """Compute squared L2 distances using a matrix-vector product."""
    query = np.asarray(query, dtype=np.float32)
    dot = matrix @ query
    if matrix_norm_sq is None:
        matrix_norm_sq = np.einsum("ij,ij->i", matrix, matrix, optimize=True)
    query_norm_sq = float(np.dot(query, query))
    distances = matrix_norm_sq + query_norm_sq - (2.0 * dot)
    # Round-off can produce tiny negative values for identical/near-identical vectors.
    return np.maximum(distances, 0.0)


def topk_indices(distances: np.ndarray, k: int, excluded_index: int | None = None) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float32).copy()
    if excluded_index is not None and 0 <= excluded_index < len(values):
        values[excluded_index] = np.inf
    count = min(max(int(k), 0), int(np.isfinite(values).sum()))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == len(values):
        candidates = np.flatnonzero(np.isfinite(values))
    else:
        candidates = np.argpartition(values, count - 1)[:count]
        candidates = candidates[np.isfinite(values[candidates])]
    order = np.argsort(values[candidates], kind="stable")
    return candidates[order].astype(np.int64)


def gls_correlation(global_selectivity: float, local_selectivity: float) -> float:
    """MoReVec per-query GLS: rho=(r-1)/(r+1), r=sigma_local/sigma_global."""
    if not 0.0 < global_selectivity <= 1.0:
        raise ValueError("global_selectivity must be in (0, 1]")
    ratio = local_selectivity / global_selectivity
    return float((ratio - 1.0) / (ratio + 1.0))


def gls_group(rho: float, lower: float = -0.3, upper: float = 0.3) -> str:
    if rho < lower:
        return "low"
    if rho > upper:
        return "high"
    return "medium"
