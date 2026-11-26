from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py


def inspect_hdf5(path: str | Path) -> list[dict[str, Any]]:
    """Return every dataset path, shape and dtype in an HDF5 file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    records: list[dict[str, Any]] = []
    with h5py.File(file_path, "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                records.append({
                    "key": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "size": int(obj.size),
                })
        handle.visititems(visitor)
    return records


def read_hdf5_array(path: str | Path, key: str):
    """Read one HDF5 dataset into memory with a useful key error."""
    import numpy as np

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if not key:
        raise ValueError(f"No HDF5 key configured for {file_path}")
    with h5py.File(file_path, "r") as handle:
        if key not in handle:
            available: list[str] = []
            handle.visititems(
                lambda name, obj: available.append(name) if isinstance(obj, h5py.Dataset) else None
            )
            preview = ", ".join(available[:30])
            raise KeyError(
                f"HDF5 key '{key}' not found in {file_path}. Available datasets: {preview}"
            )
        return np.asarray(handle[key])
