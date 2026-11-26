import json

from bench.runner import _clean_text, _experiment_identity, _stable_id, generate_configurations
from bench.storage import table_names


def test_clean_text_treats_pandas_nan_as_empty():
    import numpy as np

    assert _clean_text(np.nan) == ""
    assert _clean_text(None) == ""
    assert _clean_text("avgrating >= 4") == "avgrating >= 4"


def test_config_identity_changes_with_clickhouse_version_and_workload(tmp_path):
    root = tmp_path
    (root / "data/prepared").mkdir(parents=True)
    (root / "data/prepared/workload_manifest.json").write_text(
        json.dumps({"workload_signature": "sig-a"}), encoding="utf-8"
    )
    data_file = root / "movies.hdf5"
    data_file.write_bytes(b"dataset")
    config = {
        "_project_root": str(root),
        "source": {
            "dataset_size": "large", "data_file": "movies.hdf5",
            "vectors_key": "train_mvector", "dimension": 768,
            "stored_vector_type": "BFloat16",
            "columns": {"filter_attribute": "avgrating"},
        },
    }
    first = _experiment_identity(config, "26.8.44.3")
    assert first["workload_signature"] == "sig-a"
    assert _stable_id(first) != _stable_id(_experiment_identity(config, "26.8.44.4"))

    (root / "data/prepared/workload_manifest.json").write_text(
        json.dumps({"workload_signature": "sig-b"}), encoding="utf-8"
    )
    assert _stable_id(first) != _stable_id(_experiment_identity(config, "26.8.44.3"))


def test_generate_configurations_marks_exact_prefilter_separately(tmp_path):
    data_file = tmp_path / "not-present.hdf5"
    config = {
        "_project_root": str(tmp_path),
        "source": {"dataset_size": "large", "data_file": str(data_file), "dimension": 768,
                   "stored_vector_type": "BFloat16", "columns": {"filter_attribute": "avgrating"}},
        "clickhouse": {"max_threads": 4, "max_limit_for_vector_search_queries": 100000},
        "docker": {"memory_gb": 6, "max_server_memory_gb": 4.5},
        "exact": {"enabled": True},
        "qbit": {"enabled": False},
        "hnsw": {"enabled": True, "m": [32], "ef_construction": [128], "ef_search": [128],
                 "fetch_multiplier": [1.0], "filter_modes": ["postfilter"],
                 "include_exact_prefilter": True},
    }
    result = generate_configurations(config, "26.8.44.3")
    methods = [item["record"]["Method"] for item in result]
    assert methods == ["Exact", "HNSW", "ExactPrefilter"]
    assert result[-1]["method"] == "ExactPrefilter"


def test_table_names_are_fingerprinted_by_prepared_dataset(tmp_path):
    prepared = tmp_path / "data/prepared"
    prepared.mkdir(parents=True)
    config = {"_project_root": str(tmp_path), "source": {"table_name": "movies"}}
    (prepared / "workload_manifest.json").write_text(
        json.dumps({"dataset_signature": "abcdef0123456789"}), encoding="utf-8"
    )
    first = table_names(config)
    assert first["exact"] == "movies_abcdef0123_exact"
    (prepared / "workload_manifest.json").write_text(
        json.dumps({"dataset_signature": "1234567890abcdef"}), encoding="utf-8"
    )
    second = table_names(config)
    assert second["exact"] == "movies_1234567890_exact"
    assert first["exact"] != second["exact"]
