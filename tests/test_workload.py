import numpy as np
import h5py

from bench.workload import prepare_workload


def test_prepare_workload_leaves_hdf5_untouched_and_writes_gls(tmp_path):
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(80, 12)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ratings = np.linspace(1.0, 5.0, len(vectors), dtype=np.float32)
    data_file = tmp_path / "movies.hdf5"
    with h5py.File(data_file, "w") as h5:
        h5.create_dataset("train_mvector", data=vectors)
        h5.create_dataset("train_avgrating", data=ratings)

    config = {
        "_project_root": str(tmp_path),
        "project": {"seed": 7},
        "source": {
            "dataset_size": "test",
            "data_file": str(data_file),
            "vectors_key": "train_mvector",
            "ids_key": None,
            "attributes": {"avgrating": {"key": "train_avgrating", "file": None}},
            "columns": {"id": "mid", "vector": "mvector", "filter_attribute": "avgrating"},
            "dimension": 12,
            "normalized_vectors": True,
            "queries": {"file": None, "count": 5},
        },
        "workload": {
            "local_neighbors": 8,
            "local_neighbors_sensitivity": [16],
            "sigma_targets": [0.1, 0.5],
            "include_unfiltered": True,
            "k_values": [2],
            "query_count": 5,
            "max_ground_truth_k": 4,
            "gls_group_thresholds": [-0.3, 0.3],
        },
    }
    prepare_workload(config)
    with h5py.File(data_file, "r") as h5:
        assert np.array_equal(h5["train_mvector"][:], vectors)
        assert np.array_equal(h5["train_avgrating"][:], ratings)
    assert (tmp_path / "data/prepared/query_workload.csv").exists()
    assert (tmp_path / "data/prepared/gls_coverage.csv").exists()
    assert len(list((tmp_path / "data/prepared/ground_truth").glob("*.npy"))) > 0
