import numpy as np

from bench.math_utils import (
    float32_to_bfloat16_inplace,
    float32_to_bfloat16_roundtrip,
    gls_correlation,
    gls_group,
    topk_indices,
)


def test_bfloat16_roundtrip_reduces_precision_without_changing_shape():
    values = np.array([[1.0, 0.1, -2.25]], dtype=np.float32)
    rounded = float32_to_bfloat16_roundtrip(values)
    assert rounded.shape == values.shape
    assert rounded.dtype == np.float32
    assert np.allclose(rounded, values, atol=0.01)
    assert not np.shares_memory(rounded, values)


def test_gls_identity_and_extremes():
    assert gls_correlation(0.1, 0.1) == 0.0
    assert gls_correlation(0.1, 0.0) == -1.0
    assert gls_correlation(0.1, 0.5) > 0.0
    assert gls_group(-0.5) == "low"
    assert gls_group(0.0) == "medium"
    assert gls_group(0.5) == "high"


def test_topk_indices_returns_sorted_smallest_values():
    distances = np.array([5.0, 1.0, 3.0, 2.0], dtype=np.float32)
    assert topk_indices(distances, 2).tolist() == [1, 3]
    assert topk_indices(distances, 4).tolist() == [1, 3, 2, 0]


def test_inplace_bfloat16_roundtrip_matches_copy_and_mutates_only_working_array():
    source = np.array([[1.123456, -0.333333], [0.0, 12.34567]], dtype=np.float32)
    expected = float32_to_bfloat16_roundtrip(source)
    working = source.copy()
    returned = float32_to_bfloat16_inplace(working, chunk_rows=1)
    assert returned is working
    assert np.array_equal(working, expected)
    assert not np.array_equal(source, working)
