from bench.sql import build_search_sql


def _config(tmp_path):
    return {
        "_project_root": str(tmp_path),
        "source": {
            "table_name": "movies",
            "dimension": 768,
            "columns": {"id": "mid", "vector": "mvector", "filter_attribute": "avgrating"},
        },
        "clickhouse": {
            "database": "vector_benchmark",
            "max_threads": 4,
            "max_limit_for_vector_search_queries": 100000,
        },
        "hnsw": {"rescoring": False},
    }


def test_hnsw_sql_carries_search_settings(tmp_path):
    config = _config(tmp_path)
    config["_active_ef_search"] = 256
    config["_active_fetch_multiplier"] = 2.0
    config["_active_filter_mode"] = "postfilter"
    sql, params, settings, table = build_search_sql(
        config, "HNSW", "avgrating >= 4.0", 100, 0, m=32, ef_build=128
    )
    assert "vector_similarity" not in sql  # The index is defined in table DDL, not query SQL.
    assert "L2Distance" in sql
    assert "avgrating >= 4.0" in sql
    assert settings["hnsw_candidate_list_size_for_search"] == 256
    assert settings["vector_search_index_fetch_multiplier"] == 2.0
    assert settings["vector_search_filter_strategy"] == "postfilter"
    assert params["k"] == 100
    assert table.startswith("movies_hnsw_m32_ef128")


def test_qbit_query_uses_query_time_precision(tmp_path):
    config = _config(tmp_path)
    sql, params, settings, table = build_search_sql(config, "QBit", "", 10, 0)
    assert "L2DistanceTransposed" in sql
    assert "{bits:UInt8}" in sql
    assert settings["allow_experimental_qbit_type"] == 1
    assert params["k"] == 10
    assert table == "movies_qbit"
