CONFIG_FIELDS = [
    "Config_ID", "Method", "CH_version", "Dataset", "Rows_N", "Dim_D",
    "Vector_type", "Distance", "Filter_mode", "HNSW_M", "HNSW_ef_build",
    "HNSW_ef_search", "Fetch_multiplier", "Rescoring", "QBit_p",
    "max_limit", "max_threads", "Container_RAM_GB", "max_server_memory_GB",
    "Codec", "Parts", "Build_time_s", "Index_size_MB", "Config_notes", "Table_size_MB",
]

MEASUREMENT_FIELDS = [
    "Config_ID", "Run_ID", "Query_ID", "Filter_ID", "Repeat", "Cache_state", "Query_type",
    "Filter_type", "Sigma_target", "Sigma_actual", "GLS_target", "GLS_actual",
    "GLS_Group", "GLS_kNN", "k_requested", "eligible_rows", "k_effective", "GT_count", "TP",
    "returned_rows", "Recall_at_k", "Fill_rate", "Latency_ms", "read_rows",
    "read_bytes", "peak_mem_MB", "EXPLAIN_ref", "Run_notes",
]

RESOURCE_FIELDS = [
    "Config_ID", "Timestamp", "Phase", "Container_RSS_MB",
    "Vector_Index_Cache_Bytes", "Container_Limit_MB",
]

WORKLOAD_FIELDS = [
    "Query_ID", "Filter_ID", "Filter_SQL", "Filter_type", "Sigma_target",
    "Sigma_actual", "Local_K", "Local_Selectivity", "GLS", "GLS_Group",
    "Eligible_Rows", "Excluded_ID",
]
