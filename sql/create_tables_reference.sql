-- Reference DDL. The executable table builder renders these definitions from configs/pilot.yaml.
-- QBit is experimental and requires: SET allow_experimental_qbit_type = 1;

CREATE TABLE movies_exact
(
    mid String,
    avgrating Float32,
    mvector Array(BFloat16)
)
ENGINE = MergeTree
ORDER BY mid;

CREATE TABLE movies_qbit
(
    mid String,
    avgrating Float32,
    mvector QBit(BFloat16, 768)
)
ENGINE = MergeTree
ORDER BY mid;

CREATE TABLE movies_hnsw
(
    mid String,
    avgrating Float32,
    mvector Array(BFloat16),
    INDEX ann_idx mvector TYPE vector_similarity('hnsw', 'L2Distance', 768, 'bf16', 32, 128)
        GRANULARITY 100000000
)
ENGINE = MergeTree
ORDER BY mid;
