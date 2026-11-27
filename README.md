# ClickHouse Vector Search Benchmark

Reproducible benchmark harness for comparing ClickHouse HNSW vector similarity indexes, QBit scans and exact filtered scans on the original MoReVec Movies data.

> **Current status: v0.1 scaffold.** The code, CLI, CSV schema, workload preparation, HDF5 inspection, query runner and offline analysis are implemented. The HDF5 key mappings in `configs/pilot.yaml` are prefilled from the upstream MoReVec loader; inspect the downloaded local files to verify them, then run the smoke test against the selected ClickHouse image. This repository has not yet been validated against a live ClickHouse server in this development environment.

## What it measures

- HNSW: `postfilter` and `auto` search strategies, with `M`, construction candidate list, search candidate list and fetch multiplier.
- QBit: full scan using `L2DistanceTransposed`, varying the number of bit planes read at query time.
- Exact baseline: exact L2 scan over the BFloat16-rounded values.
- Exact prefilter: ClickHouse's `vector_search_filter_strategy='prefilter'` path on an HNSW table. This is an exact scan over filtered rows, **not** traversal of a filtered HNSW subgraph.
- Query-level GLS is computed from the BFloat16-rounded working representation and the existing `avgrating` attribute. The MoReVec HDF5 source files and metadata remain read-only and unchanged. The primary local neighborhood is L=2048; the configured sensitivity neighborhood is L=8192. `GLS_Group` stores the measured natural group; `GLS_target` is intentionally left blank because this version does not impose artificial correlation targets.

The HDF5 source files are read-only. The benchmark converts the in-memory working vectors to BFloat16 precision and imports that representation into ClickHouse so the HNSW, QBit and exact paths compare the same stored values. The conversion is performed in memory only; it does not overwrite or edit MoReVec files. Dataset import is batched to avoid allocating a second full vector matrix.

## Project layout

```text
clickhouse-vector-search-benchmark/
├── benchmark.py
├── docker-compose.yml
├── configs/pilot.yaml
├── docker/clickhouse/              # server and user profiles
├── bench/                          # loader, GLS, runner, metrics, analyzer
├── sql/                            # reference DDL
├── data/                            # downloaded/prepared data (not committed)
├── results/raw/                     # CSV observations (not committed by default)
└── results/analysis/                # aggregates, plots and summary
```

## Setup

The benchmark environment and CI use Python 3.12.14. Docker Engine / Docker Desktop with Compose v2 is required for the ClickHouse service.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e .[dev]
cp .env.example .env
```

The starter `.env.example` pins ClickHouse to `26.8.44.3` (an exact patch tag with vector similarity indexes and QBit support). Keep this tag fixed for all runs reported in the paper; the run manifest records the actual running server version and Docker image ID. Change the tag only as a deliberate experiment-version change.

Download the MoReVec HDF5 package:

```bash
python benchmark.py download-morevec
```

The upstream benchmark repository publishes separate vector, filter and query HDF5 files under `data/datasets/MoRe_{size}`. Inspect them before setting key paths:

```bash
python benchmark.py inspect-data --path data/datasets/MoRe_large
```

Open `data/hdf5_inspection.json` to verify the downloaded files. The upstream MoReVec loader uses `train_mvector`, `train_mid`, `train_avgrating`, `train_year`, `train_genre`, `train_num_votes`, and `train_title` for the Movies table; these mappings are prefilled in `configs/pilot.yaml` based on the upstream loader. If the local HDF5 inspection differs, update the explicit keys rather than guessing.

The current pilot creates deterministic numeric-range predicates from the existing `train_avgrating` values. It does not modify the source arrays, add columns to MoReVec, or synthesize new vector values. Query vectors are sampled from the original movie collection and each sampled movie ID is excluded from its own candidate set. The separate original MoReVec filter/query HDF5 workload files are not yet wired into this first workload generator; this pilot therefore uses generated predicates over an existing metadata attribute, not the benchmark authors' pre-defined query/filter workload. The distinction is recorded in `data/prepared/workload_manifest.json` and must be stated when reporting results.

## Run the pipeline

Start ClickHouse:

```bash
docker compose up -d
python benchmark.py preflight
```

Prepare the query/filter workload, GLS and exact ground truth. The progress bar counts completed queries, including all filters for each query:

```bash
python benchmark.py prepare
```

Load the BFloat16 working table and build QBit/HNSW tables:

```bash
python benchmark.py load
```

Run a small validation first. Smoke tests execute a filtered and an unfiltered query (or the requested number of query/filter pairs) for every generated configuration, so all HNSW/QBit/Exact plans get exercised:

```bash
python benchmark.py smoke --max-cases 2
```

Then execute the configured pilot:

```bash
python benchmark.py run --resume
```

Analyze the saved CSVs without connecting to ClickHouse:

```bash
python benchmark.py analyze
```

`analyze` writes `aggregates.csv`, `pareto.csv`, `summary.md`, and PNG/PDF plots under `results/analysis/`. Pareto frontiers are computed only within matching filter/selectivity/GLS/k scenarios. The precision/latency and other slice-dependent plots use `analysis.focus_k`, `analysis.focus_sigma`, and `analysis.focus_gls_group` from YAML; they do not silently mix other workloads when the selected slice is absent. `--no-plots` skips figures; `--only-plots` rebuilds figures from the existing CSVs without recomputing aggregates.

## CSV outputs

- `results/raw/configurations.csv`: one row per method configuration.
- `results/raw/measurements.csv`: one row per `(configuration, query, filter, k, repeat)`. `Filter_ID` is included so same query IDs under different predicates remain unambiguous.
- `results/raw/resource_snapshots.csv`: container RSS and vector index cache snapshots where available.
- `results/raw/build_info.json`: table build times, part counts and on-disk sizes.
- `results/raw/run_manifest.json` and `results/raw/run_manifests/`: latest and archived run manifests, including the exact server version, Docker image ID, config hash, dataset/workload signatures and environment details.

The measurement columns preserve the workbook schema, with `Filter_ID` and `GLS_Group` added. `Filter_ID` uniquely identifies each query/filter pair; `GLS_Group` distinguishes the measured natural category from the reserved, intentionally empty `GLS_target` field. Source CSV files are append-only and are not modified by analysis. Tables in ClickHouse are namespaced with a short fingerprint of the prepared dataset so changing the source arrays cannot silently reuse same-row-count tables from an earlier dataset.

## Commands

```bash
python benchmark.py --help
python benchmark.py inspect-data --path <file-or-directory>
python benchmark.py reset-tables --yes   # destructive: drops benchmark tables
```

## Important experimental notes

1. The pilot uses a small workload. It is not a paper result configuration.
2. If queries are sampled from the indexed collection, their own IDs are excluded from result candidates. For an externally supplied query set, configure `source.queries.file` and `vectors_key`.
3. Ground truth is computed over the BFloat16-rounded working representation. This isolates search approximation from storage-type conversion. A separate Float32-vs-BFloat16 quality check can be added as a sensitivity analysis.
4. At low selectivity the L=2048 GLS estimate is noisy because only a few local neighbors may pass a predicate. `GLS_8192` is collected as a sensitivity value; interpret edge values, especially -1, cautiously.
5. QBit precision controls the number of bit planes read during search; it does not change the physical stored QBit column size.
6. `system.query_log` is the source for server-side query duration, read rows, read bytes and query memory. Container RSS is sampled separately and is not attributed to a single query. The vector similarity index cache is explicitly cleared between configurations; the warm-up is then performed before timed observations. Only warm-cache measurements are currently supported; the config rejects a `cold` label rather than pretending that relabeling clears OS/page caches.
7. The code saves EXPLAIN artifacts. Verify plans in the smoke phase before starting long runs. In particular, confirm that HNSW `postfilter` and `auto` actually use the vector index, and that forced `prefilter` executes a filtered exact scan.

## Upstream data and implementation references

- MoReVec benchmark data and workload repository: <https://github.com/aabylay/ANN-benchmark-HQ>
- ClickHouse vector search and QBit documentation: <https://clickhouse.com/docs/guides/use-cases/ai-ml/vector-search>

The dataset is not redistributed in this repository. Review the upstream terms before publishing any derived artifacts or redistributing the data.

## License

The benchmark harness code is released under the MIT License. This license does not apply to the third-party MoReVec dataset or ClickHouse.
