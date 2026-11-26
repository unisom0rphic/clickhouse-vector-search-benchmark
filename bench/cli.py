from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .analyzer import run_analysis
from .clickhouse import create_client, wait_until_healthy
from .config import load_config, resolve_path
from .dataset import load_movie_dataset
from .hdf5 import inspect_hdf5
from .manifest import write_run_manifest
from .runner import run_benchmark
from .storage import build_indexes, import_dataset, reset_tables
from .workload import prepare_workload

LOG = logging.getLogger("chvecbench")


def _load(config_path: str) -> dict[str, Any]:
    return load_config(config_path)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def cmd_preflight(config: dict[str, Any], args: argparse.Namespace) -> int:
    client = create_client(config)
    try:
        client.query("SELECT 1")
        version = client.query("SELECT version()").result_rows[0][0]
        database = config["clickhouse"]["database"]
        settings = client.query(
            """SELECT name, value FROM system.settings
               WHERE name IN (
                 'max_threads', 'max_limit_for_vector_search_queries',
                 'hnsw_candidate_list_size_for_search',
                 'vector_search_filter_strategy', 'allow_experimental_qbit_type'
               ) ORDER BY name"""
        ).result_rows
        _print_json({
            "status": "ok",
            "version": version,
            "database": database,
            "settings": {str(name): str(value) for name, value in settings},
            "qbit_setting": "allow_experimental_qbit_type" in {str(row[0]) for row in settings},
        })
        return 0
    finally:
        client.close()


def cmd_inspect_data(config: dict[str, Any], args: argparse.Namespace) -> int:
    paths: list[Path] = []
    if args.path:
        path = Path(args.path).expanduser().resolve()
        if path.is_dir():
            paths = sorted(path.rglob("*.hdf5")) + sorted(path.rglob("*.h5"))
        else:
            paths = [path]
    else:
        source = config["source"]
        vector_file = resolve_path(config, source.get("data_file"))
        if vector_file:
            paths.append(vector_file)
        for spec in source.get("attributes", {}).values():
            if isinstance(spec, dict) and spec.get("file"):
                path = resolve_path(config, spec["file"])
                if path and path not in paths:
                    paths.append(path)
        query_file = resolve_path(config, source.get("queries", {}).get("file"))
        if query_file and query_file not in paths:
            paths.append(query_file)
    if not paths:
        raise ValueError("No HDF5 paths found. Pass --path or set source.data_file.")

    report: dict[str, Any] = {}
    for path in paths:
        report[str(path)] = inspect_hdf5(path)
    out = Path(config["_project_root"]) / "data" / "hdf5_inspection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_json(report)
    LOG.info("Inspection saved to %s", out)
    return 0


def cmd_download_morevec(config: dict[str, Any], args: argparse.Namespace) -> int:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install project dependencies first: pip install -e .") from exc
    url = "https://drive.google.com/drive/folders/1AqAVI8ASROqrFCQdEMPB8RNzPwilijRp?usp=drive_link"
    output = Path(args.output or (Path(config["_project_root"]) / "data")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download_folder(url=url, output=str(output), quiet=False, remaining_ok=True)
    if not downloaded:
        raise RuntimeError("MoReVec download failed. See gdown output above.")
    hdf5_files = sorted(set(output.rglob("*.hdf5")) | set(output.rglob("*.h5")))
    LOG.info("MoReVec files downloaded under %s", output)
    if hdf5_files:
        LOG.info("Discovered HDF5 files:\n%s", "\n".join(str(path) for path in hdf5_files))
    else:
        LOG.warning("No HDF5 files discovered under %s; inspect the gdown output above.", output)
    LOG.warning("Inspect HDF5 keys before configuring source.* key mappings.")
    return 0


def cmd_prepare(config: dict[str, Any], args: argparse.Namespace) -> int:
    dataset = prepare_workload(config)
    LOG.info(
        "Prepared dataset rows=%d, dim=%d, queries=%d",
        dataset.size, dataset.dimension, len(dataset.queries),
    )
    return 0


def cmd_load(config: dict[str, Any], args: argparse.Namespace) -> int:
    client = create_client(config)
    try:
        dataset = load_movie_dataset(config)
        imported = import_dataset(client, config, dataset)
        indexes = build_indexes(client, config)
        build_dir = Path(config["_project_root"]) / config["project"].get("output_dir", "results") / "raw"
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "build_info.json").write_text(
            json.dumps({"import": imported, "indexes": indexes}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _print_json({"import": imported, "indexes": indexes})
        return 0
    finally:
        client.close()


def cmd_reset(config: dict[str, Any], args: argparse.Namespace) -> int:
    if not args.yes:
        raise RuntimeError("Destructive operation refused. Re-run with --yes to drop benchmark tables.")
    client = create_client(config)
    try:
        reset_tables(client, config)
        LOG.info("Benchmark tables dropped")
        return 0
    finally:
        client.close()


def cmd_run(config: dict[str, Any], args: argparse.Namespace, smoke: bool = False) -> int:
    client = create_client(config)
    try:
        version = str(client.query("SELECT version()").result_rows[0][0])
        if smoke:
            config["execution"]["smoke_mode"] = True
            config["execution"]["smoke_pairs_per_config"] = max(1, int(args.max_cases or 2))
            config["execution"]["max_cases"] = None
            config["execution"]["repeats"] = 1
            config["execution"]["warmup_queries_per_config"] = 0
            config["execution"]["randomize_order"] = False
            config["execution"]["stop_on_error"] = False
            config["workload"]["k_values"] = [min(int(x) for x in config["workload"]["k_values"])]
        manifest_path = write_run_manifest(config, version, "smoke" if smoke else "run")
        LOG.info("Run manifest written to %s", manifest_path)
        run_benchmark(client, config, version, resume=bool(args.resume))
        return 0
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chvecbench",
        description="Reproducible HNSW/QBit benchmark harness for ClickHouse and MoReVec.",
    )
    parser.add_argument("--config", default="configs/pilot.yaml", help="YAML experiment configuration")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="Check ClickHouse connection and settings")
    inspect = sub.add_parser("inspect-data", help="Inspect HDF5 keys, shapes and dtypes")
    inspect.add_argument("--path", help="HDF5 file or directory; defaults to configured source paths")
    download = sub.add_parser("download-morevec", help="Download the public MoReVec HDF5 package")
    download.add_argument("--output", help="Output directory (default: ./data)")
    sub.add_parser("prepare", help="Compute query workload, GLS and exact ground truth")
    sub.add_parser("load", help="Import BFloat16 vectors and build QBit/HNSW tables")
    run = sub.add_parser("run", help="Run the configured benchmark matrix")
    run.add_argument("--resume", action="store_true", help="Skip completed config/query/filter/k/repeat tuples")
    smoke = sub.add_parser("smoke", help="Run a short benchmark smoke test")
    smoke.add_argument("--max-cases", type=int, default=2, help="query/filter pairs per configuration (filtered + unfiltered)")
    smoke.add_argument("--resume", action="store_true")
    analyze = sub.add_parser("analyze", help="Aggregate CSV, create plots and write summary.md")
    analyze.add_argument("--no-plots", action="store_true", help="Only create aggregates and summary")
    analyze.add_argument("--only-plots", action="store_true", help="Rebuild plots from existing CSVs only")
    reset = sub.add_parser("reset-tables", help="Drop benchmark tables (destructive)")
    reset.add_argument("--yes", action="store_true", help="Confirm dropping benchmark tables")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = _load(args.config)
        dispatch = {
            "preflight": lambda: cmd_preflight(config, args),
            "inspect-data": lambda: cmd_inspect_data(config, args),
            "download-morevec": lambda: cmd_download_morevec(config, args),
            "prepare": lambda: cmd_prepare(config, args),
            "load": lambda: cmd_load(config, args),
            "run": lambda: cmd_run(config, args),
            "smoke": lambda: cmd_run(config, args, smoke=True),
            "analyze": lambda: (run_analysis(config, no_plots=args.no_plots, only_plots=args.only_plots) or 0),
            "reset-tables": lambda: cmd_reset(config, args),
        }
        return int(dispatch[args.command]())
    except Exception as exc:
        LOG.exception("Command failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
