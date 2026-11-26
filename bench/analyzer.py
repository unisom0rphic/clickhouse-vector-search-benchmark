from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _bootstrap_mean_ci(
    values: pd.Series,
    clusters: pd.Series,
    n_boot: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "cluster": clusters.astype(str)})
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        return np.nan, np.nan
    cluster_means = frame.groupby("cluster", sort=False)["value"].mean().to_numpy(dtype=float)
    if len(cluster_means) == 1:
        value = float(cluster_means[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cluster_means), size=(int(n_boot), len(cluster_means)))
    estimates = cluster_means[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def aggregate_results(
    measurements_path: Path,
    configurations_path: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if not measurements_path.exists():
        raise FileNotFoundError(f"Measurements CSV not found: {measurements_path}")
    measurements = pd.read_csv(measurements_path)
    # Backward compatibility with early CSVs where the natural GLS group was
    # temporarily stored under the now-reserved GLS_target column.
    if "GLS_Group" not in measurements.columns and "GLS_target" in measurements.columns:
        measurements["GLS_Group"] = measurements["GLS_target"]
    configurations = pd.read_csv(configurations_path) if configurations_path.exists() else pd.DataFrame()
    if "Run_notes" in measurements.columns:
        notes = measurements["Run_notes"].fillna("").astype(str)
        measurements = measurements[~notes.str.startswith("ERROR:") & (notes != "query_log_timeout")]
    if "Latency_ms" in measurements.columns:
        measurements = measurements[pd.to_numeric(measurements["Latency_ms"], errors="coerce").notna()]
    if measurements.empty:
        raise ValueError("Measurements CSV contains no completed rows with server-side latency")

    _safe_numeric(measurements, [
        "Sigma_target", "Sigma_actual", "GLS_actual", "GLS_kNN", "k_requested", "eligible_rows",
        "k_effective", "GT_count", "TP", "returned_rows", "Recall_at_k", "Fill_rate",
        "Latency_ms", "read_rows", "read_bytes", "peak_mem_MB",
    ])
    if not configurations.empty:
        add_columns = [c for c in [
            "Config_ID", "Method", "Filter_mode", "HNSW_ef_search", "Fetch_multiplier",
            "QBit_p", "HNSW_M", "HNSW_ef_build", "Rescoring",
        ] if c in configurations.columns]
        measurements = measurements.merge(
            configurations[add_columns], on="Config_ID", how="left", suffixes=("", "_config")
        )

    group_columns = [
        col for col in [
            "Config_ID", "Method", "Filter_mode", "Filter_type", "Sigma_target", "GLS_Group",
            "k_requested", "QBit_p", "HNSW_ef_search", "Fetch_multiplier", "HNSW_M",
        ] if col in measurements.columns
    ]
    aggregates = measurements.groupby(group_columns, dropna=False, observed=False).agg(
        n_observations=("Latency_ms", "count"),
        n_queries=("Query_ID", "nunique"),
        recall_mean=("Recall_at_k", "mean"),
        recall_std=("Recall_at_k", "std"),
        fill_rate_mean=("Fill_rate", "mean"),
        latency_mean_ms=("Latency_ms", "mean"),
        read_rows_mean=("read_rows", "mean"),
        read_bytes_mean=("read_bytes", "mean"),
        peak_mem_mean_mb=("peak_mem_MB", "mean"),
    ).reset_index()

    # Repeats are technical replicates. Percentiles describe query-to-query latency,
    # so first collapse repeats for each query/filter pair, then take percentiles.
    cluster_key = measurements["Query_ID"].astype(str)
    if "Filter_ID" in measurements:
        cluster_key = cluster_key + "::" + measurements["Filter_ID"].astype(str)
    latency_cluster = measurements.assign(_query_cluster=cluster_key).groupby(
        group_columns + ["_query_cluster"], dropna=False, observed=False
    )["Latency_ms"].mean().reset_index()
    latency_percentiles = latency_cluster.groupby(
        group_columns, dropna=False, observed=False
    )["Latency_ms"].agg(
        latency_p50_ms="median",
        latency_p95_ms=lambda values: values.quantile(0.95),
        latency_p99_ms=lambda values: values.quantile(0.99),
    ).reset_index()
    aggregates = aggregates.merge(latency_percentiles, on=group_columns, how="left")

    n_boot = int(settings.get("bootstrap_samples", 1000))
    confidence = float(settings.get("confidence_level", 0.95))
    grouped = measurements.groupby(group_columns, dropna=False, observed=False)
    ci_rows: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(grouped):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_columns, key_tuple, strict=True))
        clusters = group["Query_ID"].astype(str)
        if "Filter_ID" in group:
            clusters = clusters + "::" + group["Filter_ID"].astype(str)
        recall_low, recall_high = _bootstrap_mean_ci(
            group["Recall_at_k"], clusters, n_boot, confidence, index + 13
        )
        latency_low, latency_high = _bootstrap_mean_ci(
            group["Latency_ms"], clusters, n_boot, confidence, index + 29
        )
        ci_rows.append({
            **key_dict,
            "recall_ci_low": recall_low,
            "recall_ci_high": recall_high,
            "latency_mean_ci_low_ms": latency_low,
            "latency_mean_ci_high_ms": latency_high,
        })
    if ci_rows:
        aggregates = aggregates.merge(pd.DataFrame(ci_rows), on=group_columns, how="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates.to_csv(output_dir / "aggregates.csv", index=False)
    return aggregates


def _pareto_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Pareto frontiers only within identical workload scenarios."""
    required = {"recall_mean", "latency_p50_ms", "peak_mem_mean_mb", "Config_ID"}
    if not required.issubset(frame.columns):
        return frame.iloc[0:0].copy()
    clean = frame.dropna(subset=list(required - {"Config_ID"})).copy()
    scenario_columns = [
        column for column in ["Filter_type", "Sigma_target", "GLS_Group", "k_requested"]
        if column in clean.columns
    ]
    if not scenario_columns:
        scenario_groups = [((), clean)]
    else:
        scenario_groups = clean.groupby(scenario_columns, dropna=False, sort=False)
    frontiers: list[pd.DataFrame] = []
    for _, group in scenario_groups:
        values = group[["recall_mean", "latency_p50_ms", "peak_mem_mean_mb"]].to_numpy(dtype=float)
        keep: list[bool] = []
        for point in values:
            dominated = np.any(
                (values[:, 0] >= point[0])
                & (values[:, 1] <= point[1])
                & (values[:, 2] <= point[2])
                & ((values[:, 0] > point[0]) | (values[:, 1] < point[1]) | (values[:, 2] < point[2]))
            )
            keep.append(not bool(dominated))
        frontiers.append(group.loc[keep])
    return pd.concat(frontiers, ignore_index=True) if frontiers else clean.iloc[0:0].copy()


def _display_config(row: pd.Series) -> str:
    method = str(row.get("Method", "Method"))
    parts = [method]
    if pd.notna(row.get("QBit_p")):
        parts.append(f"p={int(row['QBit_p'])}")
    if pd.notna(row.get("HNSW_ef_search")):
        parts.append(f"ef={int(row['HNSW_ef_search'])}")
    if pd.notna(row.get("Fetch_multiplier")):
        parts.append(f"fetch={row['Fetch_multiplier']}")
    if pd.notna(row.get("Filter_mode")):
        parts.append(str(row["Filter_mode"]))
    return " / ".join(parts)


def _focus_slice(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Select the exact configured plot slice; never silently mix other workloads."""
    result = frame.copy()
    focus_k = settings.get("focus_k")
    focus_sigma = settings.get("focus_sigma")
    focus_gls = settings.get("focus_gls_group")
    if focus_k is not None and "k_requested" in result:
        result = result[pd.to_numeric(result["k_requested"], errors="coerce") == int(focus_k)]
    if focus_sigma is not None and "Sigma_target" in result:
        sigma_values = pd.to_numeric(result["Sigma_target"], errors="coerce")
        result = result[np.isclose(sigma_values, float(focus_sigma))]
    if focus_gls and "GLS_Group" in result:
        result = result[result["GLS_Group"].astype(str) == str(focus_gls)]
    return result


def _save_plot(fig, png_path: Path, pdf: bool) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    if pdf:
        fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def create_plots(measurements_path: Path, configurations_path: Path, output_dir: Path, settings: dict[str, Any]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    measurements = pd.read_csv(measurements_path)
    if "GLS_Group" not in measurements.columns and "GLS_target" in measurements.columns:
        measurements["GLS_Group"] = measurements["GLS_target"]
    if "Run_notes" in measurements.columns:
        notes = measurements["Run_notes"].fillna("").astype(str)
        measurements = measurements[~notes.str.startswith("ERROR:") & (notes != "query_log_timeout")]
    if "Latency_ms" in measurements.columns:
        measurements = measurements[pd.to_numeric(measurements["Latency_ms"], errors="coerce").notna()]
    if measurements.empty:
        LOG.warning("No completed measurements available for plotting")
        return
    configs = pd.read_csv(configurations_path) if configurations_path.exists() else pd.DataFrame()
    _safe_numeric(measurements, [
        "Sigma_target", "Sigma_actual", "GLS_actual", "k_requested", "Recall_at_k", "Latency_ms",
        "read_bytes", "peak_mem_MB", "QBit_p", "HNSW_ef_search", "Fetch_multiplier",
    ])
    if not configs.empty:
        add_columns = [c for c in [
            "Config_ID", "Method", "Filter_mode", "QBit_p", "HNSW_ef_search", "Fetch_multiplier", "HNSW_M",
        ] if c in configs.columns]
        measurements = measurements.merge(configs[add_columns], on="Config_ID", how="left", suffixes=("", "_config"))
    if "Method" not in measurements:
        measurements["Method"] = measurements["Config_ID"]
    measurements["Config_Label"] = measurements.apply(_display_config, axis=1)

    # 01: scatter points are grouped by config and workload slice; no hidden averaging across k/GLS.
    trade = measurements.groupby(
        ["Config_ID", "Config_Label", "Method", "k_requested", "Sigma_target", "GLS_Group"],
        dropna=False,
    ).agg(recall=("Recall_at_k", "mean"), latency=("Latency_ms", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for method, group in trade.groupby("Method", dropna=False):
        ax.scatter(group["latency"], group["recall"], label=str(method), alpha=0.7)
    ax.set_xlabel("Median latency (ms)")
    ax.set_ylabel("Mean Recall@k")
    ax.set_title("Recall vs latency — points are separate workload slices")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, plot_dir / "01_recall_vs_latency.png", bool(settings.get("make_pdf", True)))

    focus = _focus_slice(measurements, settings)

    # 02: latency vs selectivity for a configurable slice (default pilot k/sigma/GLS).
    sigma = measurements.copy()
    focus_k = settings.get("focus_k")
    focus_gls = settings.get("focus_gls_group")
    if focus_k is not None:
        sigma = sigma[pd.to_numeric(sigma["k_requested"], errors="coerce") == int(focus_k)]
    if focus_gls and "GLS_Group" in sigma:
        sigma = sigma[sigma["GLS_Group"].astype(str) == str(focus_gls)]
    sigma_agg = sigma.groupby(["Config_ID", "Config_Label", "Sigma_target"], dropna=False).agg(
        latency=("Latency_ms", "median")
    ).reset_index()
    if not sigma_agg.empty:
        fig, ax = plt.subplots(figsize=(9.0, 5.8))
        for label, group in sigma_agg.groupby("Config_Label", dropna=False):
            group = group.sort_values("Sigma_target")
            ax.plot(group["Sigma_target"], group["latency"], marker="o", label=str(label))
        ax.set_xscale("log")
        ax.set_xlabel("Target selectivity σ")
        ax.set_ylabel("Median latency (ms)")
        ax.set_title(f"Latency vs selectivity (k={focus_k or 'all'}, GLS={focus_gls or 'all'})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        _save_plot(fig, plot_dir / "02_latency_vs_sigma.png", bool(settings.get("make_pdf", True)))
    else:
        LOG.warning("Plot 02 skipped: no observations match the configured k/GLS focus slice")

    # QBit curves are explicitly sliced by the focus workload to avoid averaging across k/GLS.
    qbit = focus[focus["Method"] == "QBit"] if "Method" in focus else pd.DataFrame()
    if not qbit.empty and "QBit_p" in qbit:
        qagg = qbit.groupby("QBit_p", dropna=False).agg(
            recall=("Recall_at_k", "mean"), latency=("Latency_ms", "median")
        ).reset_index().sort_values("QBit_p")
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        ax.plot(qagg["QBit_p"], qagg["recall"], marker="o")
        ax.set_xlabel("QBit precision (bits)")
        ax.set_ylabel("Mean Recall@k")
        ax.set_title("QBit precision vs recall — configured focus slice")
        ax.grid(True, alpha=0.25)
        _save_plot(fig, plot_dir / "03_qbit_precision_vs_recall.png", bool(settings.get("make_pdf", True)))
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        ax.plot(qagg["QBit_p"], qagg["latency"], marker="o")
        ax.set_xlabel("QBit precision (bits)")
        ax.set_ylabel("Median latency (ms)")
        ax.set_title("QBit precision vs latency — configured focus slice")
        ax.grid(True, alpha=0.25)
        _save_plot(fig, plot_dir / "04_qbit_precision_vs_latency.png", bool(settings.get("make_pdf", True)))

    # 05: one heatmap per physical/query configuration; never average different ef/p settings together.
    if {"GLS_Group", "Sigma_target"}.issubset(measurements.columns):
        for config_id, group in measurements.groupby("Config_ID", dropna=False):
            if focus_k is not None:
                group = group[pd.to_numeric(group["k_requested"], errors="coerce") == int(focus_k)]
            if group.empty:
                continue
            pivot = group.pivot_table(index="GLS_Group", columns="Sigma_target", values="Recall_at_k", aggfunc="mean")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
            ax.set_xlabel("Target selectivity σ")
            ax.set_ylabel("Natural GLS group")
            ax.set_title(f"Recall heatmap — {group['Config_Label'].iloc[0] if 'Config_Label' in group else config_id}")
            fig.colorbar(image, ax=ax, label="Mean Recall@k")
            _save_plot(fig, plot_dir / f"05_sigma_gls_heatmap_{config_id}.png", bool(settings.get("make_pdf", True)))

    # 06: read volume by sigma for the same focus k/GLS slice (do not pin sigma).
    read = measurements.copy()
    if focus_k is not None:
        read = read[pd.to_numeric(read["k_requested"], errors="coerce") == int(focus_k)]
    if focus_gls and "GLS_Group" in read:
        read = read[read["GLS_Group"].astype(str) == str(focus_gls)]
    read = read.groupby(["Config_ID", "Config_Label", "Sigma_target"], dropna=False).agg(
        read_bytes=("read_bytes", "mean")
    ).reset_index()
    if read["read_bytes"].notna().any():
        fig, ax = plt.subplots(figsize=(9.0, 5.8))
        for label, group in read.groupby("Config_Label", dropna=False):
            group = group.sort_values("Sigma_target")
            ax.plot(group["Sigma_target"], group["read_bytes"], marker="o", label=str(label))
        ax.set_xscale("log")
        ax.set_xlabel("Target selectivity σ")
        ax.set_ylabel("Mean read_bytes")
        ax.set_title("Read volume vs selectivity — configured focus k/GLS")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        _save_plot(fig, plot_dir / "06_read_bytes_vs_sigma.png", bool(settings.get("make_pdf", True)))

    # 07: recall vs k for a configured selectivity/GLS slice.
    by_k = measurements.copy()
    focus_sigma = settings.get("focus_sigma")
    if focus_sigma is not None:
        by_k = by_k[np.isclose(pd.to_numeric(by_k["Sigma_target"], errors="coerce"), float(focus_sigma))]
    if focus_gls and "GLS_Group" in by_k:
        by_k = by_k[by_k["GLS_Group"].astype(str) == str(focus_gls)]
    by_k = by_k.groupby(["Config_ID", "Config_Label", "k_requested"], dropna=False).agg(
        recall=("Recall_at_k", "mean")
    ).reset_index()
    if not by_k.empty:
        fig, ax = plt.subplots(figsize=(9.0, 5.8))
        for label, group in by_k.groupby("Config_Label", dropna=False):
            group = group.sort_values("k_requested")
            ax.plot(group["k_requested"], group["recall"], marker="o", label=str(label))
        ax.set_xscale("log")
        ax.set_xlabel("Requested k")
        ax.set_ylabel("Mean Recall@k")
        ax.set_title(f"Recall vs result size (σ={focus_sigma or 'all'}, GLS={focus_gls or 'all'})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        _save_plot(fig, plot_dir / "07_recall_vs_k.png", bool(settings.get("make_pdf", True)))
    else:
        LOG.warning("Plot 07 skipped: no observations match the configured sigma/GLS focus slice")

    # 08: query memory by exact configuration, not by method-wide average.
    if measurements["peak_mem_MB"].notna().any():
        memory = measurements.groupby(["Config_ID", "Config_Label"], dropna=False)["peak_mem_MB"].mean().dropna()
        fig, ax = plt.subplots(figsize=(10.0, 5.8))
        ax.bar([str(index[1]) for index in memory.index], memory.values)
        ax.set_ylabel("Mean query memory (MB)")
        ax.set_title("Query memory by configuration")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        _save_plot(fig, plot_dir / "08_memory_comparison.png", bool(settings.get("make_pdf", True)))

    # 09: fetch multiplier response at the same focus slice.
    hnsw = focus[focus["Method"] == "HNSW"] if "Method" in focus else pd.DataFrame()
    if not hnsw.empty and "Fetch_multiplier" in hnsw and hnsw["Fetch_multiplier"].notna().any():
        fetch = hnsw.groupby(["Filter_mode", "HNSW_ef_search", "Fetch_multiplier"], dropna=False).agg(
            recall=("Recall_at_k", "mean"), latency=("Latency_ms", "median")
        ).reset_index()
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for (mode, ef), group in fetch.groupby(["Filter_mode", "HNSW_ef_search"], dropna=False):
            group = group.sort_values("Fetch_multiplier")
            ax.plot(group["Fetch_multiplier"], group["recall"], marker="o", label=f"{mode}, ef={ef}")
        ax.set_xlabel("vector_search_index_fetch_multiplier")
        ax.set_ylabel("Mean Recall@k")
        ax.set_title("HNSW fetch multiplier vs recall — configured focus slice")
        ax.grid(True, alpha=0.25)
        ax.legend()
        _save_plot(fig, plot_dir / "09_fetch_multiplier.png", bool(settings.get("make_pdf", True)))


def run_analysis(config: dict[str, Any], *, no_plots: bool = False, only_plots: bool = False) -> None:
    root = Path(config["_project_root"])
    raw = root / config["project"].get("output_dir", "results") / "raw"
    output = root / config["project"].get("output_dir", "results") / "analysis"
    measurements = raw / "measurements.csv"
    configurations = raw / "configurations.csv"
    output.mkdir(parents=True, exist_ok=True)

    if not only_plots:
        _ = aggregate_results(measurements, configurations, output, config.get("analysis", {}))
        aggregates = pd.read_csv(output / "aggregates.csv")
        pareto = _pareto_frontier(aggregates)
        pareto.to_csv(output / "pareto.csv", index=False)
        summary = [
            "# Benchmark analysis summary",
            "",
            f"- Measurements: {len(pd.read_csv(measurements))}",
            f"- Aggregate groups: {len(aggregates)}",
            f"- Pareto configurations/groups: {len(pareto)}",
            "- Raw CSV files were not modified.",
            "",
            "## Best median latency by method",
        ]
        if "Method" in aggregates and "latency_p50_ms" in aggregates:
            best = aggregates.sort_values("latency_p50_ms").groupby("Method", dropna=False).head(1)
            for _, row in best.iterrows():
                summary.append(f"- {row['Method']}: p50={row['latency_p50_ms']:.3f} ms; recall={row.get('recall_mean', float('nan')):.4f}")
        (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    if not no_plots:
        create_plots(measurements, configurations, output, config.get("analysis", {}))
    LOG.info("Analysis outputs written to %s", output)
