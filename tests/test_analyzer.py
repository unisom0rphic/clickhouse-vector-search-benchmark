import pandas as pd

from bench.analyzer import run_analysis


def test_analyze_is_offline_and_creates_csv_and_plots(tmp_path):
    raw = tmp_path / "results/raw"
    raw.mkdir(parents=True)
    pd.DataFrame([
        {"Config_ID": "exact", "Method": "Exact", "Filter_mode": "scan_filter_first"},
        {"Config_ID": "q8", "Method": "QBit", "Filter_mode": "scan_filter_first", "QBit_p": 8},
    ]).to_csv(raw / "configurations.csv", index=False)
    pd.DataFrame([
        {"Config_ID": "exact", "Run_ID": "r1", "Query_ID": "q1", "Filter_ID": "f1", "Sigma_target": 0.1,
         "Sigma_actual": 0.1, "GLS_Group": "medium", "k_requested": 10, "Recall_at_k": 1.0,
         "Fill_rate": 1.0, "Latency_ms": 15.0, "read_rows": 100, "read_bytes": 10000, "peak_mem_MB": 20},
        {"Config_ID": "exact", "Run_ID": "r1", "Query_ID": "q2", "Filter_ID": "f1", "Sigma_target": 0.1,
         "Sigma_actual": 0.1, "GLS_Group": "medium", "k_requested": 10, "Recall_at_k": 1.0,
         "Fill_rate": 1.0, "Latency_ms": 14.0, "read_rows": 100, "read_bytes": 10000, "peak_mem_MB": 20},
        {"Config_ID": "q8", "Run_ID": "r1", "Query_ID": "q1", "Filter_ID": "f1", "Sigma_target": 0.1,
         "Sigma_actual": 0.1, "GLS_Group": "medium", "k_requested": 10, "Recall_at_k": 0.9,
         "Fill_rate": 1.0, "Latency_ms": 5.0, "read_rows": 100, "read_bytes": 3000, "peak_mem_MB": 5},
        {"Config_ID": "q8", "Run_ID": "r1", "Query_ID": "q2", "Filter_ID": "f1", "Sigma_target": 0.1,
         "Sigma_actual": 0.1, "GLS_Group": "medium", "k_requested": 10, "Recall_at_k": 0.9,
         "Fill_rate": 1.0, "Latency_ms": 6.0, "read_rows": 100, "read_bytes": 3000, "peak_mem_MB": 5},
    ]).to_csv(raw / "measurements.csv", index=False)

    config = {
        "_project_root": str(tmp_path),
        "project": {"output_dir": "results"},
        "analysis": {"bootstrap_samples": 30, "confidence_level": 0.95, "make_pdf": False},
    }
    run_analysis(config)
    assert (tmp_path / "results/analysis/aggregates.csv").exists()
    assert (tmp_path / "results/analysis/pareto.csv").exists()
    assert (tmp_path / "results/analysis/summary.md").exists()
    assert (tmp_path / "results/analysis/plots/01_recall_vs_latency.png").exists()
