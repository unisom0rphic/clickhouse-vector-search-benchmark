from bench.csvio import CsvAppender, load_completed_measurements, read_csv_dicts
from bench.schema import MEASUREMENT_FIELDS


def test_csv_appender_and_resume_key(tmp_path):
    path = tmp_path / "measurements.csv"
    writer = CsvAppender(path, MEASUREMENT_FIELDS)
    row = {
        "Config_ID": "cfg1", "Query_ID": "q1", "Filter_ID": "f1",
        "k_requested": 10, "Repeat": 1, "Run_notes": "", "Recall_at_k": 0.9,
        "Latency_ms": 12.5,
    }
    writer.append(row)
    writer.append({**row, "Query_ID": "q2", "Run_notes": "ERROR: test"})
    rows = read_csv_dicts(path)
    assert len(rows) == 2
    completed = load_completed_measurements(path)
    assert ("cfg1", "q1", "f1", 10, 1) in completed
    assert ("cfg1", "q2", "f1", 10, 1) not in completed
