"""Read-only replay of one Movies smoke case, using existing prepared files and tables."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.clickhouse import create_client, qualified_table, quote_identifier
from bench.config import load_config
from bench.sql import build_search_sql
from bench.workload import load_ground_truth, load_prepared_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--query", default="q_000000")
    parser.add_argument("--filter", default="avgrating_ge_9p200000_t0p0100")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.config)
    prepared = Path(config["_project_root"]) / "data/prepared"
    vectors, query_ids, excluded_ids = load_prepared_queries(config)
    idx = query_ids.index(args.query)
    with (prepared / "query_workload.csv").open() as stream:
        rows = {r["Filter_ID"]: r for r in csv.DictReader(stream) if r["Query_ID"] == args.query}
    filters = {f["filter_id"]: f for f in json.loads((prepared / "filters.json").read_text())}
    cols = config["source"]["columns"]
    mid, attr, vec = [quote_identifier(cols[key]) for key in ("id", "filter_attribute", "vector")]
    dist = f"L2Distance({vec}, CAST({{q:Array(Float32)}}, 'Array(BFloat16)'))"
    client = create_client(config)
    try:
        print("ClickHouse", client.command("SELECT version()"))
        for fid in ("unfiltered", args.filter):
            row = rows[fid]
            assert int(row["Query_Index"]) == idx and row["Excluded_ID"] == excluded_ids[idx]
            k = min(args.k, int(row["Eligible_Rows"]))
            gt = load_ground_truth(config, idx, fid)[:k].tolist()
            print("\n", args.query, fid, "GT", gt, "eligible", row["Eligible_Rows"])
            predicates = {"prepared": row["Filter_SQL"]}
            if fid != "unfiltered":
                predicates["full precision"] = f"{attr} >= {filters[fid]['threshold']!r}"
            for label, predicate in predicates.items():
                terms = [predicate] if predicate else []
                if excluded_ids[idx]:
                    terms.insert(0, f"{mid} != {{excluded:String}}")
                condition = " AND ".join(terms)
                for method in ("Exact", "ExactPrefilter"):
                    sql, params, settings, name = build_search_sql(
                        config, method, condition, k, idx,
                        m=config["hnsw"]["m"][0], ef_build=config["hnsw"]["ef_construction"][0],
                    )
                    table = qualified_table(config["clickhouse"]["database"], name)
                    params.update(q=vectors[idx].tolist(), excluded=excluded_ids[idx])
                    def query(statement):
                        return client.query(statement, parameters=params, settings=settings).result_rows
                    where = f" WHERE {condition}" if condition else ""
                    eligible = query(f"SELECT count() FROM {table}{where}")[0][0]
                    results = query(sql)
                    ids = {str(r[0]) for r in results}
                    print(method, label, "eligible", eligible, "top-k (ID, L2)", results,
                          "TP", len(ids.intersection(gt)), "/", len(gt))
                    print("\n".join(str(r[0]) for r in query("EXPLAIN indexes=1 " + sql)))
                    mismatches = ids.symmetric_difference(gt)
                    if mismatches:
                        params["diff"] = sorted(mismatches)
                        print("differing (ID, rating as Float64, L2, eligible)",
                              query(f"SELECT {mid}, toFloat64({attr}), {dist}, ({condition or '1'}) "
                                    f"FROM {table} WHERE {mid} IN {{diff:Array(String)}} ORDER BY {mid}"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
