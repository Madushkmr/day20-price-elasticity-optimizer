#!/usr/bin/env python3
"""
Command-line interface for the Price Elasticity & Promotion ROI Optimizer.

Usage:
    python cli.py compute
    python cli.py list-runs
    python cli.py show-run <run_id>
    python cli.py show-product <run_id> <product_id>
"""
import argparse
import json
import os
import sys

import yaml

from src import engine, db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SALES = os.path.join(BASE_DIR, "sample_data", "sales_history.csv")
DEFAULT_COSTS = os.path.join(BASE_DIR, "sample_data", "product_costs.csv")
DEFAULT_DB = os.path.join(BASE_DIR, "pricing_runs.db")
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config", "settings.yaml")


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_compute(args):
    config = load_config(args.config)
    result = engine.run_pipeline(args.sales, args.costs, args.db, config)
    print(f"Run #{result['run_id']} complete.")
    print(f"  Products priced:            {len(result['products'])}")
    print(f"  Optimized incremental margin: ${result['summary']['total_incremental_margin']:,.2f}")
    print(f"  Baseline incremental margin:  ${result['summary']['baseline_incremental_margin']:,.2f}")
    print(f"  Prob. optimized beats baseline: {result['summary']['prob_beats_baseline']:.0%}")
    if result["warnings"]:
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")
    print()
    print(result["summary"]["narrative"])


def cmd_list_runs(args):
    conn = db.get_connection(args.db)
    runs = db.list_runs(conn)
    conn.close()
    if not runs:
        print("No runs yet. Run `python cli.py compute` first.")
        return
    for r in runs:
        print(f"#{r['id']:<4} {r['created_at']:<26} "
              f"incremental_margin=${r['total_incremental_margin']:,.2f}  "
              f"prob_beats_baseline={r['prob_beats_baseline']:.0%}")


def cmd_show_run(args):
    conn = db.get_connection(args.db)
    run = db.get_run(conn, args.run_id)
    conn.close()
    if run is None:
        print(f"Run #{args.run_id} not found.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(run, indent=2))


def cmd_show_product(args):
    conn = db.get_connection(args.db)
    row = db.get_run_product(conn, args.run_id, args.product_id)
    conn.close()
    if row is None:
        print(f"Product '{args.product_id}' not found in run #{args.run_id}.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(row, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Price Elasticity & Promotion ROI Optimizer CLI")
    parser.add_argument("--db", default=DEFAULT_DB, help="path to SQLite database")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to settings.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compute = sub.add_parser("compute", help="run the full pipeline once and persist the result")
    p_compute.add_argument("--sales", default=DEFAULT_SALES)
    p_compute.add_argument("--costs", default=DEFAULT_COSTS)
    p_compute.set_defaults(func=cmd_compute)

    p_list = sub.add_parser("list-runs", help="list all persisted runs")
    p_list.set_defaults(func=cmd_list_runs)

    p_show = sub.add_parser("show-run", help="show full detail for one run")
    p_show.add_argument("run_id", type=int)
    p_show.set_defaults(func=cmd_show_run)

    p_prod = sub.add_parser("show-product", help="show detail for one product within a run")
    p_prod.add_argument("run_id", type=int)
    p_prod.add_argument("product_id")
    p_prod.set_defaults(func=cmd_show_product)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
