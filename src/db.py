"""
SQLite persistence for pricing/optimization runs. Schema:

  runs(id, created_at, budget_cap, total_incremental_margin, total_discount_cost,
       baseline_incremental_margin, prob_beats_baseline, narrative, warnings_json)
  run_products(run_id, product_id, category, elasticity, ci_low, ci_high, r_squared,
               discount_pct, price, predicted_units, incremental_margin, discount_cost,
               narrative)
"""
import sqlite3
import json
import datetime
import os

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    budget_cap REAL,
    total_incremental_margin REAL,
    total_discount_cost REAL,
    baseline_incremental_margin REAL,
    baseline_discount_cost REAL,
    prob_beats_baseline REAL,
    prob_positive_margin REAL,
    narrative TEXT,
    warnings_json TEXT
);

CREATE TABLE IF NOT EXISTS run_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    product_id TEXT NOT NULL,
    category TEXT,
    elasticity REAL,
    ci_low REAL,
    ci_high REAL,
    r_squared REAL,
    discount_pct REAL,
    price REAL,
    predicted_units REAL,
    incremental_margin REAL,
    discount_cost REAL,
    narrative TEXT
);
"""


def get_connection(db_path):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_run(conn, summary, product_rows):
    """
    summary: dict with keys budget_cap, total_incremental_margin, total_discount_cost,
             baseline_incremental_margin, baseline_discount_cost, prob_beats_baseline,
             prob_positive_margin, narrative, warnings (list[str])
    product_rows: list of dicts, one per product
    """
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO runs
           (created_at, budget_cap, total_incremental_margin, total_discount_cost,
            baseline_incremental_margin, baseline_discount_cost, prob_beats_baseline,
            prob_positive_margin, narrative, warnings_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.datetime.utcnow().isoformat(),
            summary["budget_cap"],
            summary["total_incremental_margin"],
            summary["total_discount_cost"],
            summary["baseline_incremental_margin"],
            summary["baseline_discount_cost"],
            summary["prob_beats_baseline"],
            summary["prob_positive_margin"],
            summary["narrative"],
            json.dumps(summary.get("warnings", [])),
        ),
    )
    run_id = cur.lastrowid
    for row in product_rows:
        cur.execute(
            """INSERT INTO run_products
               (run_id, product_id, category, elasticity, ci_low, ci_high, r_squared,
                discount_pct, price, predicted_units, incremental_margin, discount_cost, narrative)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, row["product_id"], row.get("category"), row["elasticity"],
                row["ci_low"], row["ci_high"], row["r_squared"], row["discount_pct"],
                row["price"], row["predicted_units"], row["incremental_margin"],
                row["discount_cost"], row["narrative"],
            ),
        )
    conn.commit()
    return run_id


def list_runs(conn):
    rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def get_run(conn, run_id):
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    run = dict(row)
    run["warnings"] = json.loads(run.pop("warnings_json") or "[]")
    products = conn.execute(
        "SELECT * FROM run_products WHERE run_id = ? ORDER BY incremental_margin DESC", (run_id,)
    ).fetchall()
    run["products"] = [dict(p) for p in products]
    return run


def get_run_product(conn, run_id, product_id):
    row = conn.execute(
        "SELECT * FROM run_products WHERE run_id = ? AND product_id = ?", (run_id, product_id)
    ).fetchone()
    return dict(row) if row else None
