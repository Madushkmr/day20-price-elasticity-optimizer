#!/usr/bin/env python3
"""
Flask REST API + dashboard for the Price Elasticity & Promotion ROI Optimizer.

Read endpoints (GET) are open. Write endpoints (POST) require an
`X-API-Key` header matching config/settings.yaml's api.api_key (or the
PRICING_API_KEY environment variable) — see src/auth.py.

Endpoints:
    GET  /                              dashboard
    POST /api/run                       run the pipeline, persist, return the new run
    GET  /api/runs                      list runs
    GET  /api/runs/<run_id>             full run detail
    GET  /api/runs/<run_id>/products/<product_id>   single product detail
"""
import os

import yaml
from flask import Flask, jsonify, render_template

from src import engine, db
from src.auth import require_api_key

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_PATH = os.path.join(BASE_DIR, "sample_data", "sales_history.csv")
COSTS_PATH = os.path.join(BASE_DIR, "sample_data", "product_costs.csv")
DB_PATH = os.path.join(BASE_DIR, "pricing_runs.db")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "settings.yaml")

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

app = Flask(__name__)


@app.route("/")
def dashboard():
    conn = db.get_connection(DB_PATH)
    runs = db.list_runs(conn)
    latest = db.get_run(conn, runs[0]["id"]) if runs else None
    conn.close()
    return render_template("dashboard.html", runs=runs, latest=latest)


@app.route("/api/run", methods=["POST"])
@require_api_key(CONFIG)
def api_run():
    try:
        result = engine.run_pipeline(SALES_PATH, COSTS_PATH, DB_PATH, CONFIG)
    except Exception as exc:  # noqa: BLE001 - surface pipeline errors as 400s, not 500 tracebacks
        return jsonify({"error": str(exc)}), 400
    return jsonify(result), 201


@app.route("/api/runs", methods=["GET"])
def api_list_runs():
    conn = db.get_connection(DB_PATH)
    runs = db.list_runs(conn)
    conn.close()
    return jsonify(runs)


@app.route("/api/runs/<int:run_id>", methods=["GET"])
def api_get_run(run_id):
    conn = db.get_connection(DB_PATH)
    run = db.get_run(conn, run_id)
    conn.close()
    if run is None:
        return jsonify({"error": f"run {run_id} not found"}), 404
    return jsonify(run)


@app.route("/api/runs/<int:run_id>/products/<product_id>", methods=["GET"])
def api_get_product(run_id, product_id):
    conn = db.get_connection(DB_PATH)
    row = db.get_run_product(conn, run_id, product_id)
    conn.close()
    if row is None:
        return jsonify({"error": f"product {product_id} not found in run {run_id}"}), 404
    return jsonify(row)


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
