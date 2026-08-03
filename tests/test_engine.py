import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import engine, db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES_PATH = os.path.join(BASE_DIR, "sample_data", "sales_history.csv")
COSTS_PATH = os.path.join(BASE_DIR, "sample_data", "product_costs.csv")

TEST_CONFIG = {
    "optimizer": {"discount_grid": [0.0, 0.10, 0.20, 0.30], "budget_cap": 5000.0},
    "elasticity": {"bootstrap_iterations": 200, "confidence_level": 0.90, "random_seed": 42},
    "simulation": {"monte_carlo_iterations": 300, "random_seed": 7},
}


def test_end_to_end_pipeline_and_sqlite_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        result = engine.run_pipeline(SALES_PATH, COSTS_PATH, db_path, TEST_CONFIG)

        assert result["run_id"] == 1
        assert len(result["products"]) > 0
        assert "narrative" in result["summary"]

        # round-trip through SQLite
        conn = db.get_connection(db_path)
        runs = db.list_runs(conn)
        assert len(runs) == 1

        run_detail = db.get_run(conn, runs[0]["id"])
        assert run_detail is not None
        assert len(run_detail["products"]) == len(result["products"])

        first_pid = run_detail["products"][0]["product_id"]
        product_detail = db.get_run_product(conn, runs[0]["id"], first_pid)
        assert product_detail is not None
        assert product_detail["product_id"] == first_pid
        conn.close()


def test_pipeline_runs_twice_and_accumulates_runs():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        engine.run_pipeline(SALES_PATH, COSTS_PATH, db_path, TEST_CONFIG)
        engine.run_pipeline(SALES_PATH, COSTS_PATH, db_path, TEST_CONFIG)

        conn = db.get_connection(db_path)
        runs = db.list_runs(conn)
        assert len(runs) == 2
        conn.close()


def test_pipeline_budget_never_exceeded_by_more_than_one_bucket():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        result = engine.run_pipeline(SALES_PATH, COSTS_PATH, db_path, TEST_CONFIG)
        assert result["optimized"]["total_discount_cost"] <= TEST_CONFIG["optimizer"]["budget_cap"] + 60.0
