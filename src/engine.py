"""
Orchestrates the full pipeline:
  ingest -> estimate elasticity -> build discount options -> optimize (budget-
  constrained) -> compute flat baseline -> Monte Carlo simulate both plans ->
  generate narrative -> persist to SQLite.

This is the single entry point both cli.py and app.py call so the CLI and
the web API always run the exact same logic.
"""
import os

from . import ingest, elasticity, optimizer, simulate, narrative, db


def run_pipeline(sales_path, costs_path, db_path, config):
    sales_by_product, costs_by_product, warnings = ingest.load_inputs(sales_path, costs_path)

    if not sales_by_product:
        raise RuntimeError("no valid products to price after ingestion — check input CSVs")

    elasticity_cfg = config.get("elasticity", {})
    elasticity_by_product = elasticity.estimate_all(
        sales_by_product,
        bootstrap_iterations=elasticity_cfg.get("bootstrap_iterations", 2000),
        confidence_level=elasticity_cfg.get("confidence_level", 0.90),
        random_seed=elasticity_cfg.get("random_seed", 42),
    )

    opt_cfg = config.get("optimizer", {})
    discount_grid = opt_cfg.get("discount_grid", [0.0, 0.1, 0.2])
    budget_cap = opt_cfg.get("budget_cap", 10000.0)

    products_options = {}
    for pid, elas in elasticity_by_product.items():
        cost = costs_by_product[pid]
        products_options[pid] = optimizer.build_options(
            pid, cost["base_price"], cost["unit_cost"], elas, discount_grid
        )

    optimized_result = optimizer.optimize(products_options, budget_cap)
    baseline_result = optimizer.flat_baseline(products_options, flat_discount_pct=0.15)

    sim_cfg = config.get("simulation", {})
    optimized_sim, baseline_sim = simulate.simulate_plans(
        optimized_result, baseline_result, elasticity_by_product, costs_by_product,
        iterations=sim_cfg.get("monte_carlo_iterations", 3000),
        random_seed=sim_cfg.get("random_seed", 7),
    )

    # per-product narratives
    product_rows = []
    for pid, opt in optimized_result.selections.items():
        elas = elasticity_by_product[pid]
        cat = costs_by_product[pid].get("category")
        text = narrative.product_narrative(pid, elas, opt, category=cat)
        product_rows.append({
            "product_id": pid,
            "category": cat,
            "elasticity": elas.elasticity,
            "ci_low": elas.ci_low,
            "ci_high": elas.ci_high,
            "r_squared": elas.r_squared,
            "discount_pct": opt.discount_pct,
            "price": opt.price,
            "predicted_units": opt.predicted_units,
            "incremental_margin": opt.incremental_margin,
            "discount_cost": opt.discount_cost,
            "narrative": text,
        })

    portfolio_text = narrative.portfolio_narrative(
        optimized_result, baseline_result, optimized_sim, baseline_sim, warnings
    )

    opt_sim_dict = optimized_sim.to_dict()
    summary = {
        "budget_cap": budget_cap,
        "total_incremental_margin": optimized_result.total_incremental_margin,
        "total_discount_cost": optimized_result.total_discount_cost,
        "baseline_incremental_margin": baseline_result.total_incremental_margin,
        "baseline_discount_cost": baseline_result.total_discount_cost,
        "prob_beats_baseline": opt_sim_dict["prob_beats_baseline"],
        "prob_positive_margin": opt_sim_dict["prob_positive_margin"],
        "narrative": portfolio_text,
        "warnings": warnings,
    }

    conn = db.get_connection(db_path)
    try:
        run_id = db.save_run(conn, summary, product_rows)
    finally:
        conn.close()

    return {
        "run_id": run_id,
        "summary": summary,
        "optimized": optimized_result.to_dict(),
        "baseline": baseline_result.to_dict(),
        "simulation": {"optimized": opt_sim_dict, "baseline": baseline_sim.to_dict()},
        "products": product_rows,
        "warnings": warnings,
    }
