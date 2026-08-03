import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elasticity import estimate_elasticity, estimate_all


def _synthetic_rows(true_elasticity, base_price=100.0, base_units=200.0, n=60, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        discount = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        price = base_price * (1 - discount)
        log_units = math.log(base_units) + true_elasticity * math.log(price / base_price) + rng.normal(0, noise)
        units = max(1, round(math.exp(log_units)))
        rows.append({"price": price, "units_sold": units})
    return rows


def test_recovers_known_elasticity_no_noise():
    rows = _synthetic_rows(true_elasticity=-1.5, noise=0.0)
    result = estimate_elasticity("P001", rows, bootstrap_iterations=200)
    assert abs(result.elasticity - (-1.5)) < 0.05
    assert result.r_squared > 0.99


def test_recovers_approx_elasticity_with_noise():
    rows = _synthetic_rows(true_elasticity=-0.8, noise=0.08, seed=3)
    result = estimate_elasticity("P002", rows, bootstrap_iterations=500)
    assert abs(result.elasticity - (-0.8)) < 0.25
    # confidence interval should actually bracket the true value most of the time
    assert result.ci_low < result.elasticity < result.ci_high


def test_confidence_interval_widens_with_fewer_observations():
    rows_many = _synthetic_rows(true_elasticity=-1.2, noise=0.1, n=200, seed=5)
    rows_few = _synthetic_rows(true_elasticity=-1.2, noise=0.1, n=15, seed=5)
    result_many = estimate_elasticity("A", rows_many, bootstrap_iterations=500, random_seed=1)
    result_few = estimate_elasticity("B", rows_few, bootstrap_iterations=500, random_seed=1)
    width_many = result_many.ci_high - result_many.ci_low
    width_few = result_few.ci_high - result_few.ci_low
    assert width_few > width_many


def test_deterministic_with_fixed_seed():
    rows = _synthetic_rows(true_elasticity=-1.0, noise=0.1, seed=9)
    r1 = estimate_elasticity("X", rows, bootstrap_iterations=300, random_seed=42)
    r2 = estimate_elasticity("X", rows, bootstrap_iterations=300, random_seed=42)
    assert r1.elasticity == r2.elasticity
    assert r1.ci_low == r2.ci_low and r1.ci_high == r2.ci_high


def test_demand_at_price_matches_fitted_curve():
    rows = _synthetic_rows(true_elasticity=-1.3, noise=0.0)
    result = estimate_elasticity("P003", rows, bootstrap_iterations=100)
    predicted = result.demand_at_price(100.0)
    actual_at_100 = [r["units_sold"] for r in rows if abs(r["price"] - 100.0) < 1e-6]
    assert abs(predicted - actual_at_100[0]) < actual_at_100[0] * 0.05


def test_estimate_all_returns_one_result_per_product():
    products = {
        "P1": _synthetic_rows(-1.0, seed=1),
        "P2": _synthetic_rows(-0.5, seed=2),
    }
    results = estimate_all(products, bootstrap_iterations=100)
    assert set(results.keys()) == {"P1", "P2"}
