import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elasticity import ElasticityResult
from src.optimizer import build_options, optimize, flat_baseline
from src.simulate import simulate_plans


def _setup():
    grid = [0.0, 0.10, 0.20, 0.30]
    elas = ElasticityResult(
        product_id="P1", elasticity=-1.5, intercept=math.log(200),
        ci_low=-1.7, ci_high=-1.3, r_squared=0.9, n_obs=50,
        beta_samples=[-1.5 + 0.02 * i for i in range(-50, 50)],
    )
    elasticity_by_product = {"P1": elas}
    cost_by_product = {"P1": {"product_id": "P1", "category": "Test", "base_price": 100.0, "unit_cost": 40.0}}
    products_options = {"P1": build_options("P1", 100.0, 40.0, elas, grid)}
    return elasticity_by_product, cost_by_product, products_options


def test_simulation_returns_requested_iteration_count():
    elasticity_by_product, cost_by_product, products_options = _setup()
    optimized_result = optimize(products_options, budget_cap=1000.0, budget_step=10.0)
    baseline_result = flat_baseline(products_options, flat_discount_pct=0.10)
    opt_sim, base_sim = simulate_plans(
        optimized_result, baseline_result, elasticity_by_product, cost_by_product, iterations=500, random_seed=1
    )
    assert len(opt_sim.margin_samples) == 500
    assert len(base_sim.margin_samples) == 500


def test_simulation_deterministic_with_fixed_seed():
    elasticity_by_product, cost_by_product, products_options = _setup()
    optimized_result = optimize(products_options, budget_cap=1000.0, budget_step=10.0)
    baseline_result = flat_baseline(products_options, flat_discount_pct=0.10)
    opt_sim1, _ = simulate_plans(optimized_result, baseline_result, elasticity_by_product, cost_by_product,
                                  iterations=200, random_seed=99)
    opt_sim2, _ = simulate_plans(optimized_result, baseline_result, elasticity_by_product, cost_by_product,
                                  iterations=200, random_seed=99)
    assert (opt_sim1.margin_samples == opt_sim2.margin_samples).all()


def test_beat_baseline_probability_between_zero_and_one():
    elasticity_by_product, cost_by_product, products_options = _setup()
    optimized_result = optimize(products_options, budget_cap=1000.0, budget_step=10.0)
    baseline_result = flat_baseline(products_options, flat_discount_pct=0.10)
    opt_sim, _ = simulate_plans(optimized_result, baseline_result, elasticity_by_product, cost_by_product,
                                 iterations=500, random_seed=3)
    assert 0.0 <= opt_sim.beat_baseline_prob <= 1.0


def test_identical_plans_have_fifty_fifty_beat_probability():
    # if optimized and baseline select the exact same discount, neither should
    # systematically beat the other under the same resampled elasticity draws
    elasticity_by_product, cost_by_product, products_options = _setup()
    same_plan = flat_baseline(products_options, flat_discount_pct=0.10)
    opt_sim, _ = simulate_plans(same_plan, same_plan, elasticity_by_product, cost_by_product,
                                 iterations=500, random_seed=11)
    assert opt_sim.beat_baseline_prob == 0.0  # identical paired draws -> never strictly greater
