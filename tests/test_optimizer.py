import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elasticity import ElasticityResult
from src.optimizer import build_options, optimize, flat_baseline


def _fake_elasticity(product_id, elasticity, intercept):
    return ElasticityResult(
        product_id=product_id, elasticity=elasticity, intercept=intercept,
        ci_low=elasticity - 0.1, ci_high=elasticity + 0.1, r_squared=0.9, n_obs=50,
        beta_samples=[elasticity] * 50,
    )


def _sample_products_options():
    import math
    grid = [0.0, 0.10, 0.20, 0.30]
    products_options = {}
    # elastic product: discounting should help margin
    elas1 = _fake_elasticity("ELASTIC", elasticity=-2.5, intercept=math.log(200))
    products_options["ELASTIC"] = build_options("ELASTIC", base_price=100.0, unit_cost=40.0,
                                                  elasticity_result=elas1, discount_grid=grid)
    # inelastic product: discounting should hurt margin
    elas2 = _fake_elasticity("INELASTIC", elasticity=-0.3, intercept=math.log(150))
    products_options["INELASTIC"] = build_options("INELASTIC", base_price=50.0, unit_cost=30.0,
                                                    elasticity_result=elas2, discount_grid=grid)
    return products_options


def test_zero_discount_option_always_present_and_free():
    products_options = _sample_products_options()
    for pid, options in products_options.items():
        zero_opts = [o for o in options if o.discount_pct == 0.0]
        assert len(zero_opts) == 1
        assert zero_opts[0].incremental_margin == 0.0
        assert zero_opts[0].discount_cost == 0.0


def test_optimizer_respects_budget_cap():
    products_options = _sample_products_options()
    result = optimize(products_options, budget_cap=50.0, budget_step=10.0)
    # allow one bucket's worth of rounding slack
    assert result.total_discount_cost <= 50.0 + 10.0


def test_optimizer_never_does_worse_than_doing_nothing():
    products_options = _sample_products_options()
    result = optimize(products_options, budget_cap=5000.0, budget_step=25.0)
    assert result.total_incremental_margin >= -1e-6


def test_optimizer_favors_discounting_elastic_product_with_ample_budget():
    products_options = _sample_products_options()
    result = optimize(products_options, budget_cap=5000.0, budget_step=25.0)
    assert result.selections["ELASTIC"].discount_pct > 0.0


def test_optimizer_avoids_discounting_inelastic_product_when_budget_scarce():
    # with a tiny budget, the optimizer should preferentially spend it on the
    # product where discounting actually pays off
    products_options = _sample_products_options()
    result = optimize(products_options, budget_cap=200.0, budget_step=10.0)
    # inelastic product's discounted options should lose margin, so it should stay at 0%
    assert result.selections["INELASTIC"].discount_pct == 0.0


def test_flat_baseline_applies_same_discount_to_all():
    products_options = _sample_products_options()
    result = flat_baseline(products_options, flat_discount_pct=0.20)
    for opt in result.selections.values():
        assert opt.discount_pct == 0.20
