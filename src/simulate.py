"""
Monte Carlo simulation of the optimized promotion plan's margin outcome
under elasticity estimation uncertainty.

The optimizer (src/optimizer.py) picks discount depths using the *point
estimate* of each product's elasticity. But that estimate comes from a
regression with a confidence interval, not a certainty. This module
resamples each product's elasticity from its own bootstrap distribution
(computed in src/elasticity.py) thousands of times, recomputes the
selected plan's incremental margin under each resampled elasticity, and
reports the resulting distribution — including the probability that the
optimized plan actually beats the flat-discount baseline once uncertainty
is accounted for, not just at the point estimate.
"""
import numpy as np


class SimulationResult:
    def __init__(self, plan_name, margin_samples, beat_baseline_prob=None):
        self.plan_name = plan_name
        self.margin_samples = np.asarray(margin_samples, dtype=float)
        self.beat_baseline_prob = beat_baseline_prob

    def to_dict(self):
        s = self.margin_samples
        return {
            "plan_name": self.plan_name,
            "mean_margin": round(float(np.mean(s)), 2),
            "p10_margin": round(float(np.quantile(s, 0.10)), 2),
            "median_margin": round(float(np.quantile(s, 0.50)), 2),
            "p90_margin": round(float(np.quantile(s, 0.90)), 2),
            "std_margin": round(float(np.std(s)), 2),
            "prob_positive_margin": round(float(np.mean(s > 0)), 4),
            "prob_beats_baseline": round(self.beat_baseline_prob, 4) if self.beat_baseline_prob is not None else None,
        }


def _simulate_plan_margin(selections, elasticity_by_product, cost_by_product, iterations, rng):
    """
    selections: dict[product_id, ProductOption] (the chosen discount per product)
    Returns an array of length `iterations`: total incremental margin per draw,
    resampling each product's elasticity independently from its bootstrap distribution.
    """
    product_ids = list(selections.keys())
    # pre-fetch bootstrap arrays once
    beta_pools = {pid: np.asarray(elasticity_by_product[pid].beta_samples) for pid in product_ids}
    intercepts = {pid: elasticity_by_product[pid].intercept for pid in product_ids}

    totals = np.zeros(iterations)
    for pid in product_ids:
        opt = selections[pid]
        cost = cost_by_product[pid]
        base_price = cost["base_price"]
        unit_cost = cost["unit_cost"]
        alpha = intercepts[pid]
        pool = beta_pools[pid]

        drawn_betas = rng.choice(pool, size=iterations, replace=True)

        baseline_units = np.exp(alpha + drawn_betas * np.log(base_price))
        baseline_margin = baseline_units * (base_price - unit_cost)

        promo_units = np.exp(alpha + drawn_betas * np.log(max(opt.price, 1e-6)))
        promo_margin = promo_units * (opt.price - unit_cost)

        totals += (promo_margin - baseline_margin)

    return totals


def simulate_plans(optimized_result, baseline_result, elasticity_by_product, cost_by_product,
                    iterations=3000, random_seed=7):
    """
    Runs the Monte Carlo comparison for the optimizer's plan vs. the flat-discount
    baseline plan, both drawn from the SAME resampled elasticity draws per iteration
    (paired simulation) so `beats_baseline` reflects a fair per-draw comparison.
    """
    rng = np.random.default_rng(random_seed)

    optimized_margins = _simulate_plan_margin(
        optimized_result.selections, elasticity_by_product, cost_by_product, iterations, rng
    )
    rng2 = np.random.default_rng(random_seed)  # same seed -> same elasticity draws per product for paired comparison
    baseline_margins = _simulate_plan_margin(
        baseline_result.selections, elasticity_by_product, cost_by_product, iterations, rng2
    )

    beat_prob = float(np.mean(optimized_margins > baseline_margins))

    optimized_sim = SimulationResult("optimized", optimized_margins, beat_baseline_prob=beat_prob)
    baseline_sim = SimulationResult("flat_baseline", baseline_margins, beat_baseline_prob=None)

    return optimized_sim, baseline_sim
