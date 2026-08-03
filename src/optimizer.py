"""
Budget-constrained promotion optimizer.

For each product, the elasticity model predicts demand at every candidate
discount depth in the configured grid. Applying a discount changes both
units sold (via the elasticity curve) and margin per unit — deeper
discounts can still raise *total* margin for elastic products, or destroy
it for inelastic ones. The optimizer picks exactly one discount depth per
product to maximize total incremental margin (versus the no-discount
baseline) subject to a total promotional budget cap.

This is a "multiple-choice knapsack": each product is a group, each
discount depth in that group is an item with a (cost, value) pair, and we
must pick at most one item per group without exceeding the shared budget.
Solved via dynamic programming over a discretized budget axis.
"""
from dataclasses import dataclass, field


@dataclass
class ProductOption:
    product_id: str
    discount_pct: float
    price: float
    predicted_units: float
    incremental_margin: float   # margin_at_discount - margin_at_zero_discount
    discount_cost: float        # revenue given up vs. full price, at predicted volume


@dataclass
class OptimizerResult:
    budget_cap: float
    total_incremental_margin: float
    total_discount_cost: float
    selections: dict = field(default_factory=dict)  # product_id -> ProductOption

    def to_dict(self):
        return {
            "budget_cap": round(self.budget_cap, 2),
            "total_incremental_margin": round(self.total_incremental_margin, 2),
            "total_discount_cost": round(self.total_discount_cost, 2),
            "budget_utilization_pct": round(100 * self.total_discount_cost / self.budget_cap, 1) if self.budget_cap > 0 else 0.0,
            "selections": {
                pid: {
                    "discount_pct": opt.discount_pct,
                    "price": round(opt.price, 2),
                    "predicted_units": round(opt.predicted_units, 1),
                    "incremental_margin": round(opt.incremental_margin, 2),
                    "discount_cost": round(opt.discount_cost, 2),
                }
                for pid, opt in self.selections.items()
            },
        }


def build_options(product_id, base_price, unit_cost, elasticity_result, discount_grid):
    """Computes the (cost, value) option for each candidate discount depth."""
    baseline_units = elasticity_result.demand_at_price(base_price)
    baseline_margin = baseline_units * (base_price - unit_cost)

    options = []
    for d in discount_grid:
        price = round(base_price * (1 - d), 4)
        units = elasticity_result.demand_at_price(price)
        margin = units * (price - unit_cost)
        incremental_margin = margin - baseline_margin
        discount_cost = max(0.0, units * (base_price - price))
        options.append(ProductOption(
            product_id=product_id,
            discount_pct=d,
            price=price,
            predicted_units=units,
            incremental_margin=incremental_margin,
            discount_cost=discount_cost,
        ))
    return options


def _discretize(value, step):
    return int(round(value / step))


def optimize(products_options, budget_cap, budget_step=50.0):
    """
    products_options: dict[product_id, list[ProductOption]] (one list per product,
        must include the zero-discount option so "select nothing" is always feasible).
    budget_cap: total allowed discount_cost across all selected options.

    Returns OptimizerResult. Uses multiple-choice knapsack DP discretized to
    `budget_step` dollar buckets for tractability.
    """
    product_ids = list(products_options.keys())
    n_buckets = max(1, _discretize(budget_cap, budget_step)) + 1
    return _optimize_with_backtrace(product_ids, products_options, n_buckets, budget_step, budget_cap)


def _optimize_with_backtrace(product_ids, products_options, n_buckets, budget_step, budget_cap):
    NEG_INF = float("-inf")
    stages = [[0.0] * n_buckets]  # stages[0] = before any product considered
    picks = []  # picks[i][k] = option index chosen for product i to land on bucket k of stages[i+1]

    dp = [0.0] * n_buckets
    for pid in product_ids:
        options = products_options[pid]
        new_dp = [NEG_INF] * n_buckets
        chosen = [-1] * n_buckets
        for k in range(n_buckets):
            if dp[k] == NEG_INF:
                continue
            for oi, opt in enumerate(options):
                cost_bucket = _discretize(opt.discount_cost, budget_step)
                nk = k + cost_bucket
                if nk >= n_buckets:
                    continue
                candidate = dp[k] + opt.incremental_margin
                if candidate > new_dp[nk]:
                    new_dp[nk] = candidate
                    chosen[nk] = (k, oi)
        stages.append(new_dp)
        picks.append(chosen)
        dp = new_dp

    best_k = max(range(n_buckets), key=lambda k: dp[k])
    if dp[best_k] == NEG_INF:
        best_k = 0

    # backtrack through picks
    selections = {}
    k = best_k
    for i in range(len(product_ids) - 1, -1, -1):
        pid = product_ids[i]
        entry = picks[i][k]
        if entry is None or entry == -1:
            # no product selected at all reached this bucket (shouldn't happen at k=0 start)
            k_prev, oi = 0, 0
        else:
            k_prev, oi = entry
        options = products_options[pid]
        selections[pid] = options[oi]
        k = k_prev

    total_margin = sum(o.incremental_margin for o in selections.values())
    total_cost = sum(o.discount_cost for o in selections.values())

    return OptimizerResult(
        budget_cap=budget_cap,
        total_incremental_margin=total_margin,
        total_discount_cost=total_cost,
        selections=selections,
    )


def flat_baseline(products_options, flat_discount_pct=0.15):
    """
    Naive baseline: apply the same fixed discount to every product regardless
    of its elasticity, for comparison against the optimizer's targeted plan.
    """
    selections = {}
    for pid, options in products_options.items():
        match = min(options, key=lambda o: abs(o.discount_pct - flat_discount_pct))
        selections[pid] = match
    total_margin = sum(o.incremental_margin for o in selections.values())
    total_cost = sum(o.discount_cost for o in selections.values())
    return OptimizerResult(
        budget_cap=float("nan"),
        total_incremental_margin=total_margin,
        total_discount_cost=total_cost,
        selections=selections,
    )
