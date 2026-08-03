"""
Price elasticity of demand estimation, from scratch (no sklearn/statsmodels
dependency — plain numpy linear algebra), via log-log OLS regression:

    log(units_sold) = alpha + beta * log(price)

beta is interpreted as the price elasticity of demand: the percent change
in quantity demanded for a 1% change in price. A residual bootstrap gives
a confidence interval on beta, which downstream modules (optimizer,
simulate) use to reason about estimation uncertainty rather than treating
the point estimate as exact.
"""
import numpy as np


class ElasticityResult:
    def __init__(self, product_id, elasticity, intercept, ci_low, ci_high, r_squared, n_obs, beta_samples=None):
        self.product_id = product_id
        self.elasticity = elasticity
        self.intercept = intercept
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.r_squared = r_squared
        self.n_obs = n_obs
        # full bootstrap distribution of beta (elasticity), used by src/simulate.py
        # to propagate estimation uncertainty into the Monte Carlo margin simulation
        self.beta_samples = beta_samples if beta_samples is not None else [elasticity]

    def to_dict(self):
        return {
            "product_id": self.product_id,
            "elasticity": round(self.elasticity, 4),
            "intercept": round(self.intercept, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "r_squared": round(self.r_squared, 4),
            "n_obs": self.n_obs,
        }

    def demand_at_price(self, price):
        """Predicted units sold at a given price, from the fitted log-log model."""
        log_units = self.intercept + self.elasticity * np.log(price)
        return float(np.exp(log_units))


def _ols_log_log(prices, units):
    """
    Fits log(units) = alpha + beta*log(price) via closed-form OLS.
    Returns (alpha, beta, r_squared, fitted, residuals).
    """
    x = np.log(np.asarray(prices, dtype=float))
    y = np.log(np.maximum(np.asarray(units, dtype=float), 1e-6))  # avoid log(0)

    X = np.column_stack([np.ones_like(x), x])
    # normal equations: beta_hat = (X'X)^-1 X'y
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coeffs

    fitted = X @ coeffs
    residuals = y - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(alpha), float(beta), r_squared, fitted, residuals


def estimate_elasticity(product_id, sales_rows, bootstrap_iterations=2000, confidence_level=0.90, random_seed=42):
    """
    sales_rows: list of dicts with 'price' and 'units_sold' keys (one product's history).
    Returns an ElasticityResult with a bootstrap confidence interval on beta.
    """
    prices = [r["price"] for r in sales_rows]
    units = [r["units_sold"] for r in sales_rows]

    alpha, beta, r_squared, fitted, residuals = _ols_log_log(prices, units)

    # residual bootstrap: resample residuals with replacement, refit on
    # synthetic y* = fitted + resampled residual, recompute beta each time
    rng = np.random.default_rng(random_seed)
    x = np.log(np.asarray(prices, dtype=float))
    X = np.column_stack([np.ones_like(x), x])
    n = len(residuals)

    betas = np.empty(bootstrap_iterations)
    for i in range(bootstrap_iterations):
        resampled_resid = rng.choice(residuals, size=n, replace=True)
        y_star = fitted + resampled_resid
        coeffs_star, *_ = np.linalg.lstsq(X, y_star, rcond=None)
        betas[i] = coeffs_star[1]

    alpha_tail = (1.0 - confidence_level) / 2.0
    ci_low = float(np.quantile(betas, alpha_tail))
    ci_high = float(np.quantile(betas, 1.0 - alpha_tail))

    return ElasticityResult(
        product_id=product_id,
        elasticity=beta,
        intercept=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        r_squared=r_squared,
        n_obs=len(sales_rows),
        beta_samples=betas.tolist(),
    )


def estimate_all(sales_by_product, bootstrap_iterations=2000, confidence_level=0.90, random_seed=42):
    """Returns dict[product_id, ElasticityResult]."""
    results = {}
    for pid, rows in sales_by_product.items():
        results[pid] = estimate_elasticity(
            pid, rows,
            bootstrap_iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
    return results
