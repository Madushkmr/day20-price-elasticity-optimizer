# Day 20 — Price Elasticity & Promotion ROI Optimizer

Day 20 of a daily AI-app series (BI focus). Given weekly price/discount/units-sold history and a unit-cost table for a product catalog, this app answers the question every merchandising/BI team gets asked before a promo calendar goes out: **"which products should actually be discounted, by how much, and will it help or hurt margin?"** — with the uncertainty in the underlying demand model carried through to the final recommendation, not hidden behind a single point estimate.

## Why this matters for BI work

Most promo-planning spreadsheets apply the same discount depth to everything ("15% off storewide") because nobody has product-level elasticity numbers on hand. That's expensive: highly elastic products are often *underd*-discounted (leaving margin-growing volume on the table) while inelastic products are *over*-discounted (giving away margin for units that would have sold anyway). This app estimates each product's own price elasticity of demand from its sales history, then solves a budget-constrained optimization problem to pick a *different* discount depth per product — and stress-tests the resulting plan against elasticity estimation error via Monte Carlo simulation before anyone commits budget to it.

Concretely, this run's sample data shows the optimizer recommending targeted discounts on 5 of 20 products (the elastic, high-margin Electronics line) for **+$8,913 projected incremental margin**, while a naive flat 15%-off-everything promotion on the same catalog projects **–$2,612** (a net loss) for *more* promotional spend. That gap — and being able to quantify it before running the promotion — is the point.

## Complexity tier: multi-technique decision-support pipeline with budget-constrained optimization, uncertainty simulation, and API authentication

This is a step up from Day 19's inventory engine in two specific ways: the optimizer is now a genuine constrained-optimization problem (multiple-choice knapsack via dynamic programming — pick one discount depth per product, respecting a shared dollar budget across the whole catalog) rather than a per-item independent policy calculation, and the REST API's write endpoint is now authenticated (`X-API-Key`), which Day 19's README explicitly flagged as a gap in the series so far.

## Architecture

```
day20-price-elasticity-optimizer/
├── app.py                    # Flask REST API (API-key auth on writes) + dashboard
├── cli.py                    # command-line interface
├── make_sample_data.py       # regenerates sample_data/*.csv (fixed seed, known ground-truth elasticities)
├── config/
│   └── settings.yaml         # discount grid, budget cap, bootstrap/simulation iterations, API key
├── src/
│   ├── ingest.py              # loads + validates sales/cost CSVs, cross-source checks, graceful degradation
│   ├── elasticity.py          # log-log OLS regression (from-scratch numpy) + residual bootstrap CI per product
│   ├── optimizer.py           # budget-constrained discount selection: multiple-choice knapsack via DP
│   ├── simulate.py            # Monte Carlo: resamples each product's elasticity from its own bootstrap
│   │                           #   distribution to get a margin-outcome distribution for the chosen plan
│   ├── narrative.py           # rule-based NLG summary (no external LLM API, runs offline)
│   ├── db.py                  # SQLite schema: runs + per-product results
│   ├── auth.py                # X-API-Key middleware for write endpoints
│   └── engine.py              # orchestrates ingest -> elasticity -> optimize -> simulate -> narrate -> persist
├── templates/
│   └── dashboard.html         # Chart.js incremental-margin-by-product chart, KPI cards, full table
├── sample_data/
│   ├── sales_history.csv      # 78 weeks x 20 products across 4 categories, price/discount/units
│   └── product_costs.csv      # base price + unit cost per product
├── tests/
│   ├── test_elasticity.py     # recovers known synthetic elasticity, CI width vs. sample size, determinism
│   ├── test_optimizer.py      # budget constraint respected, never worse than doing nothing, targets the
│   │                           #   right products
│   ├── test_simulate.py       # iteration counts, determinism, paired-comparison sanity checks
│   └── test_engine.py         # end-to-end pipeline + SQLite round trip
├── requirements.txt
└── Dockerfile
```

## The techniques, briefly

**Elasticity estimation (`src/elasticity.py`)** — fits `log(units) = alpha + beta * log(price)` via closed-form OLS (`numpy.linalg.lstsq`, no sklearn/statsmodels dependency). `beta` is the price elasticity of demand. A residual bootstrap (resample the model's own in-sample residuals thousands of times, refit each time) produces a confidence interval on `beta` and a full bootstrap distribution that `simulate.py` draws from later — the point estimate alone never drives a budget decision.

**Optimization (`src/optimizer.py`)** — for each product, builds a (cost, value) option per candidate discount depth in the configured grid (projected discount spend vs. projected incremental margin relative to no discount). Selecting exactly one option per product subject to a shared total budget is a classic **multiple-choice knapsack**, solved via dynamic programming over a discretized budget axis, with full backtracking to recover which discount was chosen per product. A flat-discount baseline (same depth applied to every product, the common real-world default) is computed the same way for comparison.

**Simulation (`src/simulate.py`)** — resamples each product's elasticity from its own bootstrap distribution (not a normal approximation) thousands of times, recomputing both the optimized plan's and the baseline's total incremental margin under each paired draw. Reports the probability the optimized plan actually beats the flat baseline once estimation uncertainty is honestly accounted for — not just at the point estimate.

**Narrative (`src/narrative.py`)** — turns the numbers into plain-English per-product and portfolio-level summaries (elasticity label, recommended action, budget utilization, simulated win probability), with any data-quality warnings surfaced inline.

## Running it

```bash
cd day20-price-elasticity-optimizer
pip install -r requirements.txt

# (optional) regenerate sample data -- already checked in with a fixed seed
python make_sample_data.py

# CLI: run the full pipeline once
python cli.py compute
python cli.py list-runs
python cli.py show-run 1
python cli.py show-product 1 P003

# Dashboard + API
python app.py   # http://localhost:5000
```

### REST API

Read endpoints are open; the write endpoint requires the `X-API-Key` header (default demo key is `day20-demo-key` in `config/settings.yaml`, override via the `PRICING_API_KEY` environment variable).

```bash
curl -X POST -H "X-API-Key: day20-demo-key" localhost:5000/api/run
curl localhost:5000/api/runs
curl localhost:5000/api/runs/1
curl localhost:5000/api/runs/1/products/P003
curl localhost:5000/api/health
```

### Tests

```bash
pytest tests/ -v
```

19 tests covering: elasticity recovery against known synthetic ground truth (with and without noise), confidence-interval width scaling with sample size, bootstrap determinism under a fixed seed, the optimizer respecting its budget cap, never recommending a plan worse than doing nothing, correctly targeting elastic/high-margin products over inelastic ones under a scarce budget, the flat-baseline comparator, Monte Carlo simulation determinism and iteration counts, paired-draw sanity (identical plans can't "beat" themselves), and an end-to-end engine run with full SQLite persistence and round-trip retrieval.

### Docker

```bash
docker build -t price-elasticity-optimizer .
docker run -p 5000:5000 price-elasticity-optimizer
```

## Sample data

`make_sample_data.py` simulates 78 weeks of weekly price/discount/units-sold history for 20 products across 4 categories (Electronics, Grocery, Apparel, HomeGoods), each generated from a **known ground-truth elasticity and margin structure** with weekly seasonality and noise, so the elasticity module's recovered estimates can be checked against the truth in tests. Electronics is deliberately tuned high-elasticity + high-margin so the optimizer has a genuine decision to make (discount Electronics, leave the rest alone) rather than a trivial uniform answer. All CSVs are checked in with a fixed random seed.

## Notes / limitations

- The elasticity model is a single-variable log-log regression per product; it doesn't account for cross-product cannibalization/halo effects, competitor pricing, or promo fatigue (declining lift from repeated discounting of the same product).
- The residual bootstrap resamples independently per observation, which understates any autocorrelation in real demand shocks — a production version would block-bootstrap.
- The optimizer's budget-cost model assumes the predicted-units forecast at each discount depth is realized exactly when computing "discount cost" (revenue given up); it does not itself carry the elasticity uncertainty into the selection step (that's what the separate Monte Carlo simulation stage is for).
- `X-API-Key` auth is a single static shared secret with no rotation, rate limiting, or per-user scoping — appropriate for a demo, not a production auth system.
- This is a demo/portfolio project over synthetic data with a fixed seed, not a production pricing system — a real version would need competitor price feeds, cannibalization modeling across substitute products, and validation of recommendations against realized outcomes after actually running a promotion.
