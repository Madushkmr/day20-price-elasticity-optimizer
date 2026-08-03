"""
Regenerates sample_data/sales_history.csv and sample_data/product_costs.csv
with a fixed random seed so the checked-in files are reproducible.

Simulates 18 months of weekly price/discount/units-sold observations for 20
products across 4 categories, each with a *known* underlying price elasticity
so the elasticity-estimation module (src/elasticity.py) can be validated
against ground truth in tests.
"""
import csv
import random
import math
import os

SEED = 42
N_WEEKS = 78  # ~18 months of weekly observations
OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

CATEGORIES = {
    # margin_range controls unit_cost = base_price * uniform(*margin_range) — a LOWER
    # unit-cost fraction means a HIGHER margin ratio, which lowers the elasticity
    # threshold (~ -1/margin_fraction) beyond which a discount actually grows total
    # margin rather than just revenue. Electronics is tuned high-elasticity +
    # high-margin so it clearly clears that threshold; the other categories are
    # tuned to clearly NOT clear it, giving the optimizer a real decision to make
    # (discount Electronics, leave the rest alone) instead of a uniform answer.
    "Electronics": {"base_price": 120.0, "elasticity_range": (-3.8, -2.6), "margin_range": (0.28, 0.40)},
    "Grocery":     {"base_price": 6.5,   "elasticity_range": (-0.6, -0.3), "margin_range": (0.45, 0.60)},
    "Apparel":     {"base_price": 45.0,  "elasticity_range": (-1.6, -1.0), "margin_range": (0.45, 0.60)},
    "HomeGoods":   {"base_price": 30.0,  "elasticity_range": (-1.1, -0.7), "margin_range": (0.45, 0.60)},
}

PRODUCTS_PER_CATEGORY = 5


def gen_products(rng):
    products = []
    pid = 1
    for cat, spec in CATEGORIES.items():
        for _ in range(PRODUCTS_PER_CATEGORY):
            elasticity = rng.uniform(*spec["elasticity_range"])
            base_price = spec["base_price"] * rng.uniform(0.85, 1.2)
            unit_cost = base_price * rng.uniform(*spec["margin_range"])
            base_weekly_units = rng.uniform(80, 400) if cat != "Electronics" else rng.uniform(20, 90)
            products.append({
                "product_id": f"P{pid:03d}",
                "category": cat,
                "base_price": round(base_price, 2),
                "unit_cost": round(unit_cost, 2),
                "true_elasticity": round(elasticity, 3),  # ground truth, NOT written to the app's input CSVs
                "base_weekly_units": base_weekly_units,
            })
            pid += 1
    return products


def gen_sales_history(products, rng):
    rows = []
    for p in products:
        # log-log demand model with weekly seasonality + noise:
        # log(units) = log(base_units) + elasticity * log(price/base_price) + season + noise
        for week in range(N_WEEKS):
            # promotions happen on ~25% of weeks, discount depth 0-30%
            on_promo = rng.random() < 0.25
            discount = round(rng.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]), 2) if on_promo else 0.0
            price = round(p["base_price"] * (1 - discount), 2)

            season = 0.15 * math.sin(2 * math.pi * week / 52.0)
            noise = rng.gauss(0, 0.12)
            log_units = (
                math.log(p["base_weekly_units"])
                + p["true_elasticity"] * math.log(price / p["base_price"])
                + season
                + noise
            )
            units = max(0, round(math.exp(log_units)))

            rows.append({
                "product_id": p["product_id"],
                "category": p["category"],
                "week": week + 1,
                "price": price,
                "discount_pct": discount,
                "units_sold": units,
            })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    products = gen_products(rng)
    sales = gen_sales_history(products, rng)

    with open(os.path.join(OUT_DIR, "sales_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["product_id", "category", "week", "price", "discount_pct", "units_sold"])
        w.writeheader()
        w.writerows(sales)

    with open(os.path.join(OUT_DIR, "product_costs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["product_id", "category", "base_price", "unit_cost"])
        w.writeheader()
        for p in products:
            w.writerow({
                "product_id": p["product_id"],
                "category": p["category"],
                "base_price": p["base_price"],
                "unit_cost": p["unit_cost"],
            })

    print(f"Wrote {len(sales)} sales rows and {len(products)} products to {OUT_DIR}")


if __name__ == "__main__":
    main()
