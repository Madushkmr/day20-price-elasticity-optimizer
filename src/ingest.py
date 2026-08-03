"""
Loads and validates the two input CSVs (sales history, product costs),
cross-checks them against each other, and returns clean in-memory records.
Mirrors the graceful-degradation approach used elsewhere in the series:
a product with bad/missing data is flagged and excluded rather than
crashing the whole pipeline.
"""
import csv
from collections import defaultdict


class IngestError(Exception):
    pass


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_inputs(sales_path, costs_path):
    """
    Returns (sales_by_product: dict[str, list[dict]], costs_by_product: dict[str, dict], warnings: list[str])
    """
    warnings = []

    try:
        sales_rows = _read_csv(sales_path)
    except FileNotFoundError:
        raise IngestError(f"sales history file not found: {sales_path}")
    try:
        cost_rows = _read_csv(costs_path)
    except FileNotFoundError:
        raise IngestError(f"product cost file not found: {costs_path}")

    costs_by_product = {}
    for row in cost_rows:
        pid = row.get("product_id", "").strip()
        if not pid:
            continue
        try:
            costs_by_product[pid] = {
                "product_id": pid,
                "category": row["category"],
                "base_price": float(row["base_price"]),
                "unit_cost": float(row["unit_cost"]),
            }
        except (KeyError, ValueError):
            warnings.append(f"skipped malformed cost row for product '{pid}'")

    sales_by_product = defaultdict(list)
    bad_rows = 0
    for row in sales_rows:
        pid = row.get("product_id", "").strip()
        try:
            price = float(row["price"])
            units = int(float(row["units_sold"]))
            discount = float(row.get("discount_pct", 0.0) or 0.0)
            week = int(row["week"])
            if price <= 0 or units < 0:
                bad_rows += 1
                continue
            sales_by_product[pid].append({
                "product_id": pid,
                "category": row.get("category", ""),
                "week": week,
                "price": price,
                "discount_pct": discount,
                "units_sold": units,
            })
        except (KeyError, ValueError):
            bad_rows += 1
    if bad_rows:
        warnings.append(f"skipped {bad_rows} malformed sales rows")

    # cross-source validation: a product needs both price history AND cost data
    # to go through elasticity + optimization; report and exclude otherwise
    valid_products = {}
    for pid, rows in sales_by_product.items():
        if pid not in costs_by_product:
            warnings.append(f"product '{pid}' has sales history but no cost record — excluded")
            continue
        # need meaningful price variation to estimate elasticity at all
        distinct_prices = {r["price"] for r in rows}
        if len(distinct_prices) < 2:
            warnings.append(f"product '{pid}' has no price variation in its history — excluded")
            continue
        if len(rows) < 8:
            warnings.append(f"product '{pid}' has too few observations ({len(rows)}) — excluded")
            continue
        valid_products[pid] = sorted(rows, key=lambda r: r["week"])

    final_costs = {pid: costs_by_product[pid] for pid in valid_products}

    return valid_products, final_costs, warnings
