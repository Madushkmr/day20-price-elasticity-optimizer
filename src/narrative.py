"""
Rule-based natural-language summary generation (no external LLM API — runs
fully offline). Turns the numeric outputs of elasticity estimation,
optimization, and simulation into short plain-English paragraphs, per
product and at the portfolio level, so a BI stakeholder can read the
recommendation without digging through JSON.
"""


def _elasticity_label(beta):
    abs_beta = abs(beta)
    if abs_beta >= 1.5:
        return "highly price-elastic"
    if abs_beta >= 1.0:
        return "elastic"
    if abs_beta >= 0.5:
        return "moderately inelastic"
    return "highly inelastic"


def product_narrative(product_id, elasticity_result, chosen_option, category=None):
    label = _elasticity_label(elasticity_result.elasticity)
    cat_phrase = f" ({category})" if category else ""
    lines = [
        f"{product_id}{cat_phrase}: estimated price elasticity is {elasticity_result.elasticity:.2f} "
        f"(90% CI [{elasticity_result.ci_low:.2f}, {elasticity_result.ci_high:.2f}], R²={elasticity_result.r_squared:.2f}) "
        f"— demand is {label}."
    ]
    if chosen_option.discount_pct > 0:
        lines.append(
            f"Recommended promotion: {chosen_option.discount_pct:.0%} off (price ${chosen_option.price:.2f}), "
            f"projected {chosen_option.predicted_units:.0f} units, incremental margin "
            f"${chosen_option.incremental_margin:,.2f} versus no discount, at a promotional cost of "
            f"${chosen_option.discount_cost:,.2f}."
        )
    else:
        lines.append(
            "Recommended action: no discount — the projected margin impact of any tested discount "
            "depth was flat or negative for this product given its elasticity."
        )
    return " ".join(lines)


def portfolio_narrative(optimizer_result, baseline_result, optimized_sim, baseline_sim, warnings):
    lines = []
    lines.append(
        f"Optimized plan: {len(optimizer_result.selections)} products, "
        f"${optimizer_result.total_incremental_margin:,.2f} projected incremental margin "
        f"using ${optimizer_result.total_discount_cost:,.2f} of a ${optimizer_result.budget_cap:,.2f} budget "
        f"({optimizer_result.to_dict()['budget_utilization_pct']}% utilized)."
    )
    lines.append(
        f"Flat-discount baseline (same depth applied to every product): "
        f"${baseline_result.total_incremental_margin:,.2f} projected incremental margin, "
        f"${baseline_result.total_discount_cost:,.2f} of promotional spend."
    )
    delta = optimizer_result.total_incremental_margin - baseline_result.total_incremental_margin
    direction = "more" if delta >= 0 else "less"
    lines.append(
        f"The targeted plan projects ${abs(delta):,.2f} {direction} incremental margin than a one-size-fits-all "
        f"discount, for {'less or equal' if optimizer_result.total_discount_cost <= baseline_result.total_discount_cost else 'more'} promotional spend."
    )
    if optimized_sim is not None:
        d = optimized_sim.to_dict()
        lines.append(
            f"Under Monte Carlo simulation of elasticity estimation uncertainty ({len(optimized_sim.margin_samples)} draws), "
            f"the optimized plan has a {d['prob_beats_baseline']:.0%} probability of beating the flat-discount baseline "
            f"and a {d['prob_positive_margin']:.0%} probability of producing positive incremental margin at all "
            f"(median ${d['median_margin']:,.2f}, 10th-90th percentile range ${d['p10_margin']:,.2f} to ${d['p90_margin']:,.2f})."
        )
    if warnings:
        lines.append(f"Data-quality notes: {'; '.join(warnings)}.")
    return " ".join(lines)
