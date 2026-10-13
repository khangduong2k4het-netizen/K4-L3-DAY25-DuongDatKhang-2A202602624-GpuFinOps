"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import pricing, report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def extension_sections(r2: dict, r3: dict) -> list[str]:
    """Markdown for the Your-Turn extensions: purchasing tiers, cache gating, reasoning budget."""
    job_rows = [
        f"| {r['job_id']} | {r['gpu_type']} | {r['legacy_tier']} | "
        f"{r['tier']}{' ' + r['term'] if r['term'] else ''} | ${r['legacy']:,} | ${r['optimized']:,} |"
        for r in r3["recommendations"]]
    purchasing = "\n".join([
        "## Purchasing tiers (Extension 1)",
        "",
        "| Job | GPU | Before | After | Before $/mo | After $/mo |",
        "|---|---|---|---|---|---|",
        *job_rows,
        "",
        f"Savings vs on-demand: {r3['legacy_savings_pct']}% before -> {r3['savings_pct']}% after "
        "(reserved now billed 24h/day for the full term; spot priced at per-GPU interruption rates).",
    ])

    write_mult = m2_inference_levers.CACHE_WRITE_MULT
    reads = m2_inference_levers.AVG_CACHE_READS
    no_cache = [t for t, n in reads.items() if not pricing.cache_is_worth_it(n, write_mult)]
    cache = "\n".join([
        "## Prompt cache gating (Extension 3)",
        "",
        f"- Break-even: {pricing.cache_break_even_reads(write_mult):.2f} reads per written prefix "
        f"(write {write_mult}x, read 0.1x)",
        f"- Cache disabled for: {', '.join(no_cache) or 'none'}",
        f"- Optimized inference: ${r2['free_cache_daily']:.2f}/day if writes were free -> "
        f"${r2['optimized_daily']:.2f}/day with write cost, ${r2['optimized_per_m']:.3f}/1M-token",
    ])

    rb, tight = r2["reasoning"], r2["reasoning_tight"]
    share = lambda g: " | ".join(f"{rb['share'][g][k]:.1%}" for k in ("requests", "tokens", "cost", "wh"))
    reasoning = "\n".join([
        "## Reasoning budget (Extension 4)",
        "",
        "| Group | Requests | Tokens | Cost | Energy |",
        "|---|---|---|---|---|",
        f"| non-reasoning | {share(0)} |",
        f"| reasoning | {share(1)} |",
        "",
        f"- Cap {rb['cap']:.0%}: {rb['downgraded']} requests downgraded, saves ${rb['cost_saved'] * DAYS:,.2f}/month "
        f"and {rb['wh_saved'] * DAYS / 1000:,.1f} kWh/month (traffic already under the cap)",
        f"- What-if cap {tight['cap']:.0%}: {tight['downgraded']} downgraded, saves ${tight['cost_saved'] * DAYS:,.2f}/month "
        f"and {tight['wh_saved'] * DAYS / 1000:,.1f} kWh/month ({tight['wh_saved'] / tight['wh_before']:.0%} of inference energy); "
        "assumes downgraded requests emit 1/6 the output on the same model",
    ])
    return [purchasing, cache, reasoning]


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    md = report.build_report(baseline, optimized, levers, sustainability=sust,
                             extra_sections=extension_sections(r2, r3))
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
