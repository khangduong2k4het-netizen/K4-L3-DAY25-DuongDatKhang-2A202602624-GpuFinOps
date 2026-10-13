"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30
# ASSUMPTION: how long each job's need lasts. workloads.csv only has days/month,
# so a job active every day (days >= 30) is treated as a standing service that
# will run for SERVICE_HORIZON_MONTHS; anything shorter is a one-off project and
# never justifies a commitment. See the sensitivity table for 12/24 months.
SERVICE_HORIZON_MONTHS = 36
SENSITIVITY_HORIZONS = (12, 24, 36)


def horizon_months(job: dict, service_horizon: float = SERVICE_HORIZON_MONTHS) -> float:
    return service_horizon if num(job["days"]) >= DAYS else 0.0


def legacy_cost(gpu_hours: float, hpd: float, interruptible: bool, c: dict) -> tuple[str, float]:
    """Original policy (before Extension 1): duty-cycle rule, 3yr rate on used hours only."""
    od = num(c["on_demand_hr"])
    tier = pricing.recommend_tier(hpd, interruptible)
    if tier == "spot":
        return tier, pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
    if tier == "reserved":
        return tier, gpu_hours * num(c["reserved_3yr_hr"])
    return tier, gpu_hours * od


def plan_job(j: dict, c: dict, service_horizon: float = SERVICE_HORIZON_MONTHS) -> tuple[str, str, float]:
    return pricing.best_purchasing_option(
        hours_per_day=num(j["hours_per_day"]), days=DAYS, num_gpus=int(num(j["num_gpus"])),
        prices=c, interruptible=bool(int(num(j["interruptible"]))), gpu_type=j["gpu_type"],
        horizon_months=horizon_months(j, service_horizon))


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = legacy_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        on_demand_cost = gpu_hours * num(c["on_demand_hr"])

        old_tier, old_cost = legacy_cost(gpu_hours, hpd, interruptible, c)
        tier, term, opt_cost = plan_job(j, c)

        on_demand_monthly += on_demand_cost
        legacy_monthly += old_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier, "term": term,
                     "interrupt_rate": pricing.SPOT_INTERRUPT_RATE.get(gtype, pricing.DEFAULT_INTERRUPT_RATE),
                     "legacy_tier": old_tier, "legacy": round(old_cost),
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    legacy_pct = (1 - legacy_monthly / on_demand_monthly) * 100 if on_demand_monthly else 0.0

    sensitivity = {}
    for h in SENSITIVITY_HORIZONS:
        plans = [(j["job_id"], *plan_job(j, cat[j["gpu_type"]], h)) for j in jobs]
        total = sum(p[3] for p in plans)
        sensitivity[h] = {"monthly": round(total),
                          "savings_pct": round((1 - total / on_demand_monthly) * 100, 1),
                          "tiers": {jid: f"{t}{'_' + term if term else ''}" for jid, t, term, _ in plans}}

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"service horizon assumption = {SERVICE_HORIZON_MONTHS} months (jobs active {DAYS} days/month)")
        print(f"{'job':18}{'gpu':7}{'int/h':>6}  {'before':11}{'after':15}{'on-demand':>11}{'before $':>11}{'after $':>11}")
        for r in recs:
            after = r["tier"] + (f" ({r['term']})" if r["term"] else "")
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['interrupt_rate']:>6.2f}  {r['legacy_tier']:11}{after:15}"
                  f"${r['on_demand']:>10,}${r['legacy']:>10,}${r['optimized']:>10,}")
        print(f"\nbefore (duty-cycle rule): on-demand ${on_demand_monthly:,.0f} -> ${legacy_monthly:,.0f}  ({legacy_pct:.1f}% saved)")
        print(f"monthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print("\nsensitivity to service horizon:")
        for h, s in sensitivity.items():
            standing = {k: v for k, v in s["tiers"].items() if k.startswith("job-infer")}
            print(f"  {h:>2} months: ${s['monthly']:>7,}  ({s['savings_pct']:.1f}% saved)  {standing}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "legacy_monthly": round(legacy_monthly), "legacy_savings_pct": round(legacy_pct, 1),
            "sensitivity": sensitivity}


if __name__ == "__main__":
    run()
