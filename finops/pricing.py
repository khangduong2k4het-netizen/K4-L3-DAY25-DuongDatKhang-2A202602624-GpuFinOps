"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations

import math


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


# Anthropic prompt caching: write = 1.25x base input (5-min TTL) or 2x (1-hour
# TTL); read = 0.1x. Other providers differ (e.g. Gemini bills storage per hour).
CACHE_WRITE_MULT_5M = 1.25
CACHE_WRITE_MULT_1H = 2.0


def cache_break_even_reads(write_mult: float = CACHE_WRITE_MULT_5M, read_discount: float = 0.10) -> float:
    """Reads per written prefix at which caching costs the same as not caching.

    One prefix, written once then read R times (prices relative to base input):
      cached   = write_mult + R * read_discount
      uncached = 1 + R
    Equal at R* = (write_mult - 1) / (1 - read_discount).  1.25x/0.1x -> R* ~ 0.28.
    """
    if read_discount >= 1.0:
        return float("inf")
    return max(0.0, write_mult - 1.0) / (1.0 - read_discount)


def cache_cost_multiplier(avg_cache_reads: float, write_mult: float = CACHE_WRITE_MULT_5M,
                          read_discount: float = 0.10) -> float:
    """Effective price multiplier on cacheable input tokens, write cost included.

    Averaged over the 1 write + R reads that share a prefix:
    (write_mult + R * read_discount) / (R + 1).  -> read_discount as R -> inf.
    """
    r = max(0.0, avg_cache_reads)
    return (write_mult + r * read_discount) / (r + 1.0)


def cache_is_worth_it(avg_cache_reads: float, write_mult: float = CACHE_WRITE_MULT_5M,
                      read_discount: float = 0.10) -> bool:
    """Cache only pays when read savings outweigh the write premium (R > R*)."""
    return cache_cost_multiplier(avg_cache_reads, write_mult, read_discount) < 1.0


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Per-hour spot interruption probability by GPU type. ASSUMPTION (no live feed in
# the lab data): deck gives H100 spot "<5%"; newest/largest SKUs sit in deep
# neocloud pools, while A10G/L4 share capacity with popular general-purpose
# instance families and get reclaimed more often (cf. AWS Spot Instance Advisor
# frequency buckets). Re-baseline from your provider's spot history before acting.
SPOT_INTERRUPT_RATE = {
    "H100": 0.03, "H200": 0.03, "B200": 0.04,
    "A100": 0.05, "MI300X": 0.05,
    "L4": 0.08, "A10G": 0.10,
}
DEFAULT_INTERRUPT_RATE = 0.05
MAX_SPOT_INTERRUPT_RATE = 0.15   # above this, rework makes spot operationally painful
RESERVED_TERMS = {"1yr": (12, "reserved_1yr_hr"), "3yr": (36, "reserved_3yr_hr")}


def reserved_waste_factor(horizon_months: float, term_months: int) -> float:
    """Months paid / months needed when a commitment outlives the need.

    A 3yr term bought for a 24-month need pays 36/24 = 1.5x; 1yr terms renew in
    whole years, so an 18-month need pays 24/18.
    """
    if horizon_months <= 0:
        return float("inf")
    terms = math.ceil(horizon_months / term_months)
    return terms * term_months / horizon_months


def purchasing_options(
    hours_per_day: float,
    days: float,
    num_gpus: int,
    prices: dict,
    interruptible: bool,
    gpu_type: str | None = None,
    horizon_months: float = 0.0,
) -> dict:
    """Monthly cost of every eligible way to buy this job's GPU-hours.

    Keys: 'on_demand', 'spot', 'reserved_1yr', 'reserved_3yr'.
      - spot: only if interruptible and the GPU's interrupt rate is tolerable;
        priced with the checkpoint/rework simulation at that GPU's rate.
      - reserved: a commitment bills 24h/day whether used or not, and the whole
        term even if the need ends earlier (see reserved_waste_factor).
    """
    od_hr = float(prices["on_demand_hr"])
    used_hours = max(0.0, hours_per_day) * days * num_gpus
    opts = {"on_demand": used_hours * od_hr}

    rate = SPOT_INTERRUPT_RATE.get(gpu_type, DEFAULT_INTERRUPT_RATE)
    if interruptible and rate <= MAX_SPOT_INTERRUPT_RATE:
        sim = spot_checkpoint_cost(used_hours, float(prices["spot_hr"]), od_hr, interrupt_rate=rate)
        opts["spot"] = sim["spot_cost"]

    if horizon_months > 0:
        committed_hours = 24.0 * days * num_gpus
        for term, (months, col) in RESERVED_TERMS.items():
            opts[f"reserved_{term}"] = (committed_hours * float(prices[col])
                                        * reserved_waste_factor(horizon_months, months))
    return opts


def best_purchasing_option(**kwargs) -> tuple[str, str, float]:
    """(tier, term, monthly_cost) of the cheapest option; on_demand wins ties."""
    opts = purchasing_options(**kwargs)
    best = min(opts, key=lambda k: (round(opts[k], 6), k != "on_demand"))
    tier, _, term = best.partition("_") if best.startswith("reserved") else (best, "", "")
    return tier, term, opts[best]


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    prices: dict | None = None,
    days: float = 30,
    num_gpus: int = 1,
    horizon_months: float = 0.0,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    With `prices` (a price_catalog row): cost-based — cheapest of purchasing_options,
    which accounts for per-GPU spot interruption rates and 1yr/3yr commitments
    over `horizon_months`. Use best_purchasing_option() to also get the term.

    Without `prices`, the DOCUMENTED simple policy:
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    if prices is not None:
        tier, _, _ = best_purchasing_option(
            hours_per_day=hours_per_day, days=days, num_gpus=num_gpus, prices=prices,
            interruptible=interruptible, gpu_type=gpu_type, horizon_months=horizon_months)
        return tier
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
