"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# ASSUMPTION: token_usage.csv has no reads-per-prefix. Avg reads of each written
# cache prefix before TTL expiry, by team: assistant/rag share a static system
# prompt at high QPS; eval's nightly batch reuses its rubric within a run;
# search prefixes are per-query web snippets that are rarely seen twice.
AVG_CACHE_READS = {"assistant": 20.0, "rag": 10.0, "eval": 3.0, "search": 0.2}
CACHE_WRITE_MULT = pricing.CACHE_WRITE_MULT_5M

# Reasoning budget: max share of requests allowed to use reasoning mode.
REASONING_CAP = 0.10
# ASSUMPTION: data/generate.py inflates reasoning output 6x; a downgraded request
# is assumed to produce 1/6 of the output on the same model (answer quality may drop).
REASONING_OUTPUT_MULT = 6.0


def _tokens(r: dict) -> tuple[int, int, int, bool]:
    return (int(num(r["input_tokens"])), int(num(r["output_tokens"])),
            int(num(r["cached_input_tokens"])), bool(int(num(r["is_batch"]))))


def optimized_cost(r: dict, cascade: bool = True, cache: str = "gated", batch: bool = True,
                   reads: float | None = None, out_tok: int | None = None) -> float:
    """Cost of one request with the chosen levers.

    cache: 'off' | 'free' (cache reads at 0.1x, writes ignored — the original model)
           | 'always' (writes charged, cache always on) | 'gated' (writes charged,
           cache only where cache_is_worth_it for the team's reads/prefix).
    """
    inp, out, cached, is_batch = _tokens(r)
    out = out if out_tok is None else out_tok
    pin, pout = MODEL_PRICES[r["route_tier"] if cascade else "large"]
    reads = AVG_CACHE_READS.get(r["team"], 1.0) if reads is None else reads
    if cache == "off" or (cache == "gated" and not pricing.cache_is_worth_it(reads, CACHE_WRITE_MULT)):
        cached, mult = 0, 1.0
    elif cache == "free":
        mult = 0.10
    else:
        mult = pricing.cache_cost_multiplier(reads, CACHE_WRITE_MULT)
    return pricing.request_cost(inp, out, pin, pout, cached_in=cached, cache_discount=mult,
                                batch=batch and is_batch)


def reasoning_budget(rows: list[dict], cap: float = REASONING_CAP) -> dict:
    """Split cost/tokens/Wh by is_reasoning, then enforce a cap on reasoning traffic.

    Over the cap, keep reasoning for large-route requests first (the router already
    judged them hard) and downgrade the rest to non-reasoning.
    """
    groups = {g: {"requests": 0, "tokens": 0, "cost": 0.0, "wh": 0.0} for g in (0, 1)}
    for r in rows:
        inp, out, _, _ = _tokens(r)
        g = int(num(r["is_reasoning"]))
        groups[g]["requests"] += 1
        groups[g]["tokens"] += inp + out
        groups[g]["cost"] += optimized_cost(r)
        groups[g]["wh"] += sustainability.wh_per_query(inp + out, is_reasoning=bool(g))
    tot = {k: sum(groups[g][k] for g in groups) for k in ("requests", "tokens", "cost", "wh")}
    share = {g: {k: groups[g][k] / tot[k] if tot[k] else 0.0 for k in tot} for g in groups}

    reasoning = sorted((r for r in rows if int(num(r["is_reasoning"]))),
                       key=lambda r: r["route_tier"] != "large")
    allowed = int(cap * len(rows))
    downgraded = reasoning[allowed:]
    d_cost = d_wh = 0.0
    for r in downgraded:
        inp, out, _, _ = _tokens(r)
        new_out = int(out / REASONING_OUTPUT_MULT)
        d_cost += optimized_cost(r) - optimized_cost(r, out_tok=new_out)
        d_wh += (sustainability.wh_per_query(inp + out, is_reasoning=True)
                 - sustainability.wh_per_query(inp + new_out))
    return {"groups": groups, "share": share, "cap": cap, "allowed": allowed,
            "downgraded": len(downgraded), "cost_saved": d_cost, "wh_saved": d_wh,
            "cost_after": tot["cost"] - d_cost, "wh_after": tot["wh"] - d_wh,
            "cost_before": tot["cost"], "wh_before": tot["wh"]}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    total_tokens = sum(_tokens(r)[0] + _tokens(r)[1] for r in rows)

    def total(**kw) -> float:
        return sum(optimized_cost(r, **kw) for r in rows)

    # BASELINE: naive deployment — everything on the large model, no cache, no batch
    base_cost = total(cascade=False, cache="off", batch=False)
    # OPTIMIZED: cascade (route_tier), prompt caching (gated, write cost included), batch API
    opt_cost = total()
    free_cache_cost = total(cache="free")   # original model: cache writes ignored
    steps = {
        "cascade": total(cache="off", batch=False),
        "+cache": total(batch=False),
        "+batch": opt_cost,
    }
    scenarios = {}
    for label, reads in (("low reads (0.2/prefix)", 0.2), ("high reads (20/prefix)", 20.0)):
        scenarios[label] = {m: total(cache=m, reads=reads) for m in ("off", "always", "gated")}

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    rb = reasoning_budget(rows)
    rb_tight = reasoning_budget(rows, cap=0.05)   # illustrative: tighter than current traffic

    if verbose:
        pm = lambda c: pricing.dollars_per_million(c, total_tokens)
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("\nlever by lever:")
        prev = base_cost
        for name, cost in steps.items():
            print(f"  {name:8} ${cost:7.2f}/day  ${pm(cost):.3f}/1M  (-${prev - cost:.2f})")
            prev = cost

        print(f"\n-- cache_is_worth_it (write {CACHE_WRITE_MULT}x, read 0.1x) --")
        print(f"break-even reads/prefix: {pricing.cache_break_even_reads(CACHE_WRITE_MULT):.2f} "
              f"(1h TTL @2x: {pricing.cache_break_even_reads(pricing.CACHE_WRITE_MULT_1H):.2f})")
        for team, reads in AVG_CACHE_READS.items():
            print(f"  {team:10} reads={reads:<5} multiplier={pricing.cache_cost_multiplier(reads, CACHE_WRITE_MULT):.3f}"
                  f"  cache? {pricing.cache_is_worth_it(reads, CACHE_WRITE_MULT)}")
        print(f"before (writes free) : ${free_cache_cost:.2f}/day  ${pm(free_cache_cost):.3f}/1M")
        print(f"after  (writes, gated): ${opt_cost:.2f}/day  ${pm(opt_cost):.3f}/1M")
        for label, s in scenarios.items():
            print(f"  {label:24} off ${s['off']:.2f}  always ${s['always']:.2f}  gated ${s['gated']:.2f}")

        print(f"\n-- reasoning budget (cap {rb['cap']:.0%} of requests) --")
        for g, name in ((0, "non-reasoning"), (1, "reasoning")):
            d, s = rb["groups"][g], rb["share"][g]
            print(f"  {name:14} req={d['requests']:>5} ({s['requests']:.1%})  tokens={d['tokens']:>9,} ({s['tokens']:.1%})"
                  f"  ${d['cost']:.2f} ({s['cost']:.1%})  {d['wh']:,.0f} Wh ({s['wh']:.1%})")
        print(f"  allowed={rb['allowed']}  downgraded={rb['downgraded']}  "
              f"saved ${rb['cost_saved']:.2f}/day, {rb['wh_saved']:,.0f} Wh/day")
        print(f"  (what-if cap {rb_tight['cap']:.0%}: downgraded={rb_tight['downgraded']}  "
              f"saved ${rb_tight['cost_saved']:.2f}/day, {rb_tight['wh_saved']:,.0f} Wh/day "
              f"= {rb_tight['wh_saved'] / rb_tight['wh_before']:.0%} of energy)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "free_cache_daily": round(free_cache_cost, 2), "cache_scenarios": scenarios,
        "reasoning": rb, "reasoning_tight": rb_tight,
    }


if __name__ == "__main__":
    run()
