import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing

H100 = {"on_demand_hr": 2.5, "spot_hr": 1.5, "reserved_1yr_hr": 2.0, "reserved_3yr_hr": 1.4}
L4 = {"on_demand_hr": 0.8, "spot_hr": 0.35, "reserved_1yr_hr": 0.6, "reserved_3yr_hr": 0.45}


def test_cache_break_even():
    assert abs(pricing.cache_break_even_reads(1.25, 0.10) - 0.25 / 0.9) < 1e-9
    assert not pricing.cache_is_worth_it(0.2)          # below ~0.28 reads: write premium loses
    assert pricing.cache_is_worth_it(20)
    assert not pricing.cache_is_worth_it(1.0, write_mult=2.0)   # 1h TTL needs > 1.11 reads
    assert abs(pricing.cache_cost_multiplier(1e9) - 0.10) < 1e-6


def test_reserved_waste_factor():
    assert pricing.reserved_waste_factor(36, 36) == 1.0
    assert pricing.reserved_waste_factor(24, 36) == 1.5
    assert abs(pricing.reserved_waste_factor(18, 12) - 24 / 18) < 1e-9


def test_cost_based_tier():
    # 24/7 standing service over 3 years -> 3yr reservation
    assert pricing.best_purchasing_option(hours_per_day=24, days=30, num_gpus=1, prices=H100,
                                          interruptible=False, gpu_type="H100",
                                          horizon_months=36)[:2] == ("reserved", "3yr")
    # 18h/day for 12 months: 1yr bills 24h -> 0.6*24/18 = 0.8 = on-demand; tie goes to on_demand
    assert pricing.recommend_tier(18, False, prices=L4, horizon_months=12) == "on_demand"
    # one-off job (no horizon) never reserves
    assert pricing.recommend_tier(24, False, prices=H100) == "on_demand"
    assert pricing.recommend_tier(8, True, prices=H100, gpu_type="H100") == "spot"


def test_unknown_gpu_interrupt_rate_too_high_skips_spot():
    old = pricing.SPOT_INTERRUPT_RATE.get("X")
    pricing.SPOT_INTERRUPT_RATE["X"] = 0.5
    try:
        assert pricing.recommend_tier(8, True, prices=H100, gpu_type="X") == "on_demand"
    finally:
        pricing.SPOT_INTERRUPT_RATE.pop("X")
        if old is not None:
            pricing.SPOT_INTERRUPT_RATE["X"] = old
