# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,881  
**Projected savings:** $12,252  (**45%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,207 |
| Purchasing (spot/reserved) | $9,790 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

## Purchasing tiers (Extension 1)

| Job | GPU | Before | After | Before $/mo | After $/mo |
|---|---|---|---|---|---|
| job-train-llm | H100 | spot | spot | $7,596 | $7,524 |
| job-train-embed | A100 | spot | spot | $1,393 | $1,393 |
| job-finetune | H100 | spot | spot | $570 | $564 |
| job-infer-chat | A10G | reserved | reserved 3yr | $2,592 | $2,592 |
| job-infer-rag | A100 | reserved | reserved 3yr | $2,160 | $2,160 |
| job-infer-search | L4 | reserved | reserved 3yr | $972 | $1,296 |
| job-dev-sandbox | A10G | spot | spot | $203 | $207 |
| job-batch-eval | H100 | spot | spot | $142 | $141 |

Savings vs on-demand: 39.1% before -> 38.1% after (reserved now billed 24h/day for the full term; spot priced at per-GPU interruption rates).

## Prompt cache gating (Extension 3)

- Break-even: 0.28 reads per written prefix (write 1.25x, read 0.1x)
- Cache disabled for: search
- Optimized inference: $8.48/day if writes were free -> $8.65/day with write cost, $1.149/1M-token

## Reasoning budget (Extension 4)

| Group | Requests | Tokens | Cost | Energy |
|---|---|---|---|---|
| non-reasoning | 91.6% | 83.5% | 83.8% | 6.0% |
| reasoning | 8.4% | 16.5% | 16.2% | 94.0% |

- Cap 10%: 0 requests downgraded, saves $0.00/month and 0.0 kWh/month (traffic already under the cap)
- What-if cap 5%: 81 downgraded, saves $1.60/month and 357.0 kWh/month (38% of inference energy); assumes downgraded requests emit 1/6 the output on the same model

_Figures are June-2026 as-of snapshots; re-baseline before acting._