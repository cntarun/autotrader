# Backtest results: Kronos-small, BTC/EUR (October 2026)

**Verdict: failed the "backtest → paper" gate. Do not trade with real money.**

## Setup

- Model: `NeoQuasar/Kronos-small` + `Kronos-Tokenizer-base`, pinned revisions, zero-shot (no fine-tuning)
- Data: Bitstamp BTC/EUR 4h candles, decisions 2025-09-01 → 2026-10-26 (after the model's public release)
- Strategy: preset `deploy/config/config.example.yaml`, with thresholds fixed **before** seeing any real data. 16 sample paths, 24h horizon, long-only, no leverage
- Costs: 0.40% fee + 10 bps slippage per side (Kraken Pro taker); other tiers below
- Start: €20
- Two independent runs: Colab T4 GPU (every 4h bar, 2,525 decisions) and CPU in the dev sandbox
  (every 12h, 839 decisions). Same-schedule results agree (€16.73 vs €16.61), so sampling
  differences between GPU and CPU don't change the conclusion.

## Results

| | Kronos, 4h (GPU) | Kronos, 12h (CPU) | Naive baseline (4h) | Buy & hold |
|---|---|---|---|---|
| End equity (from €20) | €15.88 | €16.61 | €17.51 | €15.53 |
| Return | −20.6% (drawdown breaker hit) | −16.9% (breaker hit) | −12.5% | −22.3% |
| Sharpe | −1.80 | −1.24 | −3.47 | −0.34 |
| Directional hit rate | 49.5% | 48.6% | ~50% | — |
| Information coefficient | −0.013 | −0.008 | +0.012 | — |
| Brier score of P(up) (climatology = 0.25) | 0.336 | 0.335 | — | — |
| Real moves ÷ forecast spread | 1.93 | 1.87 | — | — |
| Hit rate when forecast > +1.5% | 44.7% (n=190) | 48.1% (n=52) | — | — |

Fee sensitivity (4h, GPU), end equity: 0.40% taker €15.88 · 0.25% maker €15.89 · 0.15% €16.13 ·
0.10% €16.20 · **zero cost €16.73**. Losses are not caused by fees; the signal has no edge.

Tune-on-first-half / test-on-second-half (4h): the best thresholds on Sep 2025–Mar 2026 lost
5.9% on Apr–Oct 2026 while buy-and-hold gained 21.4%.

Monte Carlo (1 year, 50% edge haircut, 4h): median €16.49, p5 €13.05, p95 €19.33,
**P(loss) = 98%**.

## Interpretation

1. **No directional skill.** Hit rate and rank correlation are indistinguishable from chance, and confident calls were *worse* than chance.
2. **Overconfident.** Realised 24h moves are about 1.9× the spread of Kronos's sample paths, so `p_up` swings to extremes it hasn't earned (Brier worse than always predicting the base rate).
3. **Not fixable by cheaper execution or threshold tuning** (zero-cost and tuned results above).
4. **Safety layer worked:** the 20% drawdown breaker capped losses as designed.

Scope: one asset, one year, zero-shot Kronos-small at 4h/12h. Untested: Kronos-base, daily bars,
fine-tuning on BTC history, other assets. Any further attempt should beat the naive baseline
after costs on data it was not tuned on before any paper or live trading.

Reproduce: `notebooks/kronos_backtest_colab.ipynb` (free Colab T4, ~1–1.5 h for three pairs),
or the CLI commands in `docs/AUTOTRADER_PLAN.md` §6.2.
