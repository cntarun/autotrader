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

---

# Stock indices: S&P 500 and Nifty 50, daily (October 2026)

**Verdict: not ready for money. Buying and holding the index beat the strategy in 3 of 4 windows.**

Setup: same model, daily candles (Yahoo Finance), 5-trading-day horizon, preset
`deploy/config/config.index-daily.yaml` (thresholds fixed before any index results), 0.10% fee +
5 bps slippage per side, €1,000 notional (results in %). Colab T4 run.

## Clean window: Sep 2025 → Oct 2026 (after Kronos's release)

| | Nifty 50 | S&P 500 |
|---|---|---|
| Decisions | 284 | 290 |
| Strategy return | **+3.1%** | +1.4% |
| Buy & hold | −7.6% | +19.7% |
| Naive baseline | −6.1% | −11.8% |
| Sharpe / max drawdown | 0.31 / −10.8% | 0.26 / −5.5% |
| Trades / time in market | 22 / 38% | 2 / 6% |
| Directional hit rate | 59.5% | 42.8% |
| Information coefficient | 0.15 | 0.18 |
| Correct when forecast > +1.5% | 67% (n=36) | 50% (n=6) |
| Brier P(up) vs climatology | 0.31 vs 0.25 | 0.52 vs 0.25 |
| Return at 0.25% fees | +0.7% | +1.0% |
| Monte Carlo P(loss), 1 year | 47% | 51% |

- **Nifty 50:** beat buy-and-hold by about 11 points with a smaller drawdown, beat the naive baseline, and stayed positive at higher fees. It **fails the Sharpe > 0.5 gate**, and the sample is small: 284 overlapping 5-day forecasts are about 57 independent weeks, so an IC of 0.15 is only about one standard error from zero. It could be luck.
- **S&P 500:** Kronos predicted a fall almost every day (a persistent bearish bias after the long rise), so the long-only strategy sat in cash 94% of the time and missed a +20% year. Its ranking of good vs bad days (IC 0.18) was informative, but its direction was not.
- **Both:** probabilities remain overconfident (Brier worse than always predicting the base rate).

## Long window: Jan 2022 → Aug 2025 (may overlap Kronos's training data, so treat as optimistic)

Re-run with the window-end fix (2026-10-27).

| | Nifty 50 | S&P 500 |
|---|---|---|
| Decisions | 904 | 918 |
| Strategy return | +3.1% | +26.3% |
| Buy & hold | **+39.7%** | **+33.8%** |
| Naive baseline | −19.8% | −21.0% |
| Sharpe / max drawdown | 0.15 / −11.3% | 0.65 / −12.8% |
| Time in market | 23% | 17% |
| Hit rate / IC | 51% / 0.06 | 48% / 0.10 |
| Correct when forecast > +1.5% | 68% (n=85) | 66% (n=83) |
| Zero-cost return | +13.1% | +40.5% |
| Preset on unseen 2nd half vs buy & hold | −6.6% vs +29.1% | +3.5% vs +53.0% |
| Monte Carlo P(loss), 1 year | 46% | 37% |

## Takeaways

1. **Kronos behaves like a defensive market-timer on indices.** It helped in falling or choppy markets
   (Nifty 2025–26: +3.1% vs −7.6%; S&P 2022 decline) and lagged badly in rising ones (S&P 2025–26:
   +1.4% vs +19.7%; Nifty 2022–25: +3.1% vs +39.7%; S&P 2023–25 second half: +3.5% vs +53%).
2. **Buying and holding the index returned more in 3 of 4 windows.** The strategy beat it only in the
   one falling market. It did have smaller drawdowns and always beat the naive baseline.
3. **There is a small, consistent ranking signal** (IC +0.06 to +0.18 in all four windows; strongly
   positive forecasts right 66–68% of the time where n ≥ 36), unlike BTC (IC ≈ 0). The absolute forecasts
   are biased bearish and overconfident, which keeps a long-only, in-or-out strategy out of the market.
4. Only one result clears Sharpe > 0.5 (S&P long, +0.65), and that is the window that may overlap the
   model's training data.

**Verdict:** no version is ready for real money. The one idea worth testing is a *risk overlay*: stay
invested by default and reduce exposure only on strongly bearish forecasts. This rule must be fixed
in advance and tested on indices it has not seen (clean window) before any paper trading.
