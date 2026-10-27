# Kronos Autotrader: Plan, Options, Costs, Compliance

> **Disclaimer.** This is technical and operational guidance, not financial, investment, tax or
> legal advice. Trading can lose all the money you put in. Before a large deployment, get advice
> from a tax professional and a lawyer in each country where you are tax resident.

---

> **Update (Oct 2026): every real-model test failed.** Kronos showed no edge on BTC, and on stock
> indices only a weak signal that did not survive realistic costs. A weekly index ranking and a
> fine-tuned model also failed their pre-registered tests. Summary and details:
> [`BACKTEST_RESULTS.md`](BACKTEST_RESULTS.md). Live trading is not recommended; paper trading is off.

## 0. Executive summary

Kronos (`shiyu-coder/Kronos`, MIT license) is a **price-forecasting model** for candlestick
(OHLCV) data, not a trading bot. It can say where a price might go. It cannot place orders,
size positions, manage risk or run unattended. This branch adds that missing layer in
`autotrader/`: data fetching, a **long-only, no-leverage** signal built from Kronos's sampled
forecast paths, risk limits and a kill switch, paper and live brokers, a backtester that
compares against a naive baseline, Monte Carlo, Prometheus metrics, alerts, Docker, Terraform
and CI.

The recommended path:

1. Paper-trade BTC on 4-hour bars for free on GitHub Actions for **8+ weeks**.
2. If it beats the naive baseline after costs, go live with **$20** on a cheap server with a
   trade-only, IP-whitelisted API key.
3. Add capital only when each promotion gate below is passed.

**Set expectations now.** On $20, even a very good year (+30%) earns **$6**. That is less than
one month of the cheapest paid server. With $20 you are testing the plumbing, not earning
income. The system only covers its own running costs at roughly **$500–$4,000 of capital**,
depending on hosting (break-even table in §7), and only if the edge survives real costs. The
base-case assumption for any untested ML timing signal is **zero edge after fees** until paper
and live data show otherwise.

---

## 1. What the repo is, and what was added

| Part | In upstream Kronos | Added in this branch |
|---|---|---|
| Forecast OHLCV (`KronosPredictor`) | ✅ (Kronos-mini/small/base on Hugging Face) | Wrapped as a probabilistic forecaster: N sample paths → mean return, spread, P(up) (`autotrader/forecaster.py`) |
| Fine-tuning | ✅ (Qlib/A-share + CSV pipelines) | Not needed to start (optional later, aggressive tier) |
| Backtest | Demo only (Qlib top-K; a Chinese-language script with no fees) | Walk-forward, no look-ahead, fees + slippage + latency, naive null baseline, forecast-quality stats, stress scenarios (`autotrader/backtest.py`) |
| Monte Carlo | ❌ | Block bootstrap with edge haircut (`autotrader/montecarlo.py`) |
| Execution | ❌ | `PaperBroker` + `LiveBroker` (ccxt; marketable-limit IOC orders) (`autotrader/broker.py`) |
| Risk | ❌ | Capital cap, order limits, daily-loss / drawdown / losing-streak breakers, stale-data and outlier-forecast guards, reconciliation, kill switch (`autotrader/risk.py`) |
| Ops | ❌ | Metrics + `/healthz`, Slack/Telegram/ntfy alerts, Docker, compose, systemd, k8s, Terraform, GitHub Actions |
| Tests | 1 regression test (needs Hugging Face) | 9 offline tests (`tests/test_autotrader.py`) using a tiny random model + synthetic prices |

**Verified in this session:** all 9 offline tests pass, and the backtest, scenario and Monte
Carlo commands run end to end on synthetic data. **Not verified:** real-weight backtests on
real market data. The sandbox's network policy blocked `huggingface.co` and every exchange
API. That is Week 1, step 1 in the timeline.

---

## 2. Kronos vs TradingAgents: which to build on?

`TauricResearch/TradingAgents` (Apache-2.0, very active: v0.7.0 in Oct 2026) is a **multi-agent
LLM research pipeline**. Analyst agents (market, news, fundamentals, sentiment) feed a bull/bear
debate, then a trader agent and a risk panel, then a portfolio manager. The output is a
**5-tier rating** (Buy / Overweight / Hold / Underweight / Sell) with a written report. Its own
`backtest.py` states it scores *decision quality* and *"is not a portfolio simulator, and must
not grow one"*. Like Kronos, it has **no order execution**.

| | **Kronos (+ this autotrader)** | **TradingAgents** |
|---|---|---|
| What it produces | Numeric forecast distribution of future candles | Text research report + 5-tier rating |
| Inputs | Price/volume candles only | Prices, indicators, news, SEC filings, Reddit/StockTwits, FRED macro, Polymarket |
| Cost per decision | ≈ $0. CPU inference takes seconds | A dozen or more LLM calls per ticker per date (up to 20 tool rounds per analyst). Roughly cents to a few dollars depending on model; **measure it** before relying on it |
| Fit for $20 | ✅ Marginal cost is zero | ❌ Even $0.10/day is $3/month, 15% of capital every month |
| Reproducibility | ✅ Pinned weights + seed give the same forecast every time | ❌ LLM outputs vary; the repo notes text feeds are not archived |
| Backtest validity | Good: thousands of cheap walk-forward decisions. Caveat: test after the model's release to avoid pretraining overlap | Weak: the LLM has read about the past (look-ahead via training data), so only dates after the model's cutoff are clean, and each cell costs money |
| Decision frequency | Hourly to daily; crypto trades 24/7 | Daily/weekly (default holding period 5 days); strongest on stocks with fundamentals |
| Execution / risk engine | Built in this branch | None (you would build the same layer) |
| Operational weight | ~1.2 GB RAM, one container, no external APIs except market data | Many API keys (LLM provider, Alpha Vantage, FRED…), dependent on scraped data sources (its changelog mentions Yahoo's news feed being down) |
| Explainability | Low (numbers only) | High (readable reasoning) |
| Maturity | Research code with an AAAI-2026 paper, last commit Apr 2026 | Productized CLI, CI, extensive tests, frequent releases |
| Evidence of tradable edge | Forecasting metrics in the paper; edge after costs unproven | Paper shows backtest gains on a few stocks over a few months; edge after costs unproven, contamination risk |

**Recommendation: build on Kronos for the autonomous system.** At your capital level, only
Kronos's zero-marginal-cost, reproducible, backtestable signal makes sense. TradingAgents is
worth adding later, **once capital is ~$5k+ or you trade stocks**, in one of two roles:

- **Weekly overlay / veto.** Run TradingAgents once a week per asset. If it rates
  Underweight/Sell, cap Kronos's `max_weight` at 0.5. This costs a few dollars a month and adds
  news and fundamentals context Kronos can't see.
- **Human-in-the-loop research.** Use its reports for your own longer-term stock picks, not for
  autonomous trading.

Do not let TradingAgents place trades directly. It is non-deterministic, hard to backtest
honestly, and costs money per decision.

---

## 3. Strategy design (what the bot does)

Every closed bar (default 4h, BTC/USD):

1. Fetch the last 400 closed candles.
2. Ask Kronos for **16 independent sample paths** 6 bars (24h) ahead. Upstream `predict()`
   averages its samples away; the autotrader repeats the series in `predict_batch` with
   `sample_count=1` to keep each path separate.
3. Compute `mu` (mean forecast return), `sigma` (spread) and `p_up` (share of paths ending
   higher).
4. **Enter** only if `mu > 1.5 × round-trip cost` **and** `p_up ≥ 0.58`. **Exit** when
   `p_up < 0.48` or `mu < 0`. The gap between entry and exit thresholds stops the bot flipping
   in and out on noise.
5. Size: binary (all of the sleeve or nothing; right for tiny accounts with exchange minimum
   orders) or fractional Kelly with an optional volatility target.
6. Run risk checks, then place a marketable-limit IOC order (live) or a simulated fill (paper).

Round-trip cost at Kraken's lowest tier is about 2 × (0.40% + 0.10%) = **1.0%**. The model must
expect a move of **more than 1.5%** to trade. That is why 5-minute bars, which the upstream
examples use, are unsuitable at retail fee tiers.

---

## 4. Three implementation options

All three are **spot only, long-only, no leverage**. Return ranges are *planning assumptions for
sizing and alerting*, not forecasts. Real results may be outside them in either direction.

| | **Conservative** | **Balanced (recommended start)** | **Aggressive** |
|---|---|---|---|
| Config | `config.conservative.yaml` | `config.example.yaml` | `config.aggressive.yaml` |
| Market / bars | BTC only, daily | BTC (+ETH later), 4h | 5 liquid pairs, 1h |
| Max exposure | 50% of sleeve | 100% | 100%, Kelly-sized, vol-targeted |
| Capital to start | $20 | $20 | **≥ $300** (5 sleeves must each clear ~$5–10 exchange minimums) |
| Fee tier needed | Any | ≤ 0.40%/side | **≤ 0.10%/side** (Binance-class, or maker orders) |
| Trades / month (est.) | 1–4 | 4–15 | 30–150 |
| Annual return planning range | −15% … +25% | −30% … +50% | −50% … +100% |
| Drawdown halt | 12% | 20% | 30% |
| On $20 per year | −$3 … +$5 | −$6 … +$10 | n/a (needs $300) |
| Hosting | GitHub Actions (free) → any server | GitHub Actions (free) → $0–19/mo server | Server required (more CPU) |
| Your time | ~1 h/month | ~2 h/month | ~4–6 h/month, plus fine-tuning |
| **Pros** | Few trades, low fee drag, simple, hard to blow up | Enough trades to learn from in 8 weeks; good balance of cost and signal | Most statistical evidence quickly; diversification |
| **Cons** | Few trades means it takes many months to know if it works; mostly BTC beta | Fee drag matters; needs the edge to be real | Fee- and slippage-dominated; more ways to fail; Binance unavailable to US residents |

**Recommended forbidden actions** (enforced in code where marked ✅):

- ✅ No leverage, margin, futures, perpetuals or options (`max_weight ≤ 1`, spot only)
- ✅ No shorting (long-only logic)
- ✅ No position larger than `capital_cap`; hard env ceiling `KRONOS_HARD_CAP`
- ✅ No live trading without the explicit confirmation env var
- No API key with **withdrawal** permission, ever. No bot-initiated transfers
- No DeFi, smart contracts, bridges, yield farms, staking-for-yield or lending (smart-contract and counterparty risk the bot can't see)
- No meme coins or illiquid pairs (thin order books mean real slippage far above the model's)
- No copy-trading, selling signals or managing anyone else's money (that needs a licence in all three countries; see §8)
- No running live with an un-reviewed config change (deploys are approval-gated)

---

## 5. Pre-deployment checklist

| # | Item | Notes |
|---|---|---|
| 1 | Fork → **private** copy | GitHub forks of public repos are public. Create a new private repo and push this branch, or keep the fork public (paper state isn't secret; never commit `.env`) |
| 2 | Decide **tax residence** | Determines exchange, quote currency (USD/EUR/INR) and record-keeping (§8) |
| 3 | Exchange account + KYC | US: Kraken (not NY/WA) or Coinbase Advanced. France: a MiCA-authorised CASP (Kraken, Bitvavo, Coinbase…; check the AMF/ESMA register). India: an FIU-IND-registered exchange (CoinDCX, Binance, ZebPay…) |
| 4 | **Dedicated sub-account** for the bot | Bot sees only its own money; reconciliation is meaningful |
| 5 | API key: **trade + read only, no withdrawals, IP-whitelisted** | Needs a static egress IP, so live trading runs on a server, not GitHub runners |
| 6 | 2FA with hardware key or TOTP on exchange, GitHub, cloud | Phishing-resistant if possible |
| 7 | Alert channel | Telegram bot (free), ntfy.sh (free phone push) or Slack webhook |
| 8 | GitHub: Actions enabled, `production` environment with **required reviewer = you** | Gates deploy + kill-switch workflows |
| 9 | (Live on AWS) AWS account, budget alert, OIDC deploy role | Terraform in `deploy/terraform/aws-ec2` |
| 10 | Local: Python 3.11, Docker | Repo pins `pandas==2.2.2`; tested here with Python 3.13 / pandas 3.0 too |

**Secrets (never in git):** `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`,
(`EXCHANGE_API_PASSWORD`), `SLACK_WEBHOOK_URL` / `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` /
`NTFY_TOPIC`, `KRONOS_LIVE_CONFIRM`, `GRAFANA_ADMIN_PASSWORD`, `GHCR_TOKEN` (if the image is
private). Template: `deploy/.env.example`.

---

## 6. Setup, testing and deployment

### 6.1 Local install and offline tests (no keys, no network beyond pip)

```bash
git clone <your-private-repo> kronos && cd kronos && git checkout autotrader-paper-first
python3.11 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements-autotrader.txt pytest
python -m pytest tests/test_autotrader.py -q          # expect: 9 passed
```

### 6.2 Backtest with the real model (dry-run, no money)

```bash
# 1) History (public endpoints, no keys). Kraken returns only ~720 bars, so use bitstamp/coinbase for history.
python -m autotrader.data --exchange bitstamp --symbol BTC/USD --timeframe 4h --since 2025-05-01 --out data/BTCUSD_4h.csv

# 2) Forecasts. Expensive (~0.5–2 s per decision on CPU), cached and resumable.
#    Decisions start 2025-09-01, AFTER Kronos's public release, to avoid pretraining overlap.
C=deploy/config/config.example.yaml
python -m autotrader.backtest forecast --config $C --data data/BTCUSD_4h.csv --start 2025-09-01 --end 2026-10-01 --forecaster kronos --out runs/fc_kronos.csv
python -m autotrader.backtest forecast --config $C --data data/BTCUSD_4h.csv --start 2025-09-01 --end 2026-10-01 --forecaster naive  --out runs/fc_naive.csv

# 3) Simulate (seconds), stress-test, Monte Carlo
python -m autotrader.backtest simulate  --config $C --data data/BTCUSD_4h.csv --forecasts runs/fc_kronos.csv --out runs/bt_kronos
python -m autotrader.backtest scenarios --config $C --data data/BTCUSD_4h.csv --forecasts runs/fc_kronos.csv --out runs/scen_kronos
python -m autotrader.montecarlo --equity runs/bt_kronos/equity.csv --timeframe 4h --capital 20 --days 365 --sims 5000 --haircut 0.5
```

Or run **Actions → backtest → Run workflow**. It does all of the above and posts the tables on
the run page.

**Tests to run:**

| Test | Period | What it answers |
|---|---|---|
| A. Out-of-sample base | 2025-09-01 → 2026-10-01, 4h, BTC/USD | Does it beat the naive baseline *and* buy-and-hold risk-adjusted, after 0.40% fees + 10 bps slippage? |
| B. Tune / test split | Tune thresholds on 2025-09 → 2026-03; evaluate **once** on 2026-04 → 2026-10 | Is it overfit? Report only the untouched half |
| C. Stress | Same, via `scenarios`: fees ×2, slippage ×3, 1-bar latency, all combined | Does any edge survive realistic bad days? |
| D. Cross-asset | ETH/USD, same settings, no re-tuning | Is the edge general or a BTC accident? |
| E. Regime slices | Split A into up / down / sideways months | Is it just long beta? (compare `exposure` vs return) |
| F. Monte Carlo | Bootstrap A's equity, 10-day blocks, 50% haircut | Probability of loss, of a >20% drawdown, and the p5–p95 range over 1 year |

**Metrics reported** (`summary.json`): total return, CAGR, annualised volatility, Sharpe,
Sortino, max drawdown, Calmar, trades, round trips, win rate, average round-trip return, fees
paid, exposure, breaker halts, and forecast quality (directional hit rate, Spearman IC, Brier
score of `p_up` vs a climatology baseline). All are reported against buy-and-hold.

**Example output format.** This is a real run in this session, but on **synthetic random-walk
data with the naive forecaster**. It shows the format only and is *not* a performance estimate:

```text
       scenario  end_equity  total_return  sharpe  max_dd  trades   fees  bh_return
           base       15.90       -0.2049 -1.4216 -0.2049      50 3.5122     2.0029
        fees_x2       19.55       -0.0226 -1.1688 -0.0226       4 0.6303     2.0029
    slippage_x3       15.83       -0.2085 -1.6774 -0.2085      46 3.2596     2.0029
one_bar_latency       16.24       -0.1882 -0.9779 -0.2039      54 4.0856     2.0029
     worst_case       20.00        0.0000  0.0000  0.0000       0 0.0000     2.0029
```

Lesson from this run: a signal with no edge paid **$3.51 in fees on $20** (18% of capital) and
tripped the 20% drawdown breaker. That is exactly what the cost hurdle and breakers exist to
catch.

**Monte Carlo output (same synthetic run, 2,000 sims, 1 year, $20, 50% haircut):**

| Percentile | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| Final equity ($) | 14.05 | 16.04 | 17.40 | 18.67 | 20.06 |
| Max drawdown | −32.3% | −22.9% | −16.7% | −11.4% | −5.4% |

P(loss) = 92.8%, P(drawdown > 20%) = 35.6%. A strategy showing numbers like these must not go
live.

**Promotion gates (all must pass):**

| Gate | Criteria |
|---|---|
| Backtest → paper | Test A Sharpe > naive Sharpe + 0.3 **and** > 0.5 absolute; positive total return in `fees_x2`; IC > 0.02; test B's untouched half not worse than half of the tuned half |
| Paper → live $20 | ≥ 8 weeks / ≥ 10 round trips paper; paper results inside backtest Monte Carlo p5–p95; zero unexplained errors or halts |
| $20 → $200 | ≥ 6 weeks live; realised slippage ≤ 2× modelled; reconciliation clean; monthly tax export works |
| $200 → $2k | ≥ 3 months live; live Sharpe > 0.5; drawdown within limit; tax-advisor check if crossing reporting thresholds |
| > $10k | Professional tax/legal advice; cold storage or multisig for funds not actively traded; second person able to run the kill switch |

### 6.3 Paper trading on GitHub Actions ($0)

`.github/workflows/paper-trade.yml` runs `python -m autotrader.run --once` 5 minutes after each
4h candle, using **no keys**. It commits `state/` (positions, `fills.csv`, `decisions.jsonl`,
`metrics.prom`) to the `paper-state` branch. Add the alert secrets in *Settings → Secrets*.
Choose the preset with the repo variable `KRONOS_PAPER_CONFIG` (e.g. `config.conservative.yaml`).

Cost: public repo free; private repo about 4 min × 6 runs/day ≈ 720 of the 2,000 free
minutes/month.

### 6.4 CI/CD

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | push / PR | CPU-torch install, offline tests, config validation, gitleaks secret scan, Docker build |
| `paper-trade.yml` | cron every 4h | Paper cycle, state committed to `paper-state` |
| `backtest.yml` | manual | Data → Kronos + naive forecasts → simulate, scenarios, Monte Carlo → artifact + summary |
| `deploy.yml` | tag `v*` | Multi-arch image → GHCR → **manual approval** (`production` env) → AWS SSM roll-out + health check |
| `kill-switch.yml` | manual (works from the GitHub mobile app) | HALT / FLATTEN / RESUME on the server via SSM, approval-gated, reason logged |

Sample job (from `ci.yml`):

```yaml
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip, cache-dependency-path: requirements-autotrader.txt }
      - run: |
          pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
          pip install -r requirements-autotrader.txt pytest
      - run: python -m pytest tests/test_autotrader.py -q
```

Hardening: pin third-party actions to commit SHAs once you've reviewed them, and enable
Dependabot for `requirements-autotrader.txt` and actions.

### 6.5 Infrastructure

**Hosting options for live:**

| Option | $/month | Static IP for key whitelist | Secrets | Notes |
|---|---|---|---|---|
| Oracle Cloud Always Free (ARM, 4 OCPU/24 GB) | $0 | ✅ | OCI Vault (free tier) | Capacity is often unavailable; idle instances can be reclaimed |
| Hetzner CX22 (2 vCPU / 4 GB) | ~€4–6 | ✅ | SOPS + age file | Cheapest reliable choice; EU data centres |
| **AWS t4g.small** (Terraform provided) | ~$19 incl. IPv4, disk, KMS | ✅ Elastic IP | SSM Parameter Store + KMS, IAM least privilege | Fits within your $100 budget; best tooling |
| Home machine / Raspberry Pi 5 | electricity | Depends on ISP | Local file | Fine for testing; power and network outages are your risk |

**Dockerfile** (`deploy/Dockerfile`, abridged). It bakes the pinned model weights in, runs as a
non-root user and has a healthcheck:

```dockerfile
FROM python:3.11-slim
ENV HF_HOME=/opt/hf PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
COPY requirements-autotrader.txt .
RUN pip install -r requirements-autotrader.txt
RUN python -c "from huggingface_hub import snapshot_download as d; \
  d('NeoQuasar/Kronos-Tokenizer-base', revision='0e0117…'); d('NeoQuasar/Kronos-small', revision='901c26c…')"
ENV HF_HUB_OFFLINE=1
COPY model/ model/
COPY autotrader/ autotrader/
USER trader
HEALTHCHECK CMD python -c "...urlopen('http://127.0.0.1:9108/healthz')..."
ENTRYPOINT ["python","-m","autotrader.run"]
CMD ["--config","/config/config.yaml","--loop"]
```

**docker-compose** (`deploy/docker-compose.yml`): the `bot` service has memory/CPU limits,
`no-new-privileges`, metrics on localhost only and log rotation. The `--profile monitoring`
profile adds Prometheus (90-day retention), Alertmanager and Grafana. Grafana stays on
localhost; reach it via an SSH or SSM tunnel.

```bash
cd deploy && cp .env.example .env && chmod 600 .env && cp config/config.example.yaml config/config.yaml
docker compose up -d bot                         # paper mode by default
docker compose --profile monitoring up -d        # optional dashboards
curl -s localhost:9108/healthz; curl -s localhost:9108/metrics | grep kronos_
```

**Terraform** (`deploy/terraform/aws-ec2/`, example resource):

```hcl
resource "aws_ssm_parameter" "secret" {
  for_each = toset(var.secret_names)          # EXCHANGE_API_KEY, EXCHANGE_API_SECRET, ...
  name     = "/kronos/${each.key}"
  type     = "SecureString"
  key_id   = aws_kms_key.kronos.arn
  value    = "CHANGE_ME"
  lifecycle { ignore_changes = [value] }      # real values set out of band, never in TF state
}
```

The module also creates a KMS key with rotation, an IAM role that can read only `/kronos/*`, a
security group with **no inbound ports** (admin via SSM Session Manager), IMDSv2, an encrypted
disk, an Elastic IP (output `egress_ip` is the IP to whitelist) and an **AWS Budget** e-mail
alert. `terraform apply` creates billable resources. Run `plan` first and confirm the cost.

**Scheduling alternatives:** compose `--loop` (wakes 90 s after each candle close), the systemd
timer `deploy/systemd/kronos-once.timer` (`OnCalendar=*-*-* 00/4:01:30 UTC`, one-shot), the
Kubernetes CronJob in `deploy/k8s/kronos.yaml` (`concurrencyPolicy: Forbid`), or GitHub Actions
cron (paper only). A file lock in `state/` also prevents two runs trading at once.

### 6.6 Configuration

`deploy/config/config.example.yaml` holds non-secret settings (exchange, symbols, timeframe,
model + pinned revisions, strategy thresholds, costs, risk limits, paths). `deploy/.env.example`
holds secrets and safety switches. Config loading rejects unknown keys, `max_weight > 1`,
inconsistent thresholds, and live mode without `KRONOS_LIVE_CONFIRM=I_ACCEPT_REAL_MONEY_RISK`
or with `capital_cap > KRONOS_HARD_CAP`.

---

## 7. Risk controls, monitoring, costs

### 7.1 Risk controls (implemented)

| Control | Default (balanced) | Behaviour |
|---|---|---|
| Capital cap | $20 | Bot never manages more, even if the account holds more |
| Hard cap env | $25 | Live refuses to start if the config cap exceeds it |
| Order size | min $5 / max $25 | Below-minimum orders skipped (exchange minimums) |
| Daily loss breaker | 5% | Halt + flatten; human reset |
| Drawdown breaker | 20% from peak | Halt + flatten; human reset |
| Losing streak | 5 consecutive losing round trips | Halt + flatten |
| Trades/day | 6 | Blocks new buys; exits always allowed |
| Stale data | > 2 bars old | Skip cycle + alert |
| Forecast sanity | \|mu\| > 25% or NaN | Halt |
| Slippage | > 100 bps on a fill | Halt; also caps the IOC limit price |
| Reconciliation (live) | > 2% mismatch between expected and actual balance | Halt |
| Kill switch | `state/KILL` file or `KRONOS_KILL` env | `HALT` freezes all orders; `FLATTEN` sells everything, then freezes |

Halts persist until `python -m autotrader.run --reset-halt --reason "..."`, which is logged to
`audit.jsonl`. There is no stop-loss order on the exchange (the bot is flat or long spot, and
breakers act at bar close). If you want protection between bars, add an exchange-side stop
order as a later enhancement.

**Secrets and least privilege:** exchange key is trade-only, with no withdrawals, IP-whitelisted
and on a sub-account. Server secrets are in SSM/KMS (or SOPS+age on Hetzner) and rendered to a
root-only `.env`. GitHub uses Environments with a required reviewer, and OIDC to AWS (no
long-lived cloud keys). gitleaks runs in CI. Funds above the trading float stay off-exchange; at
**>$10k**, use a hardware-wallet **multisig** (e.g. 2-of-3) for reserves.

### 7.2 Monitoring and alerting

Metrics (`/metrics` on :9108, also written to `state/metrics.prom`):
`kronos_equity_quote`, `kronos_drawdown_pct`, `kronos_daily_pnl_pct`, `kronos_halted`,
`kronos_position_weight{symbol}`, `kronos_forecast_mu{symbol}`, `kronos_forecast_p_up{symbol}`,
`kronos_data_last_bar_timestamp_seconds{symbol}`, `kronos_last_cycle_timestamp_seconds`,
`kronos_fees_paid_total`, `kronos_orders_total{symbol,side,status}`,
`kronos_errors_total{stage}`, `kronos_slippage_bps` (histogram),
`kronos_cycle_duration_seconds` (histogram).

`/healthz` returns 200 only if the last cycle succeeded within 2 bars + 10 min.

Alert rules (`deploy/prometheus/alerts.yml`):

| Alert | Condition | Severity → channel |
|---|---|---|
| KronosDown | target down 5m | critical → Slack + email |
| KronosHalted | `kronos_halted == 1` | critical |
| KronosNoCycle | no cycle in 9h | critical |
| KronosStaleData | last bar > 3 bars old | critical |
| KronosDrawdownWarning | drawdown > 10% | warning → Slack |
| KronosErrors | > 2 errors/hour | warning |
| KronosHighSlippage | p90 > 50 bps/day | warning |

The bot also pushes trade, halt and error messages directly (Slack/Telegram/ntfy), so paper mode
on GitHub Actions gets alerts without Prometheus.

Grafana dashboard (`deploy/grafana/dashboards/kronos.json`): **row 1** equity, drawdown,
halted, minutes since last cycle; **row 2** equity curve, position weight; **row 3** forecast mu
& P(up), daily PnL & fees; **row 4** orders & errors, slippage & cycle time. A lighter
alternative is Grafana Cloud free tier, or reading `decisions.jsonl` in the `paper-state`
branch.

### 7.3 Cost and revenue model

**One-time:** about 15–25 h of your time; $0 cash (optional hardware security key ~$25–50).

**Monthly infrastructure:**

| Phase | Hosting | $/mo |
|---|---|---|
| Paper (weeks 1–8+) | GitHub Actions | 0 |
| Live $20 | Oracle free / Hetzner / AWS | 0 / ~5 / ~19 |
| Optional | Grafana Cloud free, ntfy/Telegram free | 0 |
| If TradingAgents overlay added later | LLM API, 4 runs/month | measure; likely a few $ |

**Break-even capital = monthly cost ÷ monthly net return** (net of trading fees):

| Monthly net return → | 0.5% | 1% | 2% |
|---|---|---|---|
| $5/mo hosting | $1,000 | $500 | $250 |
| $19/mo hosting | $3,800 | $1,900 | $950 |
| $100/mo (your ceiling) | $20,000 | $10,000 | $5,000 |

A sustained 1%/month net is ~12.7%/year, which would be a good result for a retail systematic
strategy.

**Revenue ranges** (annual, using §4 planning ranges, before tax, after trading fees, before
hosting):

| Capital | Conservative (−15…+25%) | Balanced (−30…+50%) | Aggressive (−50…+100%) |
|---|---|---|---|
| $20 | −$3 … +$5 | −$6 … +$10 | n/a |
| $500 | −$75 … +$125 | −$150 … +$250 | −$250 … +$500 |
| $5,000 | −$750 … +$1,250 | −$1,500 … +$2,500 | −$2,500 … +$5,000 |

**Sensitivity:** each extra 0.1% fee per side costs about 0.2% per round trip. At 10 round trips
a month that is ~2%/month (~24%/year) of drag. **Fee tier matters more than the model.** Use
maker (post-only) orders or a cheaper exchange before scaling the aggressive tier. In India, also
factor in the 1% TDS withheld on each sale, which ties up cash until you file your return.

---

## 8. Compliance, tax and legal checklist

You listed India, France and the US. **Which one is your tax residence, and do you hold US
citizenship or a green card?** US persons are taxed on worldwide income wherever they live, so
two sets of rules may apply at once. Treat this as a checklist to discuss with an advisor, not a
statement of the law. Rules change often; verify current figures.

**United States**
- Crypto is property: every sale or crypto-to-crypto swap is taxable. Short-term gains are taxed at ordinary income rates. Report on Form 8949 / Schedule D.
- Brokers issue Form 1099-DA (gross proceeds from 2025 transactions; cost basis for covered assets from 2026). Reconcile against `state/fills.csv`.
- The wash-sale rule has historically applied to securities, not crypto; proposals to extend it exist, so verify each year.
- Foreign exchange accounts: FBAR (FinCEN 114) if aggregate foreign accounts exceed $10k at any time; FATCA Form 8938 above higher thresholds.
- Use US-permitted venues (Binance.com is not; Kraken and Coinbase are, with state exceptions such as NY). For stocks in a cash account, avoid good-faith/free-riding violations. The pattern-day-trader rule applies to margin accounts (check the current FINRA status).
- Managing others' money or selling personalised signals can require SEC/state adviser or CFTC/NFA registration.

**France**
- Occasional crypto disposals for fiat or goods: flat tax (PFU), historically 30% (12.8% income tax + 17.2% social levies). Check whether 2026 social-levy changes increased it. Crypto-to-crypto swaps are not taxable events. Declare on form 2086. A small annual exemption applies to total disposals (€305 historically).
- **Declare every foreign exchange account** on form 3916-bis each year (penalties per undeclared account).
- Frequent, automated trading can be reclassified as professional (BIC) income; ask an advisor once trading is regular.
- Use a MiCA-authorised CASP. The French PSAN transitional regime ended mid-2026; check the AMF/ESMA register. DAC8 means exchanges report your activity to tax authorities.
- Stocks/ETFs: PFU on gains; a PEA wrapper has tax advantages but restricts which assets you can hold.

**India**
- Virtual digital assets: 30% tax on gains plus surcharge and 4% cess. Only cost of acquisition is deductible. **Losses cannot be offset or carried forward.** For a trading strategy this hurts: you pay on winners and get no relief for losers.
- 1% TDS on transfers above the annual threshold (₹10k; ₹50k for specified persons), withheld by the exchange and claimable in your return. 18% GST on exchange fees. Report in Schedule VDA.
- The Income-tax Act 2025 took effect on 1 April 2026 and renumbered the old sections (115BBH / 194S). Confirm the current section references with your CA.
- Use an FIU-IND-registered exchange; unregistered offshore exchanges have been blocked. For foreign platforms, mind LRS limits and TCS.
- Stocks via broker APIs: SEBI's retail algo framework requires broker-approved APIs, static-IP registration and algo identification. Intraday equity profit is treated as speculative business income.
- Giving advice or signals to others needs SEBI Research Analyst / Investment Adviser registration.

**All jurisdictions:** keep `fills.csv` (timestamp, side, qty, price, fee, order id) and monthly
exchange statements for at least 6–10 years. Export them monthly (see RUNBOOK). Running only
your own money with your own bot generally needs no licence. Offering it to others as a product
does.

---

## 9. Likely pitfalls and mitigations

| Pitfall | Mitigation |
|---|---|
| Fees eat the edge (the synthetic test lost 18% to fees) | Cost hurdle in the signal; 4h/daily bars; check `fees_x2` scenario; seek maker/lower tiers before scaling |
| Backtest on data the model was pretrained on | Test after the model release (backtest defaults start 2025-09-01); pin weights by revision |
| Overfitting thresholds | Tune/test split (Test B); compare with the naive baseline; change one parameter at a time |
| "Edge" is just BTC going up | Compare against buy-and-hold and look at `exposure`; regime slices (Test E) |
| Exchange minimum order size with $20 | Binary sizing; one symbol; `min_order_notional` |
| Geo-blocking: Binance.com returns 451 from US IPs, including GitHub runners | Use Kraken/Coinbase/Bitstamp for data and US accounts |
| Kraken OHLC returns only ~720 bars | Use bitstamp/coinbase for history (`autotrader.data --exchange bitstamp`) |
| GitHub cron runs late, or GitHub disables scheduled workflows in public repos after 60 days without activity | Stale-data guard tolerates delay; check the Actions tab weekly (re-enable if disabled); the bot alerts on every trade, so silence is itself a signal |
| API key leak | No withdrawal permission, IP whitelist, sub-account, gitleaks, SSM/KMS, rotate every 90 days |
| Partial fills / balance drift | IOC orders, reconciliation halt, fills ledger |
| Model or dependency upgrade silently changes behaviour | Pinned HF revisions and pinned torch; re-run the backtest workflow before any upgrade; deploys need approval |
| Running with real money too early | Live gate env var + hard cap + promotion gates |
| Stock markets have hours and holidays (future-timestamp generation assumes 24/7) | Before using stocks, add an exchange calendar to `forecaster.py` |
| Tax surprises (India no loss offset; France 3916-bis; US 1099-DA mismatches) | Monthly export; advisor before scaling |
