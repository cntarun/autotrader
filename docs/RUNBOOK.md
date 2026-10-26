# Kronos Autotrader: Runbook

Operator guide: go-live steps, kill switch, incidents, recovery, maintenance.
Background, options and costs are in `docs/AUTOTRADER_PLAN.md`.

## A. Milestones

```
Week 1  [Setup + real backtest]  private repo · CI green · alerts wired · backtest workflow on
        BTC/USD 4h (2025-09 → 2026-10) · Kronos vs naive vs buy-and-hold · decide go/no-go for paper
Week 2  [Paper trading starts]   enable paper-trade.yml · daily glance at alerts · tune/test split
        (Test B) · ETH cross-check (Test D)
Week 3  [Hardening]              pick hosting · open exchange sub-account + trade-only key (unused)
        · terraform plan (no apply yet) or Hetzner box · dry-run compose stack in PAPER mode on the
        server · test kill switch HALT/FLATTEN/RESUME in paper
Week 4  [Review]                 4 weeks of paper vs backtest Monte Carlo band · fix anything odd
        · write down promotion decision
Weeks 5–8+                       keep paper running to ≥ 8 weeks / ≥ 10 round trips → gate review
                                 → live $20 (section B)
```

| Week | Deliverable | Exit criterion |
|---|---|---|
| 1 | Backtest report (artifact) | Gate "backtest → paper" passed, or stop and rethink |
| 2 | `paper-state` branch with decisions | ≥ 80 cycles logged, no errors |
| 3 | Server running paper mode, dashboards up, kill switch tested | All 3 kill-switch actions verified |
| 4 | 4-week paper review | Paper inside backtest p5–p95, no unexplained halts |

## B. Runbook to go live (do in order; stop at any failure)

1. **Confirm the gates.** Backtest → paper and paper → live gates in PLAN §6.2 are met. Save the backtest artifact link and the paper review in `docs/decisions/`.
2. **Exchange.** Create a dedicated sub-account. Deposit **$20** (or local equivalent). Turn on 2FA.
3. **Server.** Hetzner: create a CX22, install Docker, clone the repo to `/opt/kronos`. AWS: `cd deploy/terraform/aws-ec2 && terraform init && terraform plan -var image=... -var repo_url=... -var alert_email=...`, review the cost, then `terraform apply`.
4. **Egress IP.** Note the server's public IP (`terraform output egress_ip`).
5. **API key.** On the exchange, create a key with **query funds + create/cancel orders only**. **No withdrawal or transfer permission.** Whitelist the egress IP only.
6. **Secrets.** AWS: `aws ssm put-parameter --name /kronos/EXCHANGE_API_KEY --type SecureString --key-id alias/kronos --value '…' --overwrite` (repeat for the secret and alert tokens), then `sudo kronos-render-env <region>`. Hetzner: write `deploy/.env` with `chmod 600` (SOPS-encrypted copy in git if you want versioning).
7. **Config.** Copy `deploy/config/config.example.yaml` to `config.yaml`. Set `exchange.id`, `symbols` (quote currency of your account), `costs.fee_rate` to your **actual** tier, and `risk.capital_cap: 20`.
8. **Paper on the server first.** Run with `KRONOS_MODE=paper` for at least 2 cycles: `docker compose up -d bot && docker compose logs -f bot`. Check `curl localhost:9108/healthz` and alerts arrive.
9. **Read-only live check.** In a container shell: `python -c "from autotrader.broker import LiveBroker; ..."`, or simply confirm the bot logs balances. Verify the key can read balances and the IP whitelist works.
10. **Kill-switch drill.** Run `kill-switch.yml` with HALT, confirm `kronos_halted=1` and the alert. Run RESUME with a reason, confirm it cleared.
11. **Switch to live.** In `.env` set `KRONOS_MODE=live`, `KRONOS_LIVE_CONFIRM=I_ACCEPT_REAL_MONEY_RISK`, `KRONOS_HARD_CAP=25`. Then `docker compose up -d bot`.
12. **First live cycle.** Watch logs. If it trades, compare the fill with the exchange UI (qty, price, fee) and with `state/fills.csv`.
13. **First 72 hours.** Check the dashboard twice a day. Any reconciliation or slippage halt means investigate before RESUME.
14. **Record.** Tag the release (`git tag v0.1.0-live && git push --tags`) so the deployed version is known.

## C. Go-live checklist (all ✅ before step 11)

- [ ] CI green on the deployed commit; image tag recorded
- [ ] Backtest artifact: Kronos beats naive + survives `fees_x2`
- [ ] ≥ 8 weeks paper, inside the Monte Carlo band
- [ ] Exchange sub-account holds only bot funds; 2FA on
- [ ] API key: no withdrawals, IP-whitelisted, created < 90 days ago
- [ ] `capital_cap` ≤ `KRONOS_HARD_CAP` ≤ funds you can lose entirely
- [ ] `fee_rate` matches your real tier
- [ ] Alerts: test message received on phone
- [ ] Kill switch HALT / FLATTEN / RESUME drilled
- [ ] Backup timer enabled (`kronos-backup.timer`) and one restore tested
- [ ] Tax record export process written down (section F)
- [ ] You know how to disable the API key from the exchange website on your phone

## D. Kill switch and circuit breakers

| Situation | Action |
|---|---|
| Something looks wrong, not urgent | **HALT**: Actions → kill-switch → HALT (or `echo HALT > /state/KILL`). No orders are sent at all |
| Market crash / you want out now | **FLATTEN**: sells all positions at the next cycle, then freezes. To force an immediate cycle: `docker compose restart bot` |
| Server compromised / key leaked | **Delete the API key on the exchange website first**, then stop the server (`docker compose down` or stop the instance), then rotate every secret |
| Bot halted itself (alert "HALTED: …") | Read `decisions.jsonl` + logs, find the cause, fix it. Only then: `python -m autotrader.run --config /config/config.yaml --reset-halt --reason "…"` |

Breakers (automatic, all lead to halt + exit, human reset): daily loss ≥ 5%, drawdown ≥ 20%,
5 losing round trips in a row, forecast NaN or |mu| > 25%, fill slippage > 100 bps, balance
reconciliation mismatch > 2%. Stale data (> 2 bars old) skips the cycle without halting.

## E. Incident response

1. **Detect.** Alert, or the absence of expected trade/daily messages.
2. **Contain.** HALT (or delete the API key if a compromise is suspected).
3. **Assess.** `docker compose logs --since 24h bot`, `state/decisions.jsonl`, `state/fills.csv`, exchange order history. Compare exchange balances with `--status`.
4. **Fix.** Code or config fix via PR, CI, tag, approval-gated deploy. Never hot-edit on the server.
5. **Recover.** Reconcile state (section G), RESUME with a reason.
6. **Review.** A short note in `docs/incidents/YYYY-MM-DD.md`: what happened, impact ($), root cause, prevention.

| Symptom | Likely cause | First check |
|---|---|---|
| `KronosNoCycle` | Container crash, OOM, exchange API down | `docker compose ps`, `dmesg \| grep -i oom`, exchange status page |
| `KronosStaleData` | Exchange OHLC lag, wrong symbol, geo-block (HTTP 451) | `python -m autotrader.data --exchange … --since <yesterday> --out /tmp/x.csv` |
| reconcile_mismatch | Manual trade on the sub-account, partial fill, deposit/withdrawal | Exchange trade history vs `fills.csv` |
| slippage halt | Thin book, volatile candle, stale reference price | Order book depth; consider lower `max_order_notional` |
| Orders "unfilled" repeatedly | IOC limit too tight, below exchange minimum, insufficient funds | Exchange order history + rejection reason |

## F. Maintenance

**Weekly (10 min):** glance at the dashboard (equity, drawdown, fees, slippage); confirm the
Actions tab shows paper-trade/CI healthy; read any warning alerts.

**Monthly (30–60 min):**
- Export `fills.csv` + the exchange statement to your tax folder (keep 6–10 years)
- Compare live/paper results with the backtest Monte Carlo band
- Check exchange fee tier and update `costs.fee_rate`
- Apply OS security updates (`dnf upgrade` / `apt upgrade`) and reboot
- Check the cloud bill vs budget
- Restore-test one backup

**Quarterly:** rotate the API key (new key → update secret → render-env → restart → delete old
key); re-run the backtest workflow on the latest 12 months; review the dependency updates
Dependabot queued.

**Upgrade policy:** never upgrade the model (HF revision), torch or strategy thresholds directly
in production. Change on a branch, run the backtest workflow, paper-trade the change for at
least 2 weeks alongside the current version (second `paper-state-*` branch), then deploy via
tag + approval. Security patches to the OS or base image can ship directly after CI.

**Rollback:** re-run `deploy.yml` on the previous tag, or on the server
`sed -i 's#^KRONOS_IMAGE=.*#KRONOS_IMAGE=ghcr.io/<you>/kronos-autotrader:<previous-tag>#' .env.base && kronos-render-env <region> && docker compose up -d bot`.
State lives in the `kronos_state` volume and is unaffected by image changes.

**Log retention:** container logs rotate at 10 × 20 MB. Prometheus keeps 90 days.
`decisions.jsonl`, `fills.csv`, `audit.jsonl` are append-only and kept forever (backed up daily
via `kronos-backup.timer`, 90-day local retention; copy monthly backups off-server). The
`paper-state` git branch is the paper archive.

## G. Recovery

- **Lost server, volume intact (AWS):** re-run `terraform apply`. Re-attaching the old volume needs a snapshot restore (snapshot monthly).
- **Lost state:** restore the latest `state-*.tgz` into the volume (`docker run --rm -v kronos_kronos_state:/state -v /var/backups/kronos:/b alpine tar xzf /b/state-<ts>.tgz -C /state`). In live mode the bot re-reads real balances on the next cycle; if `entry_price` is missing, breaker stats restart from zero (acceptable).
- **State disagrees with the exchange:** the exchange is the truth. HALT, set `positions` in `state.json` to the exchange balance, RESUME with a reason.

## H. What you must supply before full deployment

- [ ] Tax residence (and whether you are a US citizen/green-card holder)
- [ ] Exchange chosen (and that your account/KYC is approved), quote currency
- [ ] Hosting choice: GitHub-only paper / Oracle free / Hetzner / AWS (+ region)
- [ ] Private GitHub repo URL you want this pushed to, with Actions enabled
- [ ] Alert channel: Telegram bot token + chat id, or ntfy topic, or Slack webhook
- [ ] (AWS) account id, region, alert e-mail; approval to run `terraform apply` (billable)
- [ ] (Live, later) API key + secret, entered **by you** into SSM/.env, never pasted into chat or git
