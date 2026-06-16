# 📈 Indian Equity Portfolio Risk & Optimization Engine

> A full-stack quantitative finance project built on the Nifty 50 universe — combining Modern Portfolio Theory, multi-metric risk analytics, and RBI monetary policy regime analysis into an interactive Streamlit dashboard.

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app) &nbsp;|&nbsp; **Built by:** Pratham &nbsp;|&nbsp; **Data:** 2019–2025

---

## Overview

Most portfolio projects stop at returns and correlation. This one goes further — it asks:
*does the optimal portfolio change depending on whether the RBI is cutting or hiking rates?*

The answer, from real NSE data: **yes, materially.**

- Financials and Auto outperform by **26%+ annualised** in cut cycles
- Consumer staples are the **only resilient sector** during hike cycles
- IT delivered **−6.5%** in the 2022–23 hike period — contrary to its defensive reputation
- The Max Sharpe portfolio **beats the Nifty 50** on risk-adjusted basis across the full period

---

## Features

| Page | What it shows |
|---|---|
| 🏠 Overview | Universe KPIs, normalised price performance, portfolio comparison |
| ⚠️ Risk Analytics | Rolling vol, EWMA vol, drawdown, Historical & Parametric VaR, CVaR |
| 🎯 Optimization | Efficient frontier, Min Variance & Max Sharpe portfolios, CML, backtest |
| 🏛️ Macro Overlay | RBI regime chart, sector heatmap, allocation shift across regimes |
| 🔍 Stock Explorer | Deep dive on any stock — price, return distribution, regime breakdown |

---

## Methodology

### Data
- **Universe:** Nifty 50 constituents (NSE) via `yfinance`
- **Period:** January 2019 – December 2025 (~1,700 trading days)
- **Benchmark:** Nifty 50 index (`^NSEI`)
- **Macro:** RBI repo rate history from RBI DBIE (official source)
- **Cleaning:** Forward-fill up to 3 days, drop stocks >5% missing, handle corporate actions and ticker changes (e.g. TATAMOTORS → TMPV post-demerger, LTIM → LTM)

### Risk Analytics
| Metric | Method |
|---|---|
| Volatility | 21-day rolling + EWMA (λ=0.94, RiskMetrics standard) |
| Value at Risk | Historical simulation + Parametric (Gaussian) at 95% & 99% |
| Expected Shortfall | CVaR — average loss beyond VaR threshold |
| Max Drawdown | Peak-to-trough decline over full period |
| Sharpe Ratio | Excess return / total volatility (rf = RBI repo ~6.5%) |
| Sortino Ratio | Excess return / downside deviation only |

### Portfolio Optimization
- **Framework:** Markowitz Mean-Variance Optimization
- **Simulation:** 10,000 random portfolios via Monte Carlo (Dirichlet weights)
- **Exact optimization:** `scipy.optimize.minimize` with SLSQP solver
- **Constraints:** Weights sum to 1, long-only (no short selling)
- **Portfolios constructed:**
  - Minimum Variance Portfolio (MVP) — leftmost point on frontier
  - Maximum Sharpe Portfolio — tangency point on Capital Market Line
  - Equal-weight baseline — naive 1/N benchmark

### RBI Macro Overlay
- Labelled every trading day with the RBI monetary policy regime in effect (CUT / HOLD / HIKE) using official MPC decision dates from RBI DBIE
- Computed sector-level annualised returns within each regime
- Re-ran Max Sharpe optimization separately for each regime to show how optimal weights shift with monetary policy

---

## Key Results

```
Portfolio          Annual Return %    Annual Vol %    Sharpe Ratio
─────────────────────────────────────────────────────────────────
Max Sharpe              XX.X%            XX.X%           X.XX
Min Variance            XX.X%            XX.X%           X.XX
Equal Weight            XX.X%            XX.X%           X.XX
Nifty 50 (benchmark)    XX.X%            XX.X%           X.XX
```

```
Sector Returns by Regime (% annualised)
────────────────────────────────────────────────────────────────────────
           Auto    Consumer    Energy    Financials    IT    Materials
CUT        26.3      23.0      20.4       26.8       21.2     32.4
HOLD       13.0       8.7      18.9        5.9       15.2     23.7
HIKE       11.4      12.4      -0.1        2.7       -6.5     -9.5
```

---

## Project Structure

```
india-portfolio-risk-engine/
│
├── app.py                          # Streamlit dashboard (5 pages)
├── requirements.txt                # Python dependencies
│
├── notebooks/
│   ├── Phase1_2_Data_Risk.ipynb    # Data acquisition + risk analytics
│   ├── Phase3_Optimization.ipynb   # Efficient frontier + MPT
│   └── Phase4_MacroOverlay.ipynb   # RBI regime analysis
│
└── data/
    ├── log_returns.csv
    ├── prices_clean.csv
    ├── benchmark_returns.csv
    ├── risk_metrics.csv
    ├── optimized_weights.csv
    ├── portfolio_comparison.csv
    ├── regime_weights.csv
    ├── regime_metrics.csv
    ├── sector_regime_returns.csv
    └── sector_regime_allocation.csv
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data | `yfinance`, `pandas`, `numpy` |
| Risk & Optimization | `scipy`, `numpy` |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Environment | Google Colab + Google Drive |
| Deployment | Streamlit Cloud |

---

## Setup & Run

**Clone the repo:**
```bash
git clone https://github.com/Pratham9s/india-portfolio-risk-engine.git
cd india-portfolio-risk-engine
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the dashboard:**
```bash
streamlit run app.py
```

**To regenerate the data from scratch**, run the notebooks in order:
1. `Phase1_2_Data_Risk.ipynb`
2. `Phase3_Optimization.ipynb`
3. `Phase4_MacroOverlay.ipynb`

Note: Notebooks are designed for Google Colab. Mount your Google Drive before running.

---

## Limitations & Caveats

- **In-sample backtest:** Portfolio weights are optimized on the same period used for backtesting. Real performance requires walk-forward or out-of-sample testing.
- **Static weights:** The backtest assumes no rebalancing. In practice, weights would drift and require periodic rebalancing.
- **Normal distribution assumption:** Parametric VaR assumes normally distributed returns. Phase 2 analysis confirms fat tails exist — Historical VaR is therefore the more conservative and appropriate measure.
- **Survivorship bias:** The universe is based on current Nifty 50 composition. Stocks that were delisted or removed from the index over the period are not included.
- **Macro regime labelling:** Regimes are labelled based on the direction of the most recent RBI decision. Transition periods may exhibit mixed characteristics.

---

## Interview Talking Points

**On the efficient frontier:**
> "The Max Sharpe portfolio concentrates into 8–10 high-quality stocks rather than diversifying equally — the optimizer is exploiting low correlations between sectors to improve risk-adjusted returns."

**On the macro overlay:**
> "Adding RBI regime analysis showed that a single static portfolio misses the biggest macro signal available. The optimal sector allocation in a cut cycle looks materially different from a hike cycle — this is the kind of macro-aware thinking that drives real asset allocation decisions."

**On the IT finding:**
> "Conventionally IT is considered defensive in India. But the data showed IT delivered −6.5% during the 2022–23 hike cycle — because that period coincided with a global tech selloff. It's a good example of how domestic monetary policy and global sector dynamics interact in ways that pure theory doesn't predict."

---

## Author

**Pratham** — MBA Finance Student  
Northcard Capital Analytics Capstone Programme  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/Pratham9s)

---

*Built for learning, portfolio demonstration, and interview preparation. Not financial advice.*
