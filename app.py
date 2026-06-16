"""
Indian Equity Portfolio Risk & Optimization Engine
Streamlit Dashboard — Phase 5

Run locally:
    streamlit run app.py

Run on Colab (temporary public URL):
    !pip install streamlit pyngrok -q
    !ngrok authtoken YOUR_TOKEN
    Run the colab launcher cell at the bottom of this file
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Portfolio Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { color: #8b92a5; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { color: #f0f2f6; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
    .metric-delta-pos { color: #00d4aa; font-size: 0.85rem; }
    .metric-delta-neg { color: #ff6b6b; font-size: 0.85rem; }
    .section-header {
        color: #c9a84c;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #2d3250;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .regime-tag-cut  { background:#1a4d2e; color:#4ade80; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .regime-tag-hike { background:#4d1a1a; color:#f87171; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .regime-tag-hold { background:#3d3a1a; color:#fbbf24; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    div[data-testid="stMetric"] { background:#1e2130; border-radius:8px; padding:0.8rem; border:1px solid #2d3250; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
TRADING_DAYS          = 250
RISK_FREE_RATE_ANNUAL = 0.065
REGIME_COLORS         = {'CUT': '#4ade80', 'HIKE': '#f87171', 'HOLD': '#fbbf24'}
SECTOR_MAP = {
    'HDFCBANK':'Financials','ICICIBANK':'Financials','KOTAKBANK':'Financials',
    'AXISBANK':'Financials','SBIN':'Financials','INDUSINDBK':'Financials',
    'BAJFINANCE':'Financials','BAJAJFINSV':'Financials','HDFCLIFE':'Financials','SBILIFE':'Financials',
    'TCS':'IT','INFY':'IT','HCLTECH':'IT','WIPRO':'IT','TECHM':'IT','LTM':'IT',
    'RELIANCE':'Energy','ONGC':'Energy','BPCL':'Energy','NTPC':'Energy','POWERGRID':'Energy',
    'HINDUNILVR':'Consumer','ITC':'Consumer','NESTLEIND':'Consumer','BRITANNIA':'Consumer',
    'TATACONSUM':'Consumer','TITAN':'Consumer','ASIANPAINT':'Consumer',
    'SUNPHARMA':'Pharma','DRREDDY':'Pharma','CIPLA':'Pharma','DIVISLAB':'Pharma','APOLLOHOSP':'Pharma',
    'MARUTI':'Auto','BAJAJ-AUTO':'Auto','HEROMOTOCO':'Auto','EICHERMOT':'Auto','TMPV':'Auto','M&M':'Auto',
    'LT':'Industrials','ADANIPORTS':'Industrials','GRASIM':'Industrials','ULTRACEMCO':'Industrials',
    'JSWSTEEL':'Materials','TATASTEEL':'Materials','HINDALCO':'Materials',
    'COALINDIA':'Materials','ADANIENT':'Materials','UPL':'Materials',
}

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all outputs from Google Drive or local path."""
    # Try Drive first, fall back to local
    paths = [
        '/content/drive/MyDrive/portfolio_project/',
        './'
    ]
    base = None
    for p in paths:
        try:
            pd.read_csv(p + 'log_returns.csv', nrows=1)
            base = p
            break
        except:
            continue

    if base is None:
        st.error('Data not found. Make sure Google Drive is mounted and portfolio_project folder exists.')
        st.stop()

    log_returns       = pd.read_csv(base + 'log_returns.csv',       index_col=0, parse_dates=True)
    prices_clean      = pd.read_csv(base + 'prices_clean.csv',      index_col=0, parse_dates=True)
    benchmark_returns = pd.read_csv(base + 'benchmark_returns.csv', index_col=0, parse_dates=True).squeeze()
    risk_metrics      = pd.read_csv(base + 'risk_metrics.csv',      index_col=0)
    optimized_weights = pd.read_csv(base + 'optimized_weights.csv', index_col=0)
    portfolio_comparison = pd.read_csv(base + 'portfolio_comparison.csv', index_col=0)

    # Regime data
    try:
        regime_weights   = pd.read_csv(base + 'regime_weights.csv',         index_col=0)
        sector_regime_df = pd.read_csv(base + 'sector_regime_returns.csv',   index_col=0)
        sector_regime_alloc = pd.read_csv(base + 'sector_regime_allocation.csv', index_col=0)
        regime_metrics   = pd.read_csv(base + 'regime_metrics.csv',          index_col=0)
    except:
        regime_weights = sector_regime_df = sector_regime_alloc = regime_metrics = None

    # Clean ticker names
    log_returns.columns  = log_returns.columns.str.replace('.NS','', regex=False)
    prices_clean.columns = prices_clean.columns.str.replace('.NS','', regex=False)

    return (log_returns, prices_clean, benchmark_returns, risk_metrics,
            optimized_weights, portfolio_comparison,
            regime_weights, sector_regime_df, sector_regime_alloc, regime_metrics, base)

# ── Portfolio metrics helper ───────────────────────────────────────────────
def portfolio_metrics(weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE_ANNUAL):
    weights      = np.array(weights)
    port_return  = np.dot(weights, mean_returns)
    port_var     = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_vol     = np.sqrt(port_var)
    sharpe       = (port_return - rf) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe

@st.cache_data
def run_monte_carlo(_log_returns, n=5000):
    mean_ret = _log_returns.mean() * TRADING_DAYS
    cov_mat  = _log_returns.cov()  * TRADING_DAYS
    n_assets = len(mean_ret)
    np.random.seed(42)
    results = []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = portfolio_metrics(w, mean_ret, cov_mat)
        results.append({'Return': r*100, 'Volatility': v*100, 'Sharpe': s})
    return pd.DataFrame(results), mean_ret, cov_mat

# ── Load data ─────────────────────────────────────────────────────────────
(log_returns, prices_clean, benchmark_returns, risk_metrics,
 optimized_weights, portfolio_comparison,
 regime_weights, sector_regime_df, sector_regime_alloc, regime_metrics, base) = load_data()

mean_returns = log_returns.mean() * TRADING_DAYS
cov_matrix   = log_returns.cov()  * TRADING_DAYS
n_assets     = len(mean_returns)
tickers      = log_returns.columns.tolist()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Portfolio Risk Engine")
    st.markdown("**Nifty 50 Universe | 2019–2025**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "⚠️ Risk Analytics",
         "🎯 Optimization",
         "🏛️ Macro Overlay",
         "🔍 Stock Explorer"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Data sources**")
    st.caption("• Yahoo Finance (NSE)")
    st.caption("• RBI DBIE (repo rate)")
    st.divider()
    st.caption("Built by Pratham | MBA Finance")
    st.caption("Northcard Capital Analytics")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Indian Equity Portfolio Risk & Optimization Engine")
    st.markdown("**Nifty 50 Universe · 2019–2025 · RBI Macro Overlay**")
    st.divider()

    # Top KPIs
    bench_ret = float(benchmark_returns.mean()) * TRADING_DAYS
    bench_vol = float(benchmark_returns.std())  * np.sqrt(TRADING_DAYS)
    bench_sharpe = (bench_ret - RISK_FREE_RATE_ANNUAL) / bench_vol

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe",       f"{n_assets} stocks",  "Nifty 50")
    c2.metric("Period",         "6 years",             "2019–2025")
    c3.metric("Nifty Return",   f"{bench_ret*100:.1f}%", "Annualised")
    c4.metric("Nifty Vol",      f"{bench_vol*100:.1f}%", "Annualised")
    c5.metric("Nifty Sharpe",   f"{bench_sharpe:.2f}",  "vs 6.5% rf")

    st.divider()

    # Normalised price chart
    st.markdown('<div class="section-header">Market Performance — Nifty 50 vs Top Stocks</div>', unsafe_allow_html=True)

    top_stocks = (log_returns.mean() * TRADING_DAYS).sort_values(ascending=False).head(6).index.tolist()
    norm_prices = (prices_clean[top_stocks] / prices_clean[top_stocks].iloc[0]) * 100

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(top_stocks):
        fig.add_trace(go.Scatter(
            x=norm_prices.index, y=norm_prices[col],
            name=col, line=dict(width=1.8, color=colors[i]), opacity=0.85
        ))

    # Nifty 50 benchmark
    nifty_norm = (1 + benchmark_returns).cumprod() * 100
    nifty_norm = nifty_norm.reindex(norm_prices.index).ffill()
    fig.add_trace(go.Scatter(
        x=nifty_norm.index, y=nifty_norm.values,
        name='Nifty 50', line=dict(width=2.5, color='white', dash='dash')
    ))

    fig.update_layout(
        template='plotly_dark', height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title='Indexed (base=100)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio comparison table
    st.divider()
    st.markdown('<div class="section-header">Portfolio Strategy Comparison</div>', unsafe_allow_html=True)

    styled = portfolio_comparison.style\
        .background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn')\
        .format({'Annual Return %': '{:.2f}', 'Annual Vol %': '{:.2f}', 'Sharpe Ratio': '{:.3f}'})\
        .set_properties(**{'text-align': 'center'})
    st.dataframe(portfolio_comparison, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Analytics":
    st.title("Risk Analytics")
    st.divider()

    # Stock selector
    selected = st.multiselect(
        "Select stocks to analyse",
        options=tickers,
        default=['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN']
    )

    if not selected:
        st.warning("Please select at least one stock.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Rolling 21-day Volatility (Annualised)</div>', unsafe_allow_html=True)
        roll_vol = log_returns[selected].rolling(21).std() * np.sqrt(TRADING_DAYS) * 100
        fig = go.Figure()
        for i, s in enumerate(selected):
            fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol[s],
                                     name=s, line=dict(width=1.5)))
        fig.update_layout(template='plotly_dark', height=340,
                          yaxis_title='Vol (%)', margin=dict(l=0,r=0,t=10,b=0),
                          hovermode='x unified', legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Drawdown from Peak</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for s in selected:
            cum = (1 + prices_clean[s].pct_change()).cumprod()
            dd  = (cum - cum.cummax()) / cum.cummax() * 100
            fig.add_trace(go.Scatter(x=dd.index, y=dd, name=s,
                                     line=dict(width=1.5), fill='tozeroy', opacity=0.4))
        fig.update_layout(template='plotly_dark', height=340,
                          yaxis_title='Drawdown (%)', margin=dict(l=0,r=0,t=10,b=0),
                          hovermode='x unified', legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Risk metrics table for selected stocks
    st.markdown('<div class="section-header">Risk Metrics Summary</div>', unsafe_allow_html=True)
    if len(risk_metrics.index) > 0:
        # risk_metrics index has no .NS already
        available = [s for s in selected if s in risk_metrics.index]
        if available:
            st.dataframe(risk_metrics.loc[available].style.format('{:.3f}'),
                         use_container_width=True)

    st.divider()

    # VaR comparison
    st.markdown('<div class="section-header">Value at Risk — Historical vs Parametric (95%)</div>', unsafe_allow_html=True)
    from scipy.stats import norm as scipy_norm

    var_data = []
    for s in selected:
        r = log_returns[s].dropna()
        h_var = -np.percentile(r, 5) * 100
        p_var = -(r.mean() + scipy_norm.ppf(0.05) * r.std()) * 100
        var_data.append({'Stock': s, 'Historical VaR 95%': h_var, 'Parametric VaR 95%': p_var})

    var_df = pd.DataFrame(var_data).set_index('Stock')

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Historical VaR 95%', x=var_df.index,
                         y=var_df['Historical VaR 95%'], marker_color='#f87171'))
    fig.add_trace(go.Bar(name='Parametric VaR 95%', x=var_df.index,
                         y=var_df['Parametric VaR 95%'], marker_color='#60a5fa'))
    fig.update_layout(template='plotly_dark', height=340, barmode='group',
                      yaxis_title='Daily VaR (%)', margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Optimization":
    st.title("Portfolio Optimization")
    st.divider()

    with st.spinner("Running Monte Carlo simulation (5,000 portfolios)..."):
        mc_df, mean_ret, cov_mat = run_monte_carlo(log_returns)

    # Load saved optimal weights
    mvp_weights = optimized_weights['MVP weight'].values
    msr_weights = optimized_weights['Max Sharpe weight'].values
    eq_weights  = np.array([1/n_assets]*n_assets)

    mvp_ret, mvp_vol, mvp_sharpe = portfolio_metrics(mvp_weights, mean_ret, cov_mat)
    msr_ret, msr_vol, msr_sharpe = portfolio_metrics(msr_weights, mean_ret, cov_mat)
    eq_ret,  eq_vol,  eq_sharpe  = portfolio_metrics(eq_weights,  mean_ret, cov_mat)
    bench_ret = float(benchmark_returns.mean()) * TRADING_DAYS
    bench_vol = float(benchmark_returns.std())  * np.sqrt(TRADING_DAYS)
    bench_sharpe = (bench_ret - RISK_FREE_RATE_ANNUAL) / bench_vol

    # Efficient frontier chart
    st.markdown('<div class="section-header">Efficient Frontier — Nifty 50 Universe</div>', unsafe_allow_html=True)

    fig = go.Figure()

    # Monte Carlo cloud
    fig.add_trace(go.Scatter(
        x=mc_df['Volatility'], y=mc_df['Return'],
        mode='markers',
        marker=dict(size=3, color=mc_df['Sharpe'], colorscale='RdYlGn',
                    showscale=True, colorbar=dict(title='Sharpe'), opacity=0.5),
        name='Random portfolios', hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))

    # Key portfolios
    points = [
        (mvp_vol*100, mvp_ret*100, f'Min Variance<br>Sharpe: {mvp_sharpe:.2f}', 'diamond', '#60a5fa', 16),
        (msr_vol*100, msr_ret*100, f'Max Sharpe<br>Sharpe: {msr_sharpe:.2f}',   'star',    '#fbbf24', 20),
        (bench_vol*100, bench_ret*100, f'Nifty 50<br>Sharpe: {bench_sharpe:.2f}', 'cross', '#f87171', 16),
        (eq_vol*100,  eq_ret*100,  f'Equal Weight<br>Sharpe: {eq_sharpe:.2f}',  'square',  '#a78bfa', 12),
    ]
    labels = ['Min Variance', 'Max Sharpe ★', 'Nifty 50', 'Equal Weight']
    for (vx, vy, tip, sym, col, sz), lbl in zip(points, labels):
        fig.add_trace(go.Scatter(
            x=[vx], y=[vy], mode='markers+text',
            marker=dict(symbol=sym, size=sz, color=col, line=dict(color='white', width=1)),
            text=[lbl], textposition='top center',
            textfont=dict(size=10, color=col),
            name=lbl, hovertemplate=tip + '<extra></extra>'
        ))

    # CML
    cml_x = np.linspace(0, msr_vol * 1.6, 100)
    cml_y = RISK_FREE_RATE_ANNUAL + msr_sharpe * cml_x
    fig.add_trace(go.Scatter(
        x=cml_x*100, y=cml_y*100,
        mode='lines', line=dict(dash='dash', color='white', width=1.5),
        name='Capital Market Line', opacity=0.6
    ))

    fig.update_layout(
        template='plotly_dark', height=550,
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Return (%)',
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation='h', y=-0.15)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Portfolio weights
    st.markdown('<div class="section-header">Optimized Portfolio Weights</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Max Sharpe Portfolio", "Min Variance Portfolio"])

    with tab1:
        msr_w_series = pd.Series(msr_weights, index=tickers).sort_values(ascending=False)
        msr_nonzero  = msr_w_series[msr_w_series > 0.005]
        fig = px.bar(msr_nonzero * 100, color=msr_nonzero.index,
                     labels={'value': 'Weight (%)', 'index': 'Stock'},
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.add_hline(y=100/n_assets, line_dash='dash', line_color='red',
                      annotation_text=f'Equal weight ({100/n_assets:.1f}%)')
        fig.update_layout(template='plotly_dark', height=360,
                          showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        mvp_w_series = pd.Series(mvp_weights, index=tickers).sort_values(ascending=False)
        mvp_nonzero  = mvp_w_series[mvp_w_series > 0.005]
        fig = px.bar(mvp_nonzero * 100, color=mvp_nonzero.index,
                     labels={'value': 'Weight (%)', 'index': 'Stock'},
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.add_hline(y=100/n_assets, line_dash='dash', line_color='red',
                      annotation_text=f'Equal weight ({100/n_assets:.1f}%)')
        fig.update_layout(template='plotly_dark', height=360,
                          showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cumulative backtest
    st.markdown('<div class="section-header">Cumulative Return Backtest (Static Weights)</div>', unsafe_allow_html=True)
    msr_daily  = log_returns.values @ msr_weights
    mvp_daily  = log_returns.values @ mvp_weights
    eq_daily   = log_returns.values @ eq_weights
    min_len    = min(len(log_returns), len(benchmark_returns))
    dates      = log_returns.index[:min_len]
    bench_vals = benchmark_returns.values[:min_len]

    def cum_ret(x): return (np.exp(np.cumsum(x)) - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=cum_ret(msr_daily[:min_len]),  name='Max Sharpe',   line=dict(color='#fbbf24', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=cum_ret(mvp_daily[:min_len]),  name='Min Variance', line=dict(color='#60a5fa', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=cum_ret(eq_daily[:min_len]),   name='Equal Weight', line=dict(color='#a78bfa', width=1.5)))
    fig.add_trace(go.Scatter(x=dates, y=cum_ret(bench_vals),           name='Nifty 50',     line=dict(color='#f87171', width=2, dash='dash')))
    fig.add_vline(x='2020-03-23', line_dash='dot', line_color='gray', annotation_text='COVID crash')
    fig.update_layout(template='plotly_dark', height=400,
                      yaxis_title='Cumulative Return (%)',
                      margin=dict(l=0,r=0,t=10,b=0),
                      hovermode='x unified',
                      legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⚠️ In-sample backtest: weights derived from the same period. For illustrative purposes only.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — MACRO OVERLAY
# ══════════════════════════════════════════════════════════════════════════
elif page == "🏛️ Macro Overlay":
    st.title("RBI Macro Overlay")
    st.divider()

    if sector_regime_df is None:
        st.warning("Phase 4 data not found. Run Phase 4 notebook first.")
        st.stop()

    # Regime tags
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div style="text-align:center"><span class="regime-tag-cut">CUT REGIME</span><br><small style="color:#8b92a5">551 days · 31.9%</small></div>', unsafe_allow_html=True)
    c2.markdown('<div style="text-align:center"><span class="regime-tag-hold">HOLD REGIME</span><br><small style="color:#8b92a5">920 days · 53.2%</small></div>', unsafe_allow_html=True)
    c3.markdown('<div style="text-align:center"><span class="regime-tag-hike">HIKE REGIME</span><br><small style="color:#8b92a5">231 days · 13.4%</small></div>', unsafe_allow_html=True)

    st.divider()

    # RBI rate + Nifty chart
    st.markdown('<div class="section-header">Nifty 50 vs RBI Repo Rate — Policy Regimes</div>', unsafe_allow_html=True)

    rbi_rate_changes = [
        ('2019-01-01',6.50,'HOLD'),('2019-02-07',6.25,'CUT'),('2019-04-04',6.00,'CUT'),
        ('2019-06-06',5.75,'CUT'),('2019-08-07',5.40,'CUT'),('2019-10-04',5.15,'CUT'),
        ('2019-12-05',5.15,'HOLD'),('2020-03-27',4.40,'CUT'),('2020-05-22',4.00,'CUT'),
        ('2020-10-09',4.00,'HOLD'),('2021-01-01',4.00,'HOLD'),('2022-05-04',4.40,'HIKE'),
        ('2022-06-08',4.90,'HIKE'),('2022-08-05',5.40,'HIKE'),('2022-09-30',5.90,'HIKE'),
        ('2022-12-07',6.25,'HIKE'),('2023-02-08',6.50,'HIKE'),('2023-04-06',6.50,'HOLD'),
        ('2024-01-01',6.50,'HOLD'),('2025-02-07',6.25,'CUT'),('2025-04-09',6.00,'CUT'),
    ]
    rbi_df = pd.DataFrame(rbi_rate_changes, columns=['date','rate','regime'])
    rbi_df['date'] = pd.to_datetime(rbi_df['date'])
    daily_rate   = rbi_df.set_index('date')['rate'].reindex(
        pd.date_range(log_returns.index[0], log_returns.index[-1], freq='D')).ffill()
    daily_regime = rbi_df.set_index('date')['regime'].reindex(
        pd.date_range(log_returns.index[0], log_returns.index[-1], freq='D')).ffill()
    daily_rate   = daily_rate.reindex(log_returns.index).ffill()
    daily_regime = daily_regime.reindex(log_returns.index).ffill()

    nifty_idx = (1 + benchmark_returns).cumprod() * 100
    nifty_idx = nifty_idx.reindex(log_returns.index).ffill()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=nifty_idx.index, y=nifty_idx.values,
                             name='Nifty 50', line=dict(color='#60a5fa', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_rate.index, y=daily_rate.values,
                             name='Repo Rate %', line=dict(color='#f87171', width=2, dash='dash')), secondary_y=True)

    # Shade regimes
    regime_color_map = {'CUT':'rgba(74,222,128,0.15)', 'HIKE':'rgba(248,113,113,0.15)', 'HOLD':'rgba(251,191,36,0.1)'}
    prev_date = log_returns.index[0]; prev_reg = daily_regime.iloc[0]
    for date, reg in daily_regime.items():
        if reg != prev_reg:
            fig.add_vrect(x0=prev_date, x1=date, fillcolor=regime_color_map.get(prev_reg,'white'),
                          layer='below', line_width=0)
            prev_date = date; prev_reg = reg
    fig.add_vrect(x0=prev_date, x1=log_returns.index[-1],
                  fillcolor=regime_color_map.get(prev_reg,'white'), layer='below', line_width=0)

    fig.update_layout(template='plotly_dark', height=420,
                      margin=dict(l=0,r=0,t=10,b=0), hovermode='x unified',
                      legend=dict(orientation='h', y=1.1))
    fig.update_yaxes(title_text='Nifty 50 (indexed)', secondary_y=False)
    fig.update_yaxes(title_text='Repo Rate (%)', secondary_y=True, range=[3,8])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Sector Returns by Regime (%)</div>', unsafe_allow_html=True)
        fig = px.imshow(
            sector_regime_df,
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            text_auto='.1f',
            aspect='auto'
        )
        fig.update_layout(template='plotly_dark', height=300,
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Sector Allocation Shift by Regime</div>', unsafe_allow_html=True)
        fig = px.bar(
            sector_regime_alloc.T,
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(template='plotly_dark', height=300,
                          yaxis_title='Allocation (%)',
                          margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Regime metrics
    if regime_metrics is not None:
        st.markdown('<div class="section-header">Optimized Portfolio Metrics per Regime</div>', unsafe_allow_html=True)
        st.dataframe(regime_metrics.style.format('{:.2f}').background_gradient(
            subset=['Sharpe'], cmap='RdYlGn'), use_container_width=True)

    st.divider()
    st.markdown("**📌 Key insight from this data:**")
    st.info("""
    - **CUT regime** → Cyclicals lead (Auto +26%, Financials +27%, Materials +32%). Cheap money lifts all boats.
    - **HOLD regime** → IT +15% outperforms, showing defensive characteristics when macro is neutral.
    - **HIKE regime** → IT goes negative (−6.5%), reflecting the 2022-23 global tech selloff coinciding with RBI hikes.
      Consumer staples are the only positive sector (+12.4%) — people still buy essentials.
    - **Implication** → A macro-aware portfolio rotates toward consumer and pharma when RBI signals hikes,
      and back toward financials and cyclicals when RBI pivots to cuts.
    """)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — STOCK EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Stock Explorer":
    st.title("Stock Explorer")
    st.divider()

    stock = st.selectbox("Select a stock", options=tickers, index=tickers.index('RELIANCE') if 'RELIANCE' in tickers else 0)
    sector = SECTOR_MAP.get(stock, 'Unknown')

    # KPIs
    r = log_returns[stock].dropna()
    ann_ret = r.mean() * TRADING_DAYS * 100
    ann_vol = r.std()  * np.sqrt(TRADING_DAYS) * 100
    sharpe  = (ann_ret/100 - RISK_FREE_RATE_ANNUAL) / (ann_vol/100)
    from scipy.stats import norm as scipy_norm
    var95   = -np.percentile(r, 5) * 100
    mdd_val = ((prices_clean[stock] / prices_clean[stock].cummax()) - 1).min() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sector",        sector)
    c2.metric("Annual Return", f"{ann_ret:.1f}%")
    c3.metric("Annual Vol",    f"{ann_vol:.1f}%")
    c4.metric("Sharpe Ratio",  f"{sharpe:.2f}")
    c5.metric("Max Drawdown",  f"{mdd_val:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="section-header">{stock} — Price History</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices_clean.index, y=prices_clean[stock],
                                 fill='tozeroy', line=dict(color='#60a5fa', width=1.5),
                                 fillcolor='rgba(96,165,250,0.1)'))
        fig.update_layout(template='plotly_dark', height=340,
                          yaxis_title='Price (INR)', margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f'<div class="section-header">{stock} — Return Distribution</div>', unsafe_allow_html=True)
        from scipy.stats import norm as scipy_norm
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=r*100, nbinsx=80, name='Actual',
                                   marker_color='#60a5fa', opacity=0.7, histnorm='probability density'))
        x_range = np.linspace(r.min()*100, r.max()*100, 200)
        fig.add_trace(go.Scatter(x=x_range,
                                 y=scipy_norm.pdf(x_range, r.mean()*100, r.std()*100),
                                 name='Normal fit', line=dict(color='#f87171', width=2)))
        fig.add_vline(x=-var95, line_dash='dot', line_color='#fbbf24',
                      annotation_text=f'VaR 95%: {var95:.2f}%')
        fig.update_layout(template='plotly_dark', height=340,
                          xaxis_title='Daily return (%)',
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Regime performance for this stock
    st.divider()
    st.markdown(f'<div class="section-header">{stock} — Performance by RBI Regime</div>', unsafe_allow_html=True)

    rbi_df2 = pd.DataFrame(rbi_rate_changes, columns=['date','rate','regime'])
    rbi_df2['date'] = pd.to_datetime(rbi_df2['date'])
    dr = rbi_df2.set_index('date')['regime'].reindex(
        pd.date_range(log_returns.index[0], log_returns.index[-1], freq='D')).ffill()
    dr = dr.reindex(log_returns.index).ffill()

    regime_perf = {}
    for reg in ['CUT', 'HOLD', 'HIKE']:
        mask = dr == reg
        regime_perf[reg] = {
            'Ann. Return %': round(log_returns[stock][mask].mean() * TRADING_DAYS * 100, 2),
            'Ann. Vol %'   : round(log_returns[stock][mask].std() * np.sqrt(TRADING_DAYS) * 100, 2),
            'Days'         : int(mask.sum())
        }

    regime_perf_df = pd.DataFrame(regime_perf).T
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(regime_perf_df, use_container_width=True)
    with col2:
        fig = px.bar(regime_perf_df, y='Ann. Return %',
                     color=regime_perf_df.index,
                     color_discrete_map={'CUT':'#4ade80','HOLD':'#fbbf24','HIKE':'#f87171'})
        fig.update_layout(template='plotly_dark', height=250,
                          showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
