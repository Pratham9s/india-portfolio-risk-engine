"""
Indian Equity Portfolio Risk & Optimization Engine
Streamlit Dashboard — Phase 5
Theme: Dark Navy & Gold
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Portfolio Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme: Dark Navy & Gold ───────────────────────────────────────────────
GOLD    = "#c9a84c"
GOLD_DIM = "#8a6f2e"
BG_MAIN  = "#0d1117"
BG_CARD  = "#161b22"
BG_DEEP  = "#0d1117"
BORDER   = "#30363d"
TEXT_PRI = "#f0f6fc"
TEXT_SEC = "#8b949e"
ACCENT1  = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#f78166"
ACCENT4  = "#a78bfa"

PLOTLY_COLORS = [GOLD, ACCENT1, ACCENT2, ACCENT3, ACCENT4, "#ffa657", "#79c0ff", "#56d364"]

st.markdown(f"""
<style>
  /* ── Global ── */
  .stApp {{ background-color: {BG_MAIN}; }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
  section[data-testid="stSidebar"] {{ background-color: #010409; border-right: 0.5px solid {BORDER}; }}

  /* ── Metrics ── */
  div[data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 0.5px solid {BORDER};
    border-top: 2px solid {GOLD};
    border-radius: 8px;
    padding: 1rem 1.1rem 0.8rem;
  }}
  div[data-testid="stMetricLabel"] {{ color: {TEXT_SEC} !important; font-size: 0.72rem !important; letter-spacing: 0.08em; text-transform: uppercase; }}
  div[data-testid="stMetricValue"] {{ color: {TEXT_PRI} !important; font-size: 1.6rem !important; font-weight: 600 !important; }}
  div[data-testid="stMetricDelta"] {{ color: {GOLD} !important; font-size: 0.8rem !important; }}

  /* ── Section headers ── */
  .section-header {{
    color: {GOLD};
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-bottom: 0.5px solid {BORDER};
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{ background: {BG_CARD}; border-radius: 8px; border: 0.5px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{ color: {TEXT_SEC}; }}
  .stTabs [aria-selected="true"] {{ color: {GOLD} !important; border-bottom: 2px solid {GOLD} !important; }}

  /* ── Selectbox / Multiselect ── */
  .stMultiSelect [data-baseweb="tag"] {{ background: {GOLD_DIM}22; border: 0.5px solid {GOLD_DIM}; }}
  div[data-baseweb="select"] {{ background: {BG_CARD}; }}

  /* ── Sidebar radio ── */
  .stRadio label {{ color: {TEXT_SEC}; font-size: 0.9rem; }}
  .stRadio [data-testid="stMarkdownContainer"] p {{ color: {TEXT_PRI}; }}

  /* ── Dividers ── */
  hr {{ border-color: {BORDER}; opacity: 0.5; }}

  /* ── Dataframe ── */
  .stDataFrame {{ background: {BG_CARD}; border: 0.5px solid {BORDER}; border-radius: 8px; }}

  /* ── Regime tags ── */
  .tag-cut  {{ background:#1a3a1a; color:#3fb950; padding:3px 12px; border-radius:12px; font-size:0.78rem; font-weight:500; border:0.5px solid #3fb95066; }}
  .tag-hike {{ background:#3a1a1a; color:#f78166; padding:3px 12px; border-radius:12px; font-size:0.78rem; font-weight:500; border:0.5px solid #f7816666; }}
  .tag-hold {{ background:#3a300a; color:{GOLD};   padding:3px 12px; border-radius:12px; font-size:0.78rem; font-weight:500; border:0.5px solid {GOLD}66; }}

  /* ── Caption / info ── */
  .stAlert {{ background: {BG_CARD}; border: 0.5px solid {BORDER}; border-left: 3px solid {GOLD}; border-radius:8px; color:{TEXT_PRI}; }}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark navy gold template ────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRI, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(bgcolor=BG_CARD, bordercolor=BORDER, borderwidth=0.5),
    colorway=PLOTLY_COLORS,
)

# ── Constants ─────────────────────────────────────────────────────────────
TRADING_DAYS          = 250
RISK_FREE_RATE_ANNUAL = 0.065

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

RBI_RATE_CHANGES = [
    ('2019-01-01',6.50,'HOLD'),('2019-02-07',6.25,'CUT'),('2019-04-04',6.00,'CUT'),
    ('2019-06-06',5.75,'CUT'),('2019-08-07',5.40,'CUT'),('2019-10-04',5.15,'CUT'),
    ('2019-12-05',5.15,'HOLD'),('2020-03-27',4.40,'CUT'),('2020-05-22',4.00,'CUT'),
    ('2020-10-09',4.00,'HOLD'),('2021-01-01',4.00,'HOLD'),('2022-05-04',4.40,'HIKE'),
    ('2022-06-08',4.90,'HIKE'),('2022-08-05',5.40,'HIKE'),('2022-09-30',5.90,'HIKE'),
    ('2022-12-07',6.25,'HIKE'),('2023-02-08',6.50,'HIKE'),('2023-04-06',6.50,'HOLD'),
    ('2024-01-01',6.50,'HOLD'),('2025-02-07',6.25,'CUT'),('2025-04-09',6.00,'CUT'),
]

def build_regime_series(index):
    rbi_df = pd.DataFrame(RBI_RATE_CHANGES, columns=['date','rate','regime'])
    rbi_df['date'] = pd.to_datetime(rbi_df['date'])
    full  = pd.date_range(index[0], index[-1], freq='D')
    rate  = rbi_df.set_index('date')['rate'].reindex(full).ffill()
    reg   = rbi_df.set_index('date')['regime'].reindex(full).ffill()
    return rate.reindex(index).ffill(), reg.reindex(index).ffill()

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = ['data/', '/content/drive/MyDrive/portfolio_project/', './']
    base  = None
    for p in paths:
        try:
            pd.read_csv(p + 'log_returns.csv', nrows=1)
            base = p; break
        except: continue
    if base is None:
        st.error('Data not found. Expected a data/ folder with CSV files.')
        st.stop()

    lr   = pd.read_csv(base+'log_returns.csv',       index_col=0, parse_dates=True)
    pc   = pd.read_csv(base+'prices_clean.csv',      index_col=0, parse_dates=True)
    br   = pd.read_csv(base+'benchmark_returns.csv', index_col=0, parse_dates=True).squeeze()
    rm   = pd.read_csv(base+'risk_metrics.csv',      index_col=0)
    ow   = pd.read_csv(base+'optimized_weights.csv', index_col=0)
    pco  = pd.read_csv(base+'portfolio_comparison.csv', index_col=0)
    try:
        rw  = pd.read_csv(base+'regime_weights.csv',           index_col=0)
        srd = pd.read_csv(base+'sector_regime_returns.csv',    index_col=0)
        sra = pd.read_csv(base+'sector_regime_allocation.csv', index_col=0)
        rme = pd.read_csv(base+'regime_metrics.csv',           index_col=0)
    except: rw = srd = sra = rme = None

    lr.columns = lr.columns.str.replace('.NS','',regex=False)
    pc.columns = pc.columns.str.replace('.NS','',regex=False)
    return lr, pc, br, rm, ow, pco, rw, srd, sra, rme

def portfolio_metrics(weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE_ANNUAL):
    w   = np.array(weights)
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sh  = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sh

@st.cache_data
def run_monte_carlo(_lr, n=5000):
    mu  = _lr.mean() * TRADING_DAYS
    cov = _lr.cov()  * TRADING_DAYS
    na  = len(mu)
    np.random.seed(42)
    rows = []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(na))
        r,v,s = portfolio_metrics(w,mu,cov)
        rows.append({'Return':r*100,'Volatility':v*100,'Sharpe':s})
    return pd.DataFrame(rows), mu, cov

# ── Load ──────────────────────────────────────────────────────────────────
log_returns, prices_clean, benchmark_returns, risk_metrics, \
optimized_weights, portfolio_comparison, \
regime_weights, sector_regime_df, sector_regime_alloc, regime_metrics = load_data()

mean_returns = log_returns.mean() * TRADING_DAYS
cov_matrix   = log_returns.cov()  * TRADING_DAYS
n_assets     = len(mean_returns)
tickers      = log_returns.columns.tolist()
daily_rate, daily_regime = build_regime_series(log_returns.index)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{GOLD};font-size:1.1rem;font-weight:600;letter-spacing:0.04em;'>📈 PORTFOLIO RISK ENGINE</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{TEXT_SEC};font-size:0.78rem;margin-top:2px;margin-bottom:1rem;'>Nifty 50 Universe | 2019–2025</div>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate",
        ["🏠  Overview", "⚠️  Risk Analytics", "🎯  Optimization",
         "🏛️  Macro Overlay", "🔍  Stock Explorer"],
        label_visibility="collapsed")
    st.divider()
    st.markdown(f"<div style='color:{TEXT_SEC};font-size:0.75rem;'>Data sources</div>", unsafe_allow_html=True)
    st.caption("• Yahoo Finance (NSE)")
    st.caption("• RBI DBIE (repo rate)")
    st.divider()
    st.markdown(f"<div style='color:{TEXT_SEC};font-size:0.75rem;'>Built by Pratham<br>MBA Finance · Northcard Capital</div>", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown(f"<h1 style='color:{TEXT_PRI};margin-bottom:4px;'>Indian Equity Portfolio Risk & Optimization Engine</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_SEC};margin-top:0;'>Nifty 50 Universe · 2019–2025 · RBI Macro Overlay</p>", unsafe_allow_html=True)
    st.divider()

    bench_ret    = float(benchmark_returns.mean()) * TRADING_DAYS
    bench_vol    = float(benchmark_returns.std())  * np.sqrt(TRADING_DAYS)
    bench_sharpe = (bench_ret - RISK_FREE_RATE_ANNUAL) / bench_vol

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Universe",     f"{n_assets} stocks",    "Nifty 50")
    c2.metric("Period",       "6 years",               "2019–2025")
    c3.metric("Nifty Return", f"{bench_ret*100:.1f}%", "Annualised")
    c4.metric("Nifty Vol",    f"{bench_vol*100:.1f}%", "Annualised")
    c5.metric("Nifty Sharpe", f"{bench_sharpe:.2f}",   "vs 6.5% rf")

    st.divider()
    section("Market Performance — Nifty 50 vs Top Stocks")

    top    = (log_returns.mean()*TRADING_DAYS).sort_values(ascending=False).head(6).index.tolist()
    norm   = (prices_clean[top] / prices_clean[top].iloc[0]) * 100
    fig    = go.Figure()
    for i,col in enumerate(top):
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col,
                                 line=dict(width=1.8, color=PLOTLY_COLORS[i]), opacity=0.9))
    nb = (1+benchmark_returns).cumprod()*100
    nb = nb.reindex(norm.index).ffill()
    fig.add_trace(go.Scatter(x=nb.index, y=nb.values, name='Nifty 50',
                             line=dict(width=2.5, color='white', dash='dash')))
    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                      legend=dict(**PLOTLY_LAYOUT['legend'], orientation='h', y=1.05),
                      yaxis_title='Indexed (base=100)', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section("Portfolio Strategy Comparison")
    st.dataframe(portfolio_comparison.style
                 .format({'Annual Return %':'{:.2f}','Annual Vol %':'{:.2f}','Sharpe Ratio':'{:.3f}'})
                 .background_gradient(subset=['Sharpe Ratio'], cmap='YlOrBr'),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚠️  Risk Analytics":
    st.markdown(f"<h1 style='color:{TEXT_PRI};'>Risk Analytics</h1>", unsafe_allow_html=True)
    st.divider()

    selected = st.multiselect("Select stocks to analyse", options=tickers,
                              default=['RELIANCE','TCS','HDFCBANK','INFY','SBIN'])
    if not selected:
        st.warning("Please select at least one stock.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        section("Rolling 21-day Volatility (Annualised)")
        rv  = log_returns[selected].rolling(21).std() * np.sqrt(TRADING_DAYS) * 100
        fig = go.Figure()
        for i,s in enumerate(selected):
            fig.add_trace(go.Scatter(x=rv.index, y=rv[s], name=s,
                                     line=dict(width=1.5, color=PLOTLY_COLORS[i])))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, yaxis_title='Vol (%)',
                          hovermode='x unified', legend=dict(**PLOTLY_LAYOUT['legend'], orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Drawdown from Peak")
        fig = go.Figure()
        for i,s in enumerate(selected):
            cum = (1+prices_clean[s].pct_change()).cumprod()
            dd  = (cum - cum.cummax()) / cum.cummax() * 100
            fig.add_trace(go.Scatter(x=dd.index, y=dd, name=s,
                                     line=dict(width=1.5, color=PLOTLY_COLORS[i]),
                                     fill='tozeroy', opacity=0.5))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, yaxis_title='Drawdown (%)',
                          hovermode='x unified', legend=dict(**PLOTLY_LAYOUT['legend'], orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section("Risk Metrics Summary")
    avail = [s for s in selected if s in risk_metrics.index]
    if avail:
        st.dataframe(risk_metrics.loc[avail].style.format('{:.3f}')
                     .background_gradient(subset=['Sharpe ratio'], cmap='YlOrBr'),
                     use_container_width=True)

    st.divider()
    section("Value at Risk — Historical vs Parametric (95%)")
    vd = []
    for s in selected:
        r     = log_returns[s].dropna()
        h_var = -np.percentile(r,5)*100
        p_var = -(r.mean()+scipy_norm.ppf(0.05)*r.std())*100
        vd.append({'Stock':s,'Historical VaR 95%':h_var,'Parametric VaR 95%':p_var})
    vdf = pd.DataFrame(vd).set_index('Stock')
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Historical VaR 95%', x=vdf.index, y=vdf['Historical VaR 95%'], marker_color=ACCENT3))
    fig.add_trace(go.Bar(name='Parametric VaR 95%', x=vdf.index, y=vdf['Parametric VaR 95%'], marker_color=ACCENT1))
    fig.update_layout(**PLOTLY_LAYOUT, height=340, barmode='group', yaxis_title='Daily VaR (%)')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯  Optimization":
    st.markdown(f"<h1 style='color:{TEXT_PRI};'>Portfolio Optimization</h1>", unsafe_allow_html=True)
    st.divider()

    with st.spinner("Running Monte Carlo simulation (5,000 portfolios)..."):
        mc_df, mean_ret, cov_mat = run_monte_carlo(log_returns)

    mvp_w = optimized_weights['MVP weight'].values
    msr_w = optimized_weights['Max Sharpe weight'].values
    eq_w  = np.array([1/n_assets]*n_assets)

    mvp_ret,mvp_vol,mvp_sh = portfolio_metrics(mvp_w, mean_ret, cov_mat)
    msr_ret,msr_vol,msr_sh = portfolio_metrics(msr_w, mean_ret, cov_mat)
    eq_ret, eq_vol, eq_sh  = portfolio_metrics(eq_w,  mean_ret, cov_mat)
    br     = float(benchmark_returns.mean())*TRADING_DAYS
    bv     = float(benchmark_returns.std())*np.sqrt(TRADING_DAYS)
    bsh    = (br-RISK_FREE_RATE_ANNUAL)/bv

    section("Efficient Frontier — Nifty 50 Universe")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc_df['Volatility'], y=mc_df['Return'], mode='markers',
        marker=dict(size=3, color=mc_df['Sharpe'], colorscale='YlOrBr',
                    showscale=True, colorbar=dict(title='Sharpe',tickfont=dict(color=TEXT_PRI)), opacity=0.45),
        name='Random portfolios',
        hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))
    pts = [
        (mvp_vol*100,mvp_ret*100,f'Min Variance<br>Sharpe: {mvp_sh:.2f}','diamond',ACCENT1,16),
        (msr_vol*100,msr_ret*100,f'Max Sharpe<br>Sharpe: {msr_sh:.2f}','star',GOLD,22),
        (bv*100,     br*100,     f'Nifty 50<br>Sharpe: {bsh:.2f}',     'cross', ACCENT3,16),
        (eq_vol*100, eq_ret*100, f'Equal Weight<br>Sharpe: {eq_sh:.2f}','square',ACCENT4,12),
    ]
    lbs = ['Min Variance','Max Sharpe ★','Nifty 50','Equal Weight']
    for (vx,vy,tip,sym,col,sz),lbl in zip(pts,lbs):
        fig.add_trace(go.Scatter(x=[vx],y=[vy],mode='markers+text',
            marker=dict(symbol=sym,size=sz,color=col,line=dict(color='white',width=1)),
            text=[lbl],textposition='top center',textfont=dict(size=10,color=col),
            name=lbl,hovertemplate=tip+'<extra></extra>'))
    cx = np.linspace(0, msr_vol*1.6, 100)
    cy = RISK_FREE_RATE_ANNUAL + msr_sh*cx
    fig.add_trace(go.Scatter(x=cx*100,y=cy*100,mode='lines',
        line=dict(dash='dash',color=GOLD_DIM,width=1.5),name='Capital Market Line',opacity=0.7))
    fig.update_layout(**PLOTLY_LAYOUT, height=550,
                      xaxis_title='Annual Volatility (%)', yaxis_title='Annual Return (%)',
                      legend=dict(**PLOTLY_LAYOUT['legend'], orientation='h', y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section("Optimized Portfolio Weights")
    tab1,tab2 = st.tabs(["Max Sharpe Portfolio","Min Variance Portfolio"])

    with tab1:
        s = pd.Series(msr_w, index=tickers).sort_values(ascending=False)
        s = s[s>0.005]
        fig = go.Figure(go.Bar(x=s.index, y=s.values*100,
            marker=dict(color=s.values*100, colorscale='YlOrBr', showscale=False),
            text=[f'{v:.1f}%' for v in s.values*100], textposition='outside'))
        fig.add_hline(y=100/n_assets, line_dash='dash', line_color=ACCENT3,
                      annotation_text=f'Equal weight ({100/n_assets:.1f}%)',
                      annotation_font_color=ACCENT3)
        fig.update_layout(**PLOTLY_LAYOUT, height=360, yaxis_title='Weight (%)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        s = pd.Series(mvp_w, index=tickers).sort_values(ascending=False)
        s = s[s>0.005]
        fig = go.Figure(go.Bar(x=s.index, y=s.values*100,
            marker=dict(color=s.values*100, colorscale='Blues', showscale=False),
            text=[f'{v:.1f}%' for v in s.values*100], textposition='outside'))
        fig.add_hline(y=100/n_assets, line_dash='dash', line_color=ACCENT3,
                      annotation_text=f'Equal weight ({100/n_assets:.1f}%)',
                      annotation_font_color=ACCENT3)
        fig.update_layout(**PLOTLY_LAYOUT, height=360, yaxis_title='Weight (%)')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section("Cumulative Return Backtest (Static Weights)")
    msr_d = log_returns.values @ msr_w
    mvp_d = log_returns.values @ mvp_w
    eq_d  = log_returns.values @ eq_w
    ml    = min(len(log_returns), len(benchmark_returns))
    dates = log_returns.index[:ml]
    bv2   = benchmark_returns.values[:ml]

    def cr(x): return (np.exp(np.cumsum(x))-1)*100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=cr(msr_d[:ml]),name='Max Sharpe',  line=dict(color=GOLD,   width=2.5)))
    fig.add_trace(go.Scatter(x=dates,y=cr(mvp_d[:ml]),name='Min Variance',line=dict(color=ACCENT1,width=2)))
    fig.add_trace(go.Scatter(x=dates,y=cr(eq_d[:ml]), name='Equal Weight',line=dict(color=ACCENT4,width=1.5)))
    fig.add_trace(go.Scatter(x=dates,y=cr(bv2),       name='Nifty 50',   line=dict(color=ACCENT3,width=2,dash='dash')))
    fig.add_vline(x='2020-03-23',line_dash='dot',line_color=TEXT_SEC,
                  annotation_text='COVID crash',annotation_font_color=TEXT_SEC)
    fig.update_layout(**PLOTLY_LAYOUT, height=400, yaxis_title='Cumulative Return (%)',
                      hovermode='x unified',
                      legend=dict(**PLOTLY_LAYOUT['legend'], orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⚠️ In-sample backtest: weights derived from same period. For illustrative purposes only.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — MACRO OVERLAY
# ══════════════════════════════════════════════════════════════════════════
elif page == "🏛️  Macro Overlay":
    st.markdown(f"<h1 style='color:{TEXT_PRI};'>RBI Macro Overlay</h1>", unsafe_allow_html=True)
    st.divider()

    if sector_regime_df is None:
        st.warning("Phase 4 data not found. Run Phase 4 notebook first.")
        st.stop()

    c1,c2,c3 = st.columns(3)
    c1.markdown('<div style="text-align:center;padding:0.5rem;"><span class="tag-cut">CUT REGIME</span><br><small style="color:#8b949e;font-size:0.75rem;">551 days · 31.9%</small></div>', unsafe_allow_html=True)
    c2.markdown('<div style="text-align:center;padding:0.5rem;"><span class="tag-hold">HOLD REGIME</span><br><small style="color:#8b949e;font-size:0.75rem;">920 days · 53.2%</small></div>', unsafe_allow_html=True)
    c3.markdown('<div style="text-align:center;padding:0.5rem;"><span class="tag-hike">HIKE REGIME</span><br><small style="color:#8b949e;font-size:0.75rem;">231 days · 13.4%</small></div>', unsafe_allow_html=True)

    st.divider()
    section("Nifty 50 vs RBI Repo Rate — Policy Regimes")

    nifty_idx = (1+benchmark_returns).cumprod()*100
    nifty_idx = nifty_idx.reindex(log_returns.index).ffill()

    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=nifty_idx.index,y=nifty_idx.values,
                             name='Nifty 50',line=dict(color=GOLD,width=2.2)),secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_rate.index,y=daily_rate.values,
                             name='Repo Rate %',line=dict(color=ACCENT3,width=1.8,dash='dash')),secondary_y=True)

    rmap = {'CUT':'rgba(63,185,80,0.12)','HIKE':'rgba(247,129,102,0.12)','HOLD':'rgba(201,168,76,0.08)'}
    pd_ = log_returns.index[0]; pr_ = daily_regime.iloc[0]
    for dt,rg in daily_regime.items():
        if rg != pr_:
            fig.add_vrect(x0=pd_,x1=dt,fillcolor=rmap.get(pr_,'white'),layer='below',line_width=0)
            pd_=dt; pr_=rg
    fig.add_vrect(x0=pd_,x1=log_returns.index[-1],fillcolor=rmap.get(pr_,'white'),layer='below',line_width=0)

    fig.update_layout(**PLOTLY_LAYOUT, height=420, hovermode='x unified',
                      legend=dict(**PLOTLY_LAYOUT['legend'],orientation='h',y=1.1))
    fig.update_yaxes(title_text='Nifty 50 (indexed)',secondary_y=False,
                     gridcolor=BORDER,zerolinecolor=BORDER)
    fig.update_yaxes(title_text='Repo Rate (%)',secondary_y=True,range=[3,8],
                     gridcolor=BORDER,zerolinecolor=BORDER)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        section("Sector Returns by Regime (%)")
        fig = px.imshow(sector_regime_df,color_continuous_scale='YlOrBr',
                        color_continuous_midpoint=0,text_auto='.1f',aspect='auto')
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Sector Allocation Shift by Regime")
        fig = px.bar(sector_regime_alloc.T, barmode='group',
                     color_discrete_sequence=PLOTLY_COLORS)
        fig.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title='Allocation (%)',
                          legend=dict(**PLOTLY_LAYOUT['legend'],orientation='h',y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    if regime_metrics is not None:
        st.divider()
        section("Optimized Portfolio Metrics per Regime")
        st.dataframe(regime_metrics.style.format('{:.2f}')
                     .background_gradient(cmap='YlOrBr'), use_container_width=True)

    st.divider()
    st.info("""
**📌 Key insights from the data:**
- **CUT regime** → Cyclicals lead: Auto +26%, Financials +27%, Materials +32%. Cheap money lifts all boats.
- **HOLD regime** → IT +15% outperforms, showing defensive characteristics when macro is neutral.
- **HIKE regime** → IT goes negative (−6.5%), reflecting the 2022–23 global tech selloff. Consumer staples the only positive sector (+12.4%).
- **Implication** → A macro-aware portfolio rotates toward consumer and pharma when RBI signals hikes, and back toward financials and cyclicals when RBI pivots to cuts.
    """)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — STOCK EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍  Stock Explorer":
    st.markdown(f"<h1 style='color:{TEXT_PRI};'>Stock Explorer</h1>", unsafe_allow_html=True)
    st.divider()

    stock  = st.selectbox("Select a stock", options=tickers,
                          index=tickers.index('RELIANCE') if 'RELIANCE' in tickers else 0)
    sector = SECTOR_MAP.get(stock,'Unknown')

    r       = log_returns[stock].dropna()
    ann_ret = r.mean()*TRADING_DAYS*100
    ann_vol = r.std()*np.sqrt(TRADING_DAYS)*100
    sharpe  = (ann_ret/100-RISK_FREE_RATE_ANNUAL)/(ann_vol/100)
    var95   = -np.percentile(r,5)*100
    mdd_val = ((prices_clean[stock]/prices_clean[stock].cummax())-1).min()*100

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Sector",        sector)
    c2.metric("Annual Return", f"{ann_ret:.1f}%")
    c3.metric("Annual Vol",    f"{ann_vol:.1f}%")
    c4.metric("Sharpe Ratio",  f"{sharpe:.2f}")
    c5.metric("Max Drawdown",  f"{mdd_val:.1f}%")

    st.divider()
    col1,col2 = st.columns(2)

    with col1:
        section(f"{stock} — Price History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices_clean.index,y=prices_clean[stock],
                                 fill='tozeroy',line=dict(color=GOLD,width=1.8),
                                 fillcolor=f'{GOLD}18'))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, yaxis_title='Price (INR)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section(f"{stock} — Return Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=r*100,nbinsx=80,name='Actual',
                                   marker_color=GOLD,opacity=0.7,histnorm='probability density'))
        xr = np.linspace(r.min()*100,r.max()*100,200)
        fig.add_trace(go.Scatter(x=xr,y=scipy_norm.pdf(xr,r.mean()*100,r.std()*100),
                                 name='Normal fit',line=dict(color=ACCENT3,width=2)))
        fig.add_vline(x=-var95,line_dash='dot',line_color=ACCENT1,
                      annotation_text=f'VaR 95%: {var95:.2f}%',
                      annotation_font_color=ACCENT1)
        fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title='Daily return (%)')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section(f"{stock} — Performance by RBI Regime")

    rp = {}
    for reg in ['CUT','HOLD','HIKE']:
        mask = daily_regime == reg
        rp[reg] = {
            'Ann. Return %': round(log_returns[stock][mask].mean()*TRADING_DAYS*100,2),
            'Ann. Vol %'   : round(log_returns[stock][mask].std()*np.sqrt(TRADING_DAYS)*100,2),
            'Days'         : int(mask.sum())
        }
    rpdf = pd.DataFrame(rp).T
    col1,col2 = st.columns([1,2])
    with col1:
        st.dataframe(rpdf, use_container_width=True)
    with col2:
        fig = go.Figure(go.Bar(
            x=rpdf.index, y=rpdf['Ann. Return %'],
            marker_color=[{'CUT':ACCENT2,'HOLD':GOLD,'HIKE':ACCENT3}[r] for r in rpdf.index],
            text=[f'{v:.1f}%' for v in rpdf['Ann. Return %']], textposition='outside'
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title='Ann. Return (%)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
