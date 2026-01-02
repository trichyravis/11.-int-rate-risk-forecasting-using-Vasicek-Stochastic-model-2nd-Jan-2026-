
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model 
import scipy.stats as stats
import warnings
import time

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & INSTITUTIONAL THEME
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Institutional Risk & Yield Terminal", layout="wide")

CORPORATE_BLUE = "#002147" 
GOLD = "#FFD700"

st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, {CORPORATE_BLUE} 0%, #004b8d 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; border-bottom: 5px solid {GOLD};
    }}
    [data-testid="stSidebar"] {{ background-color: {CORPORATE_BLUE} !important; color: white !important; }}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ color: white !important; }}
    div.stButton > button:first-child {{
        background-color: {GOLD} !important; color: {CORPORATE_BLUE} !important;
        font-weight: bold !important; width: 100%; border-radius: 8px; border: none;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: {GOLD} !important; font-weight: bold; color: {CORPORATE_BLUE} !important; 
    }}
    .config-box {{
        background-color: #f8f9fa; padding: 15px; border-radius: 10px;
        border-left: 5px solid {CORPORATE_BLUE}; margin-bottom: 20px; color: {CORPORATE_BLUE};
    }}
    </style>
    <div class="main-header">
        <h1 style="margin-bottom: 0;">INTEREST RATE FORECASTING DASHBOARD</h1>
        <h2 style="margin-top: 0; font-size: 1.5rem; opacity: 0.9;">(ARIMA & Vasicek Stochastic)</h2>
        <p style="margin-top: 10px; font-weight: bold; font-size: 1.1rem;">
            Prof. V. Ravichandran | The Mountain Path - World of Finance
        </p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_label = st.selectbox("Benchmark Maturity", ["US 10Y (^TNX)", "US 30Y (^TYX)", "US 5Y (^FVX)"])
    ticker = ticker_label.split("(")[1].replace(")", "")
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 30)
    
    st.header("🛡️ Risk Parameters")
    conf_level = st.select_slider("Confidence Level (α)", options=[0.90, 0.95, 0.99], value=0.95)
    
    st.header("🎨 UI Settings")
    show_step = st.checkbox("Show Step-Wise Curve", value=True)
    
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

    for _ in range(8): st.write("")
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.15); border: 1px solid {GOLD};">
            <h3 style="color: white !important; margin: 0;">Prof. V. Ravichandran</h3>
            <p style="color: white !important; font-size: 0.85rem; margin: 5px 0;">The Mountain Path - World of Finance</p>
            <hr style="margin: 10px 0; border-color: {GOLD};">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank">
                <button style="background-color: #0077b5; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE & TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast (ARIMA)", "🌪️ GARCH Volatility", "🎲 Stochastic (Vasicek)", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Hub"])

with tabs[0]:
    st.header("📖 Institutional Research Methodology")
    st.markdown("### About this Platform")
    st.write("This quantitative terminal utilizes a dual-engine approach to model interest rates. It employs the **ARIMA** framework for directional pathing and **GARCH** for risk estimation, alongside **Vasicek** stochastic simulations.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📑 Assumptions")
        st.markdown("- **Stationarity:** Yields differenced to stabilize mean.\n- **Clustering:** Volatility is regime-dependent.\n- **Mean Reversion:** Rates revert to local/equilibrium trends.")
    with col2:
        st.subheader("⚠️ Limitations")
        st.markdown("- **Black Swans:** Does not predict structural 'jump' events.\n- **Univariate:** Does not include exogenous variables like GDP.")

if run_btn:
    data = pd.DataFrame()
    wait_times = [0, 5, 10, 20, 30, 60] 
    success = False

    for attempt, delay in enumerate(wait_times):
        if delay > 0:
            st.warning(f"⚠️ Yahoo Finance busy. Retrying in {delay}s...")
            time.sleep(delay)
        with st.spinner(f"Fetching Data {attempt + 1}/6..."):
            try:
                t_obj = yf.Ticker(ticker)
                data = t_obj.history(period=f"{lookback}y")
                if not data.empty:
                    success = True
                    break
            except: continue

    if success:
        yields = data['Close'].dropna()
        if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
        yields = yields.resample('B').last().ffill()
        returns = 100 * yields.pct_change().dropna()

        try:
            # ENGINES
            model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
            arima_fc = model_arima.predict(n_periods=horizon)
            f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
            
            garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
            cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

            with tabs[1]: # Forecast
                st.markdown(f'<div class="config-box"><strong>Current Configuration:</strong> {ticker_label} | ARIMA{model_arima.order}</div>', unsafe_allow_html=True)
                fig_f = go.Figure()
                if show_step:
                    fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, mode='lines+markers', line_shape='hv', line=dict(color='#FF4B4B', width=4)))
                else:
                    fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Actual"))
                    fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA", line=dict(dash='dot', color='orange')))
                st.plotly_chart(fig_f, width='stretch')

            with tabs[2]: # GARCH
                st.subheader("🌪️ Conditional Volatility (GARCH 1,1)")
                fig_v = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red')))
                st.plotly_chart(fig_v, width='stretch')

            with tabs[3]: # VASICEK
                st.subheader("🎲 Vasicek Stochastic Simulation")
                r0, kappa, theta, sigma = yields.iloc[-1]/100, 0.20, 0.045, 0.015
                dt, n_paths = 1/252, 1000
                sim_paths = np.zeros((n_paths, horizon))
                sim_paths[:, 0] = r0
                for i in range(1, horizon):
                    dW = np.random.normal(0, np.sqrt(dt), n_paths)
                    sim_paths[:, i] = sim_paths[:, i-1] + kappa * (theta - sim_paths[:, i-1]) * dt + sigma * dW
                fig_vas = go.Figure()
                for i in range(10): fig_vas.add_trace(go.Scatter(x=f_dates, y=sim_paths[i, :]*100, mode='lines', line=dict(width=1, color='rgba(0,33,71,0.2)'), showlegend=False))
                fig_vas.add_trace(go.Scatter(x=f_dates, y=np.percentile(sim_paths, 50, axis=0)*100, name="Median Path", line=dict(color='orange', width=3)))
                st.plotly_chart(fig_vas, width='stretch')

            with tabs[4]: # BACKTEST
                st.subheader("🧪 30-Day Walk-Forward Validation")
                train, test = yields.iloc[:-30], yields.iloc[-30:]
                bt_model = pm.auto_arima(train, seasonal=False)
                bt_fc = bt_model.predict(n_periods=30)
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=test.index, y=test, name="Realized"))
                fig_bt.add_trace(go.Scatter(x=test.index, y=bt_fc, name="Predicted", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig_bt, width='stretch')

            with tabs[5]: # DIAGNOSTICS
                st.subheader("🔍 Residual Analysis")
                fig_diag = go.Figure(go.Scatter(y=model_arima.resid(), mode='lines', line=dict(color='gray')))
                st.plotly_chart(fig_diag, width='stretch')

            with tabs[6]: # METRICS
                z = stats.norm.ppf(conf_level)
                v, es = cond_vol.iloc[-1]/np.sqrt(252) * z, (cond_vol.iloc[-1]/np.sqrt(252)) * (stats.norm.pdf(z)/(1-conf_level))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                c2.metric("Forecasted", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Daily VaR", f"{v:.3f}%")
                c4.metric("Exp. Shortfall", f"{es:.3f}%")

            with tabs[7]: # EXPORT
                st.download_button("📥 Download CSV", pd.DataFrame({"Date": f_dates, "Forecast": arima_fc}).to_csv().encode('utf-8'), "report.csv")

        except Exception as e: st.error(f"Error: {e}")

with tabs[8]: # Q&A
    st.header("🎓 Q&A Hub")
    with st.expander("❓ What is the Box-Jenkins Methodology?"): st.write("A 3-stage process: Identification, Estimation, and Diagnostics for ARIMA models.")
    with st.expander("❓ Vasicek vs ARIMA?"): st.write("ARIMA is technical/short-term; Vasicek is stochastic/equilibrium-based.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance</p>", unsafe_allow_html=True)
