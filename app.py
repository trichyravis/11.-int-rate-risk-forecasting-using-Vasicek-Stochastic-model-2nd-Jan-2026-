
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
# 1. PAGE CONFIG & THEME
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
        <h2 style="margin-top: 0; font-size: 1.5rem; opacity: 0.9;">(ARIMA, Vasicek & CIR Models)</h2>
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
    
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

    for _ in range(8): st.write("")
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.15); border: 1px solid {GOLD};">
            <h3 style="color: white !important; margin: 0;">Prof. V. Ravichandran</h3>
            <hr style="margin: 10px 0; border-color: {GOLD};">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank">
                <button style="background-color: #0077b5; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast (ARIMA)", "🌪️ GARCH Volatility", "🎲 Vasicek Path", "☀️ CIR Path", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Hub"])

with tabs[0]:
    st.header("📖 Institutional Research Methodology")
    st.write("This terminal provides a multi-model approach: ARIMA for short-term technicals, GARCH for volatility clustering, Vasicek for standard mean-reversion, and CIR for non-negative stochastic modeling.")

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

            # ARIMA Forecast
            with tabs[1]:
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Historical"))
                fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(dash='dot', color='orange')))
                st.plotly_chart(fig_f, width='stretch')

            # GARCH
            with tabs[2]:
                fig_v = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red')))
                st.plotly_chart(fig_v, width='stretch')

            # VASICEK
            with tabs[3]:
                st.subheader("🎲 Vasicek Monte Carlo Simulation")
                r0, kappa, theta, sigma = yields.iloc[-1]/100, 0.20, 0.045, 0.015
                dt, n_paths = 1/252, 1000
                v_paths = np.zeros((n_paths, horizon))
                v_paths[:, 0] = r0
                for i in range(1, horizon):
                    dW = np.random.normal(0, np.sqrt(dt), n_paths)
                    v_paths[:, i] = v_paths[:, i-1] + kappa * (theta - v_paths[:, i-1]) * dt + sigma * dW
                fig_vas = go.Figure()
                v_median = np.percentile(v_paths, 50, axis=0)*100
                fig_vas.add_trace(go.Scatter(x=f_dates, y=v_median, name="Vasicek Median", line=dict(color='orange', width=3)))
                st.plotly_chart(fig_vas, width='stretch')

            # CIR MODEL (NEW TAB)
            with tabs[4]:
                st.subheader("☀️ Cox-Ingersoll-Ross (CIR) Simulation")
                st.info("The CIR model introduces a square-root term $\sigma\sqrt{r_t}$, ensuring rates remain positive.")
                c_paths = np.zeros((n_paths, horizon))
                c_paths[:, 0] = r0
                for i in range(1, horizon):
                    dW = np.random.normal(0, np.sqrt(dt), n_paths)
                    # CIR Equation: dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
                    c_paths[:, i] = c_paths[:, i-1] + kappa * (theta - c_paths[:, i-1]) * dt + sigma * np.sqrt(np.maximum(c_paths[:, i-1], 0)) * dW
                
                fig_cir = go.Figure()
                for i in range(10): 
                    fig_cir.add_trace(go.Scatter(x=f_dates, y=c_paths[i, :]*100, mode='lines', line=dict(width=1, color='rgba(0,33,71,0.2)'), showlegend=False))
                c_median = np.percentile(c_paths, 50, axis=0)*100
                fig_cir.add_trace(go.Scatter(x=f_dates, y=c_median, name="CIR Median Path", line=dict(color='green', width=3)))
                fig_cir.update_layout(template="plotly_white", yaxis_title="Yield (%)")
                st.plotly_chart(fig_cir, width='stretch')

            # METRICS
            with tabs[7]:
                z_score = stats.norm.ppf(conf_level)
                latest_vol_daily = garch_fit.conditional_volatility.iloc[-1]
                var_val = latest_vol_daily * z_score
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                c2.metric("ARIMA Forecast", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("CIR Median", f"{c_median[-1]:.3f}%")
                c4.metric("Daily VaR", f"{var_val:.3f}%")

            # EXPORT
            with tabs[8]:
                export_df = pd.DataFrame({"Date": f_dates, "ARIMA": arima_fc, "Vasicek": v_median, "CIR": c_median})
                st.dataframe(export_df, width='stretch')
                st.download_button("📥 Export Models (CSV)", export_df.to_csv().encode('utf-8'), "multi_model_report.csv")

        except Exception as e: st.error(f"Execution Error: {e}")

with tabs[9]:
    st.header("🎓 Knowledge Base")
    with st.expander("❓ Why CIR over Vasicek?"):
        st.write("In Vasicek, volatility is constant. In CIR, volatility decreases as rates approach zero, making it impossible for rates to become negative.")
        

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance</p>", unsafe_allow_html=True)
