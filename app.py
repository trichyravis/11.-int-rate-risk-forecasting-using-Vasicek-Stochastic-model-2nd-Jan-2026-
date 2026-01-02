
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
tabs = st.tabs(["ℹ️ About", "📈 Forecast (ARIMA)", "🌪️ GARCH Volatility", "🎲 Stochastic (Vasicek)", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Hub"])

with tabs[0]:
    st.header("📖 Institutional Research Methodology")
    st.markdown("### About this Platform")
    st.write("Designed by Prof. V. Ravichandran to bridge academic theory with institutional practice. This terminal uses ARIMA for technical pathing, GARCH for risk regimes, and Vasicek for stochastic equilibrium.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📑 Assumptions")
        st.markdown("- **Stationarity:** differencing ($d=1$) for mean stabilization.\n- **Clustering:** Time-varying variance.\n- **Reversion:** Stochastic drift toward equilibrium.")
    with col2:
        st.subheader("⚠️ Limitations")
        st.markdown("- **Exogenous Shocks:** No 'Black Swan' detection.\n- **Univariate:** Historical price-action focus.")

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
                st.markdown(f'<div class="config-box"><strong>Active Model:</strong> ARIMA{model_arima.order} | Horizon: {horizon} Days</div>', unsafe_allow_html=True)
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Historical"))
                fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(dash='dot', color='orange')))
                fig_f.update_layout(template="plotly_white", title="Yield Directional Path")
                st.plotly_chart(fig_f, width='stretch')

            with tabs[2]: # GARCH
                st.subheader("🌪️ Conditional Volatility (GARCH 1,1)")
                fig_v = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red'), name="Ann. Volatility"))
                fig_v.update_layout(template="plotly_white")
                st.plotly_chart(fig_v, width='stretch')

            with tabs[3]: # VASICEK
                st.subheader("🎲 Vasicek Monte Carlo Simulation")
                r0, kappa, theta, sigma = yields.iloc[-1]/100, 0.20, 0.045, 0.015
                dt, n_paths = 1/252, 1000
                sim_paths = np.zeros((n_paths, horizon))
                sim_paths[:, 0] = r0
                for i in range(1, horizon):
                    dW = np.random.normal(0, np.sqrt(dt), n_paths)
                    sim_paths[:, i] = sim_paths[:, i-1] + kappa * (theta - sim_paths[:, i-1]) * dt + sigma * dW
                fig_vas = go.Figure()
                for i in range(10): fig_vas.add_trace(go.Scatter(x=f_dates, y=sim_paths[i, :]*100, mode='lines', line=dict(width=1, color='rgba(0,33,71,0.2)'), showlegend=False))
                v_median = np.percentile(sim_paths, 50, axis=0)*100
                fig_vas.add_trace(go.Scatter(x=f_dates, y=v_median, name="Vasicek Median", line=dict(color='orange', width=3)))
                st.plotly_chart(fig_vas, width='stretch')

            with tabs[4]: # BACKTESTING
                st.subheader("🧪 30-Day Walk-Forward Validation")
                train, test = yields.iloc[:-30], yields.iloc[-30:]
                bt_model = pm.auto_arima(train, seasonal=False)
                bt_fc = bt_model.predict(n_periods=30)
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=test.index, y=test, name="Realized"))
                fig_bt.add_trace(go.Scatter(x=test.index, y=bt_fc, name="Predicted", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig_bt, width='stretch')
                st.success(f"**Mean Absolute Error (MAE):** {np.mean(np.abs(test.values - bt_fc.values)):.4f}")

            with tabs[5]: # DIAGNOSTICS
                st.subheader("🔍 ARIMA Residual Analysis")
                fig_diag = go.Figure(go.Scatter(y=model_arima.resid(), mode='lines', line=dict(color='gray')))
                fig_diag.update_layout(title="Standardized Residuals (White Noise Check)", template="plotly_white")
                st.plotly_chart(fig_diag, width='stretch')

            with tabs[6]: # METRICS & VAR GRAPH
                st.subheader(f"📊 Quantitative Risk Metrics (α={conf_level})")
                z_score = stats.norm.ppf(conf_level)
                latest_vol_daily = garch_fit.conditional_volatility.iloc[-1]
                var_val = latest_vol_daily * z_score
                es_val = latest_vol_daily * (stats.norm.pdf(z_score)/(1-conf_level))
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                c2.metric("ARIMA Forecast", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Vasicek Median", f"{v_median[-1]:.3f}%")
                c4.metric("Daily VaR", f"{var_val:.3f}%")

                x_d = np.linspace(-4, 4, 200)
                y_d = stats.norm.pdf(x_d, 0, 1)
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=x_d, y=y_d, fill='tozeroy', name='Standard Normal', line=dict(color=CORPORATE_BLUE)))
                fig_r.add_trace(go.Scatter(x=x_d[x_d < -z_score], y=y_d[x_d < -z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', name='Tail Risk Zone'))
                fig_r.update_layout(title="Tail Risk Visualization (VaR Zone)", template="plotly_white")
                st.plotly_chart(fig_r, width='stretch')

            with tabs[7]: # EXPORT
                st.subheader("📋 Data Export Terminal")
                export_df = pd.DataFrame({
                    "Date": f_dates, 
                    "ARIMA_Forecast": arima_fc,
                    "Vasicek_Median": v_median
                })
                st.dataframe(export_df, width='stretch')
                st.download_button("📥 Download Full Report (CSV)", export_df.to_csv().encode('utf-8'), f"{ticker}_report.csv")

        except Exception as e: st.error(f"Execution Error: {e}")

# --- EDUCATIONAL Q&A HUB ---
with tabs[8]:
    st.header("🎓 Quantitative Knowledge Base")
    with st.expander("❓ ARIMA Model Selection"):
        st.write("ARIMA captures technical momentum. We use the Box-Jenkins methodology (Identification, Estimation, Diagnostics) to stabilize the mean and predict directional paths.")
    with st.expander("❓ Vasicek Mean Reversion"):
        st.write("Stochastic models like Vasicek assume rates revert to an equilibrium level (theta) over time. This is modeled using Brownian motion and mean-reversion speed (kappa).")
    with st.expander("❓ Understanding Tail Risk (VaR & ES)"):
        st.write("Value-at-Risk (VaR) identifies the threshold loss, while Expected Shortfall (ES) measures the average loss in the worst-case scenarios beyond that threshold.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance</p>", unsafe_allow_html=True)
