
# INTEREST RATE FORECASTING DASHBOARD (ARIMA & VASICEK)
### The Mountain Path - World of Finance
**Developed by Prof. V. Ravichandran**

---

## 📖 Project Overview
This institutional-grade terminal is a quantitative decision-support system designed to bridge the gap between academic theory and fixed-income market practice. It provides a structured framework for analyzing sovereign debt benchmarks using both traditional time-series models and modern stochastic simulations.

The platform is specifically tailored for **MBA, CFA, and FRM students** to visualize yield paths, volatility clustering, and tail-risk metrics.



---

## 🛠️ Quantitative Frameworks

### 1. Traditional Time Series (ARIMA)
Following the **Box-Jenkins methodology**, the dashboard utilizes ARIMA (AutoRegressive Integrated Moving Average) to identify momentum and trends in interest rate data. This univariate approach is highly effective for short-term (1-4 week) directional pathing.



### 2. Volatility Modeling (GARCH)
To account for **Volatility Clustering**, the engine implements GARCH (1,1). This allows users to visualize how risk is regime-dependent, moving from periods of "tranquility" to "turbulence."

### 3. Stochastic Modeling (Vasicek)
For equilibrium-based forecasting, the terminal includes the **Vasicek Model**. Unlike ARIMA, this model incorporates **Mean Reversion**, simulating 1,000 potential paths to show the "probability cloud" of where rates gravitate toward a long-term target ($\theta$).



### 4. Risk Metrics (VaR & ES)
The dashboard calculates:
* **Value-at-Risk (VaR):** The threshold for maximum expected loss at a given confidence level.
* **Expected Shortfall (ES):** A "coherent" risk measure that calculates the average loss beyond the VaR threshold during extreme tail-risk events.



---

## 🚀 Installation & Local Deployment

To run this dashboard on your local machine:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/trichyravis/interest-rate-forecasting-dashboard.git](https://github.com/trichyravis/interest-rate-forecasting-dashboard.git)
   cd interest-rate-forecasting-dashboard
