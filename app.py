# app.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 VIX vs Ticket Activity Correlation Analyzer")
st.write("Upload `Jira Daily Ticket Count.xlsx` and `VIX_history.xlsx` to analyze correlation.")

# --- File Upload ---
uploaded_vix = st.file_uploader("Upload VIX_history.xlsx", type="xlsx")
uploaded_tickets = st.file_uploader("Upload Jira Daily Ticket Count.xlsx", type="xlsx")

if uploaded_vix is None or uploaded_tickets is None:
    st.info("Please upload both files to proceed.")
    st.stop()

# --- Load Data ---
@st.cache_data
def load_data(vix_file, tickets_file):
    try:
        # Load VIX: DATE in MM/DD/YYYY format
        vix = pd.read_excel(vix_file)
        vix['DATE'] = pd.to_datetime(vix['DATE'], format='%m/%d/%Y')
        vix = vix.rename(columns={'DATE': 'Date'})
        
        # Load Tickets: Date in YYYY-MM-DD
        tickets = pd.read_excel(tickets_file)
        tickets['Date'] = pd.to_datetime(tickets['Date'])
        
        # Merge on Date
        merged = pd.merge(vix, tickets, on='Date', how='inner')
        merged = merged.sort_values('Date').reset_index(drop=True)
        merged = merged.set_index('Date')
        
        # Ensure numeric types
        vix_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        ticket_cols = ['ticket_count', 'avg_ticket_count']
        for col in vix_cols + ticket_cols:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors='coerce')
        
        return merged
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

if data is None:
    st.stop()

st.success("✅ Data loaded and merged successfully!")

st.write(f"📅 Date range: {data.index.min().date()} to {data.index.max().date()}")
st.write(f"📊 {len(data)} overlapping days found.")

# --- 🔍 Year Slicer ---
st.subheader("📅 Select Years to Include in Analysis")

available_years = sorted(data.index.year.unique())
default_years = available_years  # Default: select all

selected_years = st.multiselect(
    "Choose years to include:",
    options=available_years,
    default=default_years
)

if not selected_years:
    st.warning("Please select at least one year.")
    st.stop()

# Filter data by selected years
data_filtered = data[data.index.year.isin(selected_years)]
st.info(f"📈 Analyzing {len(data_filtered)} days from {', '.join(map(str, selected_years))}")

# Use filtered data for all downstream analysis
data = data_filtered  # Update data reference

st.success("✅ Data loaded and merged successfully!")
st.write(f"📅 Date range: **{data.index.min().date()}** to **{data.index.max().date()}**")
st.write(f"📊 {len(data)} overlapping days found.")

# --- Deseasonalize Ticket Count ---
st.subheader("🔧 Deseasonalizing Ticket Data (Weekly Seasonality)")

period = st.slider("STL Decomposition Period", min_value=5, max_value=10, value=7, step=1)

deseason_method = st.radio("Choose deseasonalization target:", 
                          ["ticket_count", "avg_ticket_count"], 
                          index=0)

data_deseason = data.copy()

try:
    stl = STL(data[deseason_method], period=period, seasonal=7)
    result = stl.fit()
    data_deseason[f'{deseason_method}_trend'] = result.trend
    data_deseason[f'{deseason_method}_resid'] = result.resid
    data_deseason[f'{deseason_method}_deseason'] = result.trend + result.resid
except Exception as e:
    st.error(f"STL failed: {e}")
    st.stop()

# --- Lagged Correlation Analysis ---
st.subheader("📈 Correlation with Lag Analysis")

vix_metric = st.selectbox("Select VIX Metric", ['CLOSE', 'HIGH', 'OPEN', 'LOW'])

max_lag = st.slider("Max Lag (days)", min_value=0, max_value=10, value=5)

correlations = []
lags = list(range(max_lag + 1))

target_col = f'{deseason_method}_deseason'

for lag in lags:
    vix_lagged = data_deseason[vix_metric].shift(lag)
    corr = vix_lagged.corr(data_deseason[target_col])
    correlations.append(corr)

# Find best lag
best_lag = lags[np.argmax(correlations)]
best_corr = max(correlations)

st.metric(label="Best Correlation", value=f"{best_corr:.3f}", delta=f"at Lag {best_lag} days")

# Plot correlation vs lag
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(lags, correlations, marker='o', color='tab:blue')
ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
ax1.set_xlabel("Lag (days)")
ax1.set_ylabel("Correlation")
ax1.set_title(f"Correlation: {vix_metric} vs {target_col} (by Lag)")
ax1.grid(alpha=0.3)
st.pyplot(fig1)

# --- Time Series Plot ---
st.subheader("📉 Time Series Comparison")

# Normalize for plotting
plot_data = data_deseason[[vix_metric, target_col]].dropna()
plot_data_norm = (plot_data - plot_data.mean()) / plot_data.std()

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(plot_data_norm.index, plot_data_norm[vix_metric], label=f"{vix_metric} (normalized)", color='red')
ax2.plot(plot_data_norm.index, plot_data_norm[target_col], label=f"{target_col} (normalized)", color='blue', alpha=0.8)
ax2.legend()
ax2.set_title("Normalized VIX vs Deseasonalized Ticket Metric")
ax2.set_ylabel("Standardized Value")
ax2.grid(alpha=0.3)
st.pyplot(fig2)

# --- Correlation Matrix ---
st.subheader("🧮 Full Correlation Matrix")

corr_matrix = data_deseason[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'ticket_count', 'avg_ticket_count']].corr()
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title("Correlation Matrix (Raw Data)")
st.pyplot(fig3)

# --- Download Processed Data ---
st.subheader("💾 Download Processed Data")
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(data_deseason)
st.download_button(
    label="Download merged & deseasonalized data as CSV",
    data=csv,
    file_name='vix_ticket_analysis.csv',
    mime='text/csv'
)

# --- 📊 Dynamic Analysis Summary ---
st.subheader("📊 Analysis Summary")

# Gather key results
date_from = data.index.min().date()
date_to = data.index.max().date()
selected_metric = deseason_method
deseason_col = f'{deseason_method}_deseason'
stl_period = period
vix_selected = vix_metric
corr_strength = abs(best_corr)
lag_direction = "same day" if best_lag == 0 else f"{best_lag} day(s) later"

# Interpret correlation strength
if corr_strength > 0.5:
    strength_desc = "strong"
    emoji = "🔥"
elif corr_strength > 0.3:
    strength_desc = "moderate"
    emoji = "📈"
elif corr_strength > 0.1:
    strength_desc = "weak"
    emoji = "🫤"
else:
    strength_desc = "negligible"
    emoji = "📉"

# Generate insight
if best_corr > 0.1:
    insight = (
        f"There is a {strength_desc} positive correlation ({best_corr:.3f}) between "
        f"**{vix_selected}** and deseasonalized **{selected_metric}** when VIX leads by {best_lag} day(s). "
        f"This suggests that higher market volatility tends to precede an increase in ticket volume {lag_direction}."
    )
elif best_corr < -0.1:
    insight = (
        f"There is a {strength_desc} negative correlation ({best_corr:.3f}) between "
        f"**{vix_selected}** and deseasonalized **{selected_metric}**, meaning higher VIX values are linked to "
        f"lower ticket activity {lag_direction}. This may indicate reduced reporting during volatile periods."
    )
else:
    insight = (
        f"No meaningful correlation was found (best = {best_corr:.3f}). "
        f"Market volatility (VIX) does not appear to be strongly linked to ticket volume changes in this dataset."
    )

# Display summary
st.markdown(f"""
- **📅 Date Range**: {date_from} to {date_to}
- **🔧 Deseasonalized Metric**: `{selected_metric}` using STL (period = {stl_period})
- **📈 VIX Metric Tested**: `{vix_selected}`
- **⏰ Best Lag**: {best_lag} day(s)
- **🔗 Correlation Strength**: **{best_corr:.3f}** → {strength_desc} relationship {emoji}
- **💡 Insight**: {insight}

> 💬 **Conclusion**: {vix_selected} at lag {best_lag} explains approximately **{best_corr**2*100:.1f}%** of the variation in deseasonalized {selected_metric}.
""")

# Optional: Add a warning if low overlap or weak correlation
if len(data) < 30:
    st.warning(f"⚠️ Caution: Only {len(data)} overlapping days available — consider extending data coverage for more reliable results.")

if abs(best_corr) < 0.2:
    st.info("💡 Tip: Try other VIX metrics (e.g., HIGH) or check if ticket patterns respond to external events beyond market volatility.")

# --- 📊 Dynamic Analysis Summary ---
st.subheader("📘 How to Interpret This Analysis")

st.markdown("""
### 🔍 What This App Does

This tool helps you explore whether **market volatility (measured by the VIX)** is related to **IT support ticket volume** in your organization.

The VIX (Volatility Index) rises when markets are uncertain or fearful.  
Your ticket data shows how busy your support team is each day.

We're looking for patterns:  
👉 *Do more tickets come in when the market is volatile?*  
👉 *Does volatility today affect tickets tomorrow?*

---

### 📊 Understanding the Charts

#### 1. **Correlation vs Lag (Line Chart)**
- Shows how strongly the **VIX** correlates with ticket activity at different time lags.
- A peak at **Lag 1** means:  
  _"High VIX today → More tickets **tomorrow**"_
- The higher the correlation (closer to 1 or -1), the stronger the link.

#### 2. **Time Series Comparison (Line Chart)**
- Plots **normalized VIX** and **de-seasonalized ticket count** on the same scale.
- Look for peaks that line up — especially after applying the best lag.
- If both lines rise and fall together, there may be a connection.

#### 3. **Correlation Matrix (Heatmap)**
- Shows pairwise correlations between all variables (e.g., VIX Close vs Ticket Count).
- 🔥 **Red** = strong positive correlation  
  🟦 **Blue** = negative or weak correlation
- Helps spot which VIX metric (e.g., `HIGH` vs `CLOSE`) matters most.

---

### 🧹 Why Deseasonalize?
- Ticket volume often follows weekly patterns (e.g., fewer tickets on weekends).
- We remove these predictable patterns so we can focus on **unusual spikes** — like those possibly caused by market stress.

---

### 📌 Key Takeaway
If the **best correlation is strong (e.g., > 0.25)** and happens at a **reasonable lag (0–2 days)**, it suggests:
> _“Market volatility may be driving IT support demand.”_

This could help you:
- Forecast busy days
- Allocate support staff proactively
- Show business impact of external market events

---

### 🛠️ Tips for Use
- Try different **VIX metrics** (e.g., `HIGH` might matter more than `CLOSE`)
- Adjust the **STL period** if your data has longer cycles
- Upload new data monthly to keep insights fresh

**Note**: Correlation ≠ causation — but it’s a great starting point!
""")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Options IT")
