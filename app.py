# app.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("📊 VIX vs Ticket Activity Correlation Analyzer")
st.write("Upload Jira Daily Ticket Count.xlsx and VIX_history.xlsx to analyze correlation.")

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
        if 'DATE' not in vix.columns:
            st.error("VIX file must contain a 'DATE' column.")
            return None
        vix['DATE'] = pd.to_datetime(vix['DATE'], format='%m/%d/%Y', errors='coerce')

        # Load Tickets: Date in YYYY-MM-DD
        tickets = pd.read_excel(tickets_file)
        if 'Date' not in tickets.columns:
            st.error("Ticket file must contain a 'Date' column.")
            return None
        tickets['Date'] = pd.to_datetime(tickets['Date'], errors='coerce')

        # Drop invalid dates
        vix = vix.dropna(subset=['DATE'])
        tickets = tickets.dropna(subset=['Date'])

        if vix.empty:
            st.error("No valid dates in VIX data after parsing.")
            return None
        if tickets.empty:
            st.error("No valid dates in ticket data after parsing.")
            return None

        # Rename and merge
        vix = vix.rename(columns={'DATE': 'Date'})
        merged = pd.merge(vix, tickets, on='Date', how='inner')

        if merged.empty:
            st.error("No overlapping dates found between the two files.")
            return None

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
        st.error(f"Error loading or processing data: {e}")
        return None

# Load data
data = load_data(uploaded_vix, uploaded_tickets)

if data is None:
    st.stop()

st.success("✅ Data loaded and merged successfully!")
st.write(f"📅 Full date range: {data.index.min().date()} to {data.index.max().date()}")
st.write(f"📊 Total overlapping days: {len(data)}")

# --- Year Slicer ---
st.subheader("📅 Select Years to Include in Analysis")

# Safety check: ensure index is datetime and not empty
if data.empty:
    st.error("❌ No data available to analyze.")
    st.stop()

try:
    available_years = sorted(data.index.year.dropna().unique())
except Exception:
    st.error("❌ Could not extract years from the date index. Please check date formatting.")
    st.stop()

if not available_years:
    st.error("❌ No valid years found in the data.")
    st.stop()

selected_years = st.multiselect(
    "Choose years to include in the analysis:",
    options=available_years,
    default=available_years  # Select all by default
)

if not selected_years:
    st.warning("Please select at least one year to proceed.")
    st.stop()

# Filter data
data_filtered = data[data.index.year.isin(selected_years)]
if data_filtered.empty:
    st.warning("⚠️ No data remains after filtering. Try selecting different years.")
    st.stop()

st.info(f"📈 Analyzing {len(data_filtered)} days from: {', '.join(map(str, sorted(selected_years)))}")

# Update data to filtered version
data = data_filtered.copy()

# Optional: Show data distribution
with st.expander("📊 Data Distribution by Year", expanded=False):
    year_counts = data.resample('Y').size()
    fig_year = plt.figure(figsize=(6, 3))
    year_counts.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.ylabel("Number of Days")
    plt.title("Number of Days per Year in Analysis")
    plt.xticks(rotation=45)
    st.pyplot(fig_year)

# --- Deseasonalize Ticket Count ---
st.subheader("🔧 Deseasonalizing Ticket Data (Weekly Seasonality)")

period = st.slider("STL Decomposition Period", min_value=5, max_value=10, value=7, step=1)
deseason_method = st.radio(
    "Choose deseasonalization target:",
    ["ticket_count", "avg_ticket_count"],
    index=0
)

data_deseason = data.copy()

try:
    stl = STL(data[deseason_method], period=period, seasonal=7)
    result = stl.fit()
    data_deseason[f'{deseason_method}_trend'] = result.trend
    data_deseason[f'{deseason_method}_resid'] = result.resid
    data_deseason[f'{deseason_method}_deseason'] = result.trend + result.resid
except Exception as e:
    st.error(f"STL decomposition failed: {e}")
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

plot_data = data_deseason[[vix_metric, target_col]].dropna()
if plot_data.empty:
    st.warning("No data available for plotting after filtering.")
else:
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

corr_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'ticket_count', 'avg_ticket_count']
valid_corr_cols = [col for col in corr_cols if col in data_deseason.columns]
corr_matrix = data_deseason[valid_corr_cols].corr()

fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title("Correlation Matrix (Raw Data)")
st.pyplot(fig3)

# --- Dynamic Analysis Summary ---
st.subheader("📊 Analysis Summary")

date_from = data.index.min().date()
date_to = data.index.max().date()
selected_metric = deseason_method
stl_period = period
vix_selected = vix_metric
corr_strength = abs(best_corr)

# Interpret strength
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

# Insight
if best_corr > 0.1:
    insight = (
        f"There is a {strength_desc} positive correlation ({best_corr:.3f}) between "
        f"**{vix_selected}** and deseasonalized **{selected_metric}** when VIX leads by {best_lag} day(s). "
        f"This suggests higher volatility precedes increased ticket volume."
    )
elif best_corr < -0.1:
    insight = (
        f"There is a {strength_desc} negative correlation ({best_corr:.3f}) between "
        f"**{vix_selected}** and **{selected_metric}**, meaning high VIX is linked to lower ticket activity {best_lag} day(s) later."
    )
else:
    insight = (
        f"No meaningful correlation was found (best = {best_corr:.3f}). "
        f"Market volatility does not appear to drive ticket volume in this period."
    )

# Display
st.markdown(f"""
- **📅 Analysis Period**: {date_from} to {date_to}
- **🔧 Deseasonalized Metric**: `{selected_metric}` using STL (period = {stl_period})
- **📈 VIX Metric Tested**: `{vix_selected}`
- **⏰ Best Lag**: {best_lag} day(s)
- **🔗 Correlation Strength**: **{best_corr:.3f}** → {strength_desc} relationship {emoji}
- **💡 Insight**: {insight}

> 💬 **Conclusion**: `{vix_selected}` at lag {best_lag} explains **{(best_corr**2)*100:.1f}%** of the variation in `{selected_metric}`.
""")

if len(data) < 30:
    st.warning(f"⚠️ Small sample size ({len(data)} days). Results may not be reliable.")

if abs(best_corr) < 0.2:
    st.info("💡 Try other VIX metrics (e.g., HIGH) or consider external factors affecting tickets.")

# --- Download Processed Data ---
st.subheader("💾 Download Processed Data")

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(data_deseason)

# Filename with selected years
year_str = "-".join(map(str, sorted(selected_years))) if len(selected_years) <= 5 else "selected"
st.download_button(
    label="Download merged & deseasonalized data as CSV",
    data=csv,
    file_name=f'vix_ticket_analysis_{year_str}.csv',
    mime='text/csv'
)

# --- Interpretation Guide ---
st.subheader("📘 How to Interpret This Analysis")
st.markdown("""
### 🔍 What This App Does

This tool explores whether market volatility (VIX) is linked to IT support ticket volume.

- VIX rises during market stress.
- Ticket volume reflects IT team load.
- We test: Does high VIX → more tickets (same day or later)?

---

### 📊 Understanding the Charts

#### 1. Correlation vs Lag
- Peak at Lag 1? High VIX today → more tickets tomorrow.
- Higher |correlation| = stronger link.

#### 2. Time Series Plot
- Normalized lines: Do peaks align after applying lag?
- Look for co-movement.

#### 3. Correlation Matrix
- Red = positive, Blue = negative.
- Shows all pairwise relationships.

---

### 🧹 Why Deseasonalize?
- Tickets often drop on weekends.
- STL removes weekly patterns to expose true anomalies.

---

### 📌 Key Takeaway
A strong (>0.3), positive correlation at a short lag (0–2 days) suggests:
> “Market volatility may drive IT support demand.”

Use this to:
- Forecast busy days
- Justify staffing during volatile periods
- Link IT load to business events

---

### 🛠️ Tips
- Try `HIGH` if `CLOSE` shows weak correlation
- Adjust STL period if your data has longer cycles
- Update monthly

> Note: Correlation ≠ causation — but it’s a powerful clue.
""")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Options IT")
