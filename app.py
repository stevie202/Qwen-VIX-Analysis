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

data = load_data(uploaded_vix, uploaded_tickets)
if data is None:
    st.stop()

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

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Options IT")
