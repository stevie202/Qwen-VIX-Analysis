# VIX vs Ticket Activity Correlation Analyzer

A Streamlit application that analyzes the correlation between market volatility (VIX) and IT support ticket volume using STL decomposition and lagged correlation analysis.

## Overview

This tool explores whether market stress (measured by VIX) correlates with increased IT support demand. It allows users to:
- Upload historical VIX data and Jira ticket counts
- Filter by specific years
- Apply seasonal decomposition to ticket data
- Perform lagged correlation analysis (0-10 days)
- Visualize relationships and export results

## Prerequisites

Required Python packages:

streamlit pandas numpy statsmodels matplotlib seaborn openpyxl

## Installation

```bash
pip install streamlit pandas numpy statsmodels matplotlib seaborn openpyxl

## Usage

streamlit run app.py

## Input Files
The app requires two Excel files:

1. VIX_history.xlsx

Column: DATE (MM/DD/YYYY format)
Columns: OPEN, HIGH, LOW, CLOSE

2. Jira Daily Ticket Count.xlsx

Column: Date
Columns: ticket_count, avg_ticket_count
Key Features
Year Filtering: Select specific years for analysis
STL Deseasonalization: Optional weekly seasonality removal
Lagged Correlation: Test correlation at 0-10 day lags
Visualization: Time series plots and correlation heatmaps
Automated Insights: Dynamic interpretation based on correlation strength
Data Export: Download processed CSV results

## Interpretation Guide

### Correlation Strength:
|r| > 0.5: Strong relationship 🔥
|r| > 0.3: Moderate relationship 📈
|r| > 0.1: Weak relationship 🫤
|r| < 0.1: Negligible 📉

### Key Insight:
A strong positive correlation at short lag (0-2 days) suggests market volatility may drive IT support demand, useful for forecasting and staffing decisions.

## Notes
Analysis requires overlapping date ranges in both files
Minimum 30 days recommended for reliable results
Correlation ≠ causation, but provides valuable operational insights
