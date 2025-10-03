#!/usr/bin/env python3
"""
Streamlit Web Application: Korean Carbon Budget Calculator

Interactive web interface for calculating Korea's carbon budget allocation.
"""

import streamlit as st
import sys
from pathlib import Path
import yaml
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from budget_chemical.modules.budget_calculator import KoreaBudgetCalculator

# Page configuration
st.set_page_config(
    page_title="Korean Carbon Budget Calculator",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("🌍 Korean Carbon Budget Calculator")
st.markdown("""
Calculate Korea's carbon budget allocation using the **BKIR formula** based on three equity principles:
- **Responsibility**: Historical cumulative emissions
- **Capability**: Economic capacity (GDP)
- **Equality**: Population share

**Data Sources (Updated Oct 2025)**:
- Responsibility: 1.09% (Statista 2021)
- Capability: 1.47% GDP PPP (IMF 2023)
- Equality: 0.646% population (Worldometer 2024)
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Monte Carlo settings
st.sidebar.subheader("Monte Carlo Settings")
n_draws = st.sidebar.slider("Number of Monte Carlo Draws", 100, 10000, 1000, 100)
seed = st.sidebar.number_input("Random Seed", value=123, step=1)

# Global budget scenarios
st.sidebar.subheader("Global Budget Scenarios (Gt CO₂)")
col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown("**1.5°C Scenario**")
    budget_1p5_low = st.number_input("Low (17%)", value=400.0, step=10.0, key="1p5_low")
    budget_1p5_mid = st.number_input("Mid (50%)", value=500.0, step=10.0, key="1p5_mid")
    budget_1p5_high = st.number_input("High (83%)", value=670.0, step=10.0, key="1p5_high")

with col2:
    st.markdown("**2.0°C Scenario**")
    budget_2p0_low = st.number_input("Low (17%)", value=1050.0, step=10.0, key="2p0_low")
    budget_2p0_mid = st.number_input("Mid (50%)", value=1150.0, step=10.0, key="2p0_mid")
    budget_2p0_high = st.number_input("High (83%)", value=1290.0, step=10.0, key="2p0_high")

# Allocation weights
st.sidebar.subheader("BKIR Formula Weights")
weight_r = st.sidebar.slider("Responsibility Weight", 0.0, 1.0, 0.3, 0.05)
weight_c = st.sidebar.slider("Capability Weight", 0.0, 1.0, 0.4, 0.05)
weight_e = st.sidebar.slider("Equality Weight", 0.0, 1.0, 0.3, 0.05)

# Normalize weights
total_weight = weight_r + weight_c + weight_e
if abs(total_weight - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.2f}, will be normalized to 1.0")

# Allocation factors (verified values)
st.sidebar.subheader("Korea's Allocation Factors")
responsibility_mid = st.sidebar.number_input("Responsibility Share (%)", value=1.09, step=0.01, format="%.2f") / 100
capability_mu = st.sidebar.number_input("Capability Share (%)", value=1.47, step=0.01, format="%.2f") / 100
equality_mu = st.sidebar.number_input("Equality Share (%)", value=0.646, step=0.001, format="%.3f") / 100

# Sector fractions
st.sidebar.subheader("Sector Allocation")
industry_fraction = st.sidebar.slider("Industry Sector (%)", 0.0, 100.0, 37.0, 1.0) / 100
petrochem_fraction = st.sidebar.slider("Petrochemical (% of Industry)", 0.0, 100.0, 10.0, 1.0) / 100

# Run button
if st.sidebar.button("🚀 Calculate Budget", type="primary"):

    # Build configuration
    config = {
        'seed': seed,
        'n_draws': n_draws,
        'global_budget': {
            '1p5C': {
                'low': budget_1p5_low * 1e9,
                'mid': budget_1p5_mid * 1e9,
                'high': budget_1p5_high * 1e9
            },
            '2p0C': {
                'low': budget_2p0_low * 1e9,
                'mid': budget_2p0_mid * 1e9,
                'high': budget_2p0_high * 1e9
            }
        },
        'user_weights': {
            'responsibility': weight_r / total_weight,
            'capability': weight_c / total_weight,
            'equality': weight_e / total_weight
        },
        'uncertainty': {
            'responsibility': {
                'low': responsibility_mid * 0.87,  # ±13% uncertainty
                'mid': responsibility_mid,
                'high': responsibility_mid * 1.17
            },
            'capability': {
                'mu': capability_mu,
                'sd_pct': 0.05
            },
            'equality': {
                'mu': equality_mu,
                'sd_pct': 0.03
            }
        },
        'industry_fraction': industry_fraction,
        'petrochem_fraction': petrochem_fraction
    }

    # Run calculation
    with st.spinner("Running Monte Carlo simulation..."):
        calculator = KoreaBudgetCalculator(config)
        results = calculator.run_monte_carlo(n_draws=n_draws, scenario_mode='mixed')

    # Store results in session state
    st.session_state['results'] = results
    st.session_state['calculator'] = calculator
    st.success("✅ Calculation complete!")

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    stats = results['statistics']

    st.header("📊 Results")

    # Key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Korea Total Budget (Median)",
            f"{stats['korea_total']['median']/1e9:.2f} Gt CO₂",
            f"Range: {stats['korea_total']['p05']/1e9:.2f} - {stats['korea_total']['p95']/1e9:.2f} Gt"
        )

    with col2:
        st.metric(
            "Industry Sector (Median)",
            f"{stats['industry']['median']/1e9:.2f} Gt CO₂",
            f"Range: {stats['industry']['p05']/1e9:.2f} - {stats['industry']['p95']/1e9:.2f} Gt"
        )

    with col3:
        st.metric(
            "Petrochemical Sector (Median)",
            f"{stats['petrochem']['median']/1e6:.0f} Mt CO₂",
            f"Range: {stats['petrochem']['p05']/1e6:.0f} - {stats['petrochem']['p95']/1e6:.0f} Mt"
        )

    # Scenario breakdown
    if 'by_scenario' in stats:
        st.subheader("Climate Scenario Breakdown")
        scenario_cols = st.columns(2)

        for idx, (scenario, scenario_stats) in enumerate(stats['by_scenario'].items()):
            temp = "1.5°C" if scenario == "1p5C" else "2.0°C"

            with scenario_cols[idx]:
                st.markdown(f"### {temp} Scenario")
                st.write(f"**Draws**: {scenario_stats['count']} ({scenario_stats['proportion']:.1%})")

                scenario_df = pd.DataFrame({
                    'Sector': ['Korea Total', 'Industry', 'Petrochemical'],
                    'Median (Gt CO₂)': [
                        scenario_stats['korea_total']['median']/1e9,
                        scenario_stats['industry']['median']/1e9,
                        scenario_stats['petrochem']['median']/1e9
                    ],
                    'P05 (Gt CO₂)': [
                        scenario_stats['korea_total']['p05']/1e9,
                        scenario_stats['industry']['p05']/1e9,
                        scenario_stats['petrochem']['p05']/1e9
                    ],
                    'P95 (Gt CO₂)': [
                        scenario_stats['korea_total']['p95']/1e9,
                        scenario_stats['industry']['p95']/1e9,
                        scenario_stats['petrochem']['p95']/1e9
                    ]
                })
                st.dataframe(scenario_df, use_container_width=True)

    # Visualizations
    st.subheader("📈 Budget Distributions")

    # Create distribution plots
    budgets_df = pd.DataFrame({
        'Korea Total': results['budgets']['korea_total'] / 1e9,
        'Industry': results['budgets']['industry'] / 1e9,
        'Petrochemical': results['budgets']['petrochem'] / 1e9,
        'Scenario': results['samples']['scenarios']
    })

    # Histogram
    fig = go.Figure()

    for sector in ['Korea Total', 'Industry', 'Petrochemical']:
        fig.add_trace(go.Histogram(
            x=budgets_df[sector],
            name=sector,
            opacity=0.7,
            nbinsx=30
        ))

    fig.update_layout(
        title="Carbon Budget Distributions",
        xaxis_title="Budget (Gt CO₂)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Scenario comparison
    st.subheader("🌡️ Scenario Comparison")

    fig2 = px.box(
        budgets_df,
        x='Scenario',
        y='Korea Total',
        color='Scenario',
        labels={'Korea Total': 'Korea Budget (Gt CO₂)', 'Scenario': 'Climate Scenario'},
        title="Budget Distribution by Climate Scenario"
    )

    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Data export
    st.subheader("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        csv = budgets_df.to_csv(index=False)
        st.download_button(
            label="Download Budget Samples (CSV)",
            data=csv,
            file_name="korea_budget_samples.csv",
            mime="text/csv"
        )

    with col2:
        import json
        json_data = json.dumps({
            'statistics': stats,
            'metadata': results['metadata']
        }, indent=2, default=str)

        st.download_button(
            label="Download Statistics (JSON)",
            data=json_data,
            file_name="korea_budget_statistics.json",
            mime="application/json"
        )

    # Summary text
    with st.expander("📄 Full Summary"):
        st.text(st.session_state['calculator'].get_summary())

else:
    st.info("👈 Configure parameters in the sidebar and click '🚀 Calculate Budget' to start")

    # Display methodology
    st.header("📖 Methodology")

    st.markdown("""
    ### BKIR Formula

    Korea's carbon budget is calculated using:

    ```
    Korea_Budget = Global_Budget × (w_r × δ_r + w_c × δ_c + w_e × δ_e)
    ```

    Where:
    - `w_r, w_c, w_e`: User-defined weights (must sum to 1.0)
    - `δ_r`: Responsibility share (historical cumulative emissions)
    - `δ_c`: Capability share (GDP share, PPP-adjusted)
    - `δ_e`: Equality share (population share)

    ### Data Sources

    | Factor | Value | Source |
    |--------|-------|--------|
    | **Responsibility** | 1.09% | Statista 2021 (~27 Gt of 2,500 Gt cumulative) |
    | **Capability** | 1.47% | IMF 2023 ($2.7T / $184.26T GDP PPP) |
    | **Equality** | 0.646% | Worldometer 2024 (51.7M / 8.0B population) |

    ### Uncertainty Quantification

    Monte Carlo simulation accounts for:
    - Global budget uncertainty (triangular distribution per scenario)
    - Allocation factor uncertainty (measurement and methodological)
    - Weight specification uncertainty (±2.5 percentage points)
    """)

    st.header("🔍 About")
    st.markdown("""
    This calculator implements climate equity principles based on:
    - UNFCCC's Common But Differentiated Responsibilities (CBDR)
    - Paris Agreement Article 2 on equity and differentiation
    - Climate Equity Reference framework

    **Version**: 1.0 (Updated with verified data October 2025)

    **Citation**: If you use this tool, please cite the data sources listed above.
    """)
