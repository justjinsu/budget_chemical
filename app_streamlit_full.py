#!/usr/bin/env python3
"""
Streamlit Web Application: Korean Carbon Budget Calculator (Full Version)

Interactive web interface with both modules:
- Module 1: Korean Budget Calculator
- Module 2: Annual Pathway Allocator

Navigate using the sidebar tabs.
"""

import streamlit as st
import sys
from pathlib import Path
import yaml
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from budget_chemical.modules.budget_calculator import KoreaBudgetCalculator
from budget_chemical.modules.pathway_allocator import PathwayAllocator

# Page configuration
st.set_page_config(
    page_title="Korean Carbon Budget Tool",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🌍 Carbon Budget Tool")
page = st.sidebar.radio(
    "Select Module:",
    ["📊 Module 1: Budget Calculator", "📈 Module 2: Pathway Allocator", "🔗 Integrated Workflow"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Version**: 1.0
**Data Updated**: Oct 2025
**Modules**: 2
""")

# ==================== MODULE 1 PAGE ====================
if page == "📊 Module 1: Budget Calculator":
    st.title("📊 Module 1: Korean Carbon Budget Calculator")
    st.markdown("""
    Calculate Korea's fair share of the global carbon budget using the **BKIR formula**:

    `Korea_Budget = Global_Budget × (w_r × δ_r + w_c × δ_c + w_e × δ_e)`

    **Verified Allocation Factors**:
    - Responsibility: 1.09% (Statista 2021)
    - Capability: 1.47% GDP PPP (IMF 2023)
    - Equality: 0.646% population (Worldometer 2024)
    """)

    # Configuration in sidebar
    st.sidebar.header("⚙️ Module 1 Settings")

    n_draws = st.sidebar.slider("Monte Carlo Draws", 100, 5000, 1000, 100)
    seed = st.sidebar.number_input("Random Seed", value=123, step=1)

    # Weights
    st.sidebar.subheader("BKIR Weights")
    weight_r = st.sidebar.slider("Responsibility", 0.0, 1.0, 0.3, 0.05)
    weight_c = st.sidebar.slider("Capability", 0.0, 1.0, 0.4, 0.05)
    weight_e = st.sidebar.slider("Equality", 0.0, 1.0, 0.3, 0.05)

    total_weight = weight_r + weight_c + weight_e
    if abs(total_weight - 1.0) > 0.01:
        st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.2f}, will be normalized")

    # Run Module 1
    if st.sidebar.button("🚀 Calculate Budget", type="primary"):
        config = {
            'seed': seed,
            'n_draws': n_draws,
            'global_budget': {
                '1p5C': {'low': 4e11, 'mid': 5e11, 'high': 6.7e11},
                '2p0C': {'low': 1.05e12, 'mid': 1.15e12, 'high': 1.29e12}
            },
            'user_weights': {
                'responsibility': weight_r / total_weight,
                'capability': weight_c / total_weight,
                'equality': weight_e / total_weight
            },
            'uncertainty': {
                'responsibility': {'low': 0.0095, 'mid': 0.0109, 'high': 0.0128},
                'capability': {'mu': 0.0147, 'sd_pct': 0.05},
                'equality': {'mu': 0.00646, 'sd_pct': 0.03}
            },
            'industry_fraction': 0.37,
            'petrochem_fraction': 0.10
        }

        with st.spinner("Running Monte Carlo simulation..."):
            calculator = KoreaBudgetCalculator(config)
            results = calculator.run_monte_carlo(n_draws=n_draws, scenario_mode='mixed')

        st.session_state['module1_results'] = results
        st.session_state['calculator'] = calculator
        st.success("✅ Budget calculation complete!")

    # Display results
    if 'module1_results' in st.session_state:
        results = st.session_state['module1_results']
        stats = results['statistics']

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Korea Budget (Median)", f"{stats['korea_total']['median']/1e9:.2f} Gt CO₂")
        with col2:
            st.metric("Industry Sector", f"{stats['industry']['median']/1e9:.2f} Gt CO₂")
        with col3:
            st.metric("Petrochemical", f"{stats['petrochem']['median']/1e6:.0f} Mt CO₂")

        # Distribution plot
        budgets_df = pd.DataFrame({
            'Korea': results['budgets']['korea_total'] / 1e9,
            'Industry': results['budgets']['industry'] / 1e9,
            'Scenario': results['samples']['scenarios']
        })

        fig = px.box(budgets_df, x='Scenario', y='Industry',
                    color='Scenario',
                    labels={'Industry': 'Industry Budget (Gt CO₂)'},
                    title="Budget Distribution by Climate Scenario")
        st.plotly_chart(fig, use_container_width=True)

        # Data export
        col1, col2 = st.columns(2)
        with col1:
            csv = budgets_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "budgets.csv", "text/csv")

# ==================== MODULE 2 PAGE ====================
elif page == "📈 Module 2: Pathway Allocator":
    st.title("📈 Module 2: Annual Pathway Allocator")
    st.markdown("""
    Generate annual emission pathways from a carbon budget.

    **Features**:
    - 8 different curve types
    - Automatic budget matching
    - Feasibility validation
    - True net-zero by 2050
    """)

    # Configuration
    st.sidebar.header("⚙️ Module 2 Settings")

    # Budget input
    budget_source = st.sidebar.radio("Budget Source", ["Manual Input", "From Module 1"])

    if budget_source == "Manual Input":
        budget = st.sidebar.number_input("Carbon Budget (Gt CO₂)", 0.1, 10.0, 2.14, 0.01) * 1e9
    else:
        if 'module1_results' not in st.session_state:
            st.warning("⚠️ Run Module 1 first to use calculated budget!")
            budget = 2.14e9
        else:
            stats = st.session_state['module1_results']['statistics']
            budget = stats['industry']['median']
            st.sidebar.info(f"Using Module 1 median: {budget/1e9:.2f} Gt CO₂")

    # Pathway settings
    start_year = st.sidebar.number_input("Start Year", 2020, 2030, 2024)
    end_year = st.sidebar.number_input("End Year", 2040, 2060, 2050)
    start_emission = st.sidebar.number_input("Start Emission (Mt CO₂/year)", 50, 500, 185) * 1e6

    curve_type = st.sidebar.selectbox(
        "Curve Type",
        ["exponential", "logarithmic", "s_curve", "linear", "plateau",
         "convex", "early_action", "delayed_action"]
    )

    compare_all = st.sidebar.checkbox("Compare All Curves")

    # Run Module 2
    if st.sidebar.button("📈 Generate Pathway", type="primary"):
        with st.spinner("Generating emission pathway..."):
            allocator = PathwayAllocator(
                start_year=start_year,
                end_year=end_year,
                start_emission=start_emission,
                config={'max_annual_reduction_rate': 0.15}  # Relax constraint slightly
            )

            if compare_all:
                comparison = allocator.compare_curves(budget)
                st.session_state['comparison'] = comparison
                st.session_state['allocator'] = allocator
                st.session_state['budget'] = budget
            else:
                result = allocator.allocate_budget(budget, curve_type, validate=True)
                st.session_state['pathway_result'] = result
                st.session_state['allocator'] = allocator

        st.success("✅ Pathway generated!")

    # Display results
    if compare_all and 'comparison' in st.session_state:
        st.subheader("Curve Comparison")
        st.dataframe(st.session_state['comparison'], use_container_width=True)

        # Plot all pathways
        fig = go.Figure()
        allocator = st.session_state['allocator']
        budget = st.session_state['budget']

        for _, row in st.session_state['comparison'].iterrows():
            if row['budget_error_pct'] < 15:  # Only show reasonable matches
                result = allocator.allocate_budget(budget, row['curve_type'], validate=False)
                fig.add_trace(go.Scatter(
                    x=result['years'],
                    y=result['pathway'] / 1e6,
                    name=row['curve_type'],
                    mode='lines'
                ))

        fig.update_layout(
            title="Emission Pathways - Curve Comparison",
            xaxis_title="Year",
            yaxis_title="Emissions (Mt CO₂/year)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif 'pathway_result' in st.session_state:
        result = st.session_state['pathway_result']

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Budget", f"{result['budget_allocated']/1e9:.2f} Gt")
        with col2:
            st.metric("Start Emission", f"{result['pathway'][0]/1e6:.0f} Mt/yr")
        with col3:
            st.metric("End Emission", f"{result['pathway'][-1]/1e3:.0f} kt/yr")
        with col4:
            reduction = (1 - result['pathway'][-1]/result['pathway'][0]) * 100
            st.metric("Total Reduction", f"{reduction:.1f}%")

        # Validation status
        if result['validation']['is_valid']:
            st.success("✅ Pathway is valid and feasible")
        else:
            st.warning(f"⚠️ Validation issues: {', '.join(result['validation']['issues'])}")

        # Plot pathway
        fig = go.Figure()

        # Annual emissions
        fig.add_trace(go.Scatter(
            x=result['years'],
            y=result['pathway'] / 1e6,
            name='Annual Emissions',
            mode='lines',
            line=dict(color='blue', width=2)
        ))

        # Cumulative
        cumulative = np.cumsum(result['pathway']) / 1e9
        fig.add_trace(go.Scatter(
            x=result['years'],
            y=cumulative,
            name='Cumulative Emissions',
            mode='lines',
            line=dict(color='green', width=2),
            yaxis='y2'
        ))

        fig.update_layout(
            title=f"Emission Pathway - {result['curve_type'].title()} Curve",
            xaxis_title="Year",
            yaxis_title="Annual Emissions (Mt CO₂/year)",
            yaxis2=dict(title="Cumulative (Gt CO₂)", overlaying='y', side='right'),
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Export
        pathway_df = pd.DataFrame({
            'year': result['years'],
            'emissions_tco2': result['pathway'],
            'emissions_mtco2': result['pathway'] / 1e6,
            'cumulative_gtco2': np.cumsum(result['pathway']) / 1e9
        })

        csv = pathway_df.to_csv(index=False)
        st.download_button("Download Pathway CSV", csv, f"pathway_{result['curve_type']}.csv", "text/csv")

# ==================== INTEGRATED WORKFLOW ====================
else:  # Integrated workflow
    st.title("🔗 Integrated Workflow: Budget → Pathway")
    st.markdown("""
    **Complete workflow**: Calculate Korea's budget (Module 1) → Generate emission pathways (Module 2)

    This workflow shows the full chain from global equity-based allocation to annual implementation pathways.
    """)

    # Step 1: Budget calculation
    st.header("Step 1: Calculate Budget")

    if st.button("Run Budget Calculation (Module 1)"):
        config = {
            'seed': 123,
            'n_draws': 500,
            'global_budget': {
                '1p5C': {'low': 4e11, 'mid': 5e11, 'high': 6.7e11},
                '2p0C': {'low': 1.05e12, 'mid': 1.15e12, 'high': 1.29e12}
            },
            'user_weights': {'responsibility': 0.3, 'capability': 0.4, 'equality': 0.3},
            'uncertainty': {
                'responsibility': {'low': 0.0095, 'mid': 0.0109, 'high': 0.0128},
                'capability': {'mu': 0.0147, 'sd_pct': 0.05},
                'equality': {'mu': 0.00646, 'sd_pct': 0.03}
            },
            'industry_fraction': 0.37,
            'petrochem_fraction': 0.10
        }

        with st.spinner("Calculating budgets..."):
            calculator = KoreaBudgetCalculator(config)
            results = calculator.run_monte_carlo(n_draws=500, scenario_mode='mixed')

        st.session_state['integrated_budget'] = results
        st.success("✅ Budget calculated!")

    # Step 2: Generate pathways
    if 'integrated_budget' in st.session_state:
        st.header("Step 2: Generate Pathways")

        results = st.session_state['integrated_budget']
        stats = results['statistics']

        # Show budget results
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**1.5°C Budget**: {stats['by_scenario']['1p5C']['industry']['median']/1e9:.2f} Gt CO₂")
        with col2:
            st.info(f"**2.0°C Budget**: {stats['by_scenario']['2p0C']['industry']['median']/1e9:.2f} Gt CO₂")

        # Generate pathways for both scenarios
        if st.button("Generate Pathways for Both Scenarios"):
            allocator = PathwayAllocator(start_year=2024, end_year=2050, start_emission=185e6)

            pathways = {}
            for scenario in ['1p5C', '2p0C']:
                budget = stats['by_scenario'][scenario]['industry']['median']
                result = allocator.allocate_budget(budget, 'exponential', validate=True)
                pathways[scenario] = result

            st.session_state['integrated_pathways'] = pathways
            st.success("✅ Pathways generated for both scenarios!")

    # Display integrated results
    if 'integrated_pathways' in st.session_state:
        st.header("Results: Integrated Analysis")

        pathways = st.session_state['integrated_pathways']

        # Plot both pathways
        fig = go.Figure()

        for scenario, result in pathways.items():
            temp = "1.5°C" if scenario == "1p5C" else "2.0°C"
            fig.add_trace(go.Scatter(
                x=result['years'],
                y=result['pathway'] / 1e6,
                name=f"{temp} Scenario",
                mode='lines',
                line=dict(width=3)
            ))

        fig.update_layout(
            title="Emission Pathways: 1.5°C vs 2.0°C Scenarios",
            xaxis_title="Year",
            yaxis_title="Emissions (Mt CO₂/year)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        comparison_data = []
        for scenario, result in pathways.items():
            temp = "1.5°C" if scenario == "1p5C" else "2.0°C"
            comparison_data.append({
                'Scenario': temp,
                'Budget (Gt CO₂)': result['budget_allocated'] / 1e9,
                'Start (Mt/yr)': result['pathway'][0] / 1e6,
                'End (kt/yr)': result['pathway'][-1] / 1e3,
                'Reduction (%)': (1 - result['pathway'][-1]/result['pathway'][0]) * 100,
                '2035 (Mt/yr)': result['pathway'][11] / 1e6  # 2035 is index 11
            })

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Start with Module 1 to calculate budgets, then use Module 2 to generate pathways!")
