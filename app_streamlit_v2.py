#!/usr/bin/env python3
"""
Streamlit V2: Korean Carbon Budget Calculator with Customizable Allocation Factors

NEW FEATURES:
- Fully customizable allocation factors (numerator/denominator)
- Country presets (Korea, USA, China, India, Germany, Japan)
- Real-time calculation of shares
- Comparison with verified research values
- Export custom configurations
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from budget_chemical.modules.budget_calculator import KoreaBudgetCalculator
from budget_chemical.modules.pathway_allocator import PathwayAllocator
from budget_chemical.modules.allocation_calculator import AllocationCalculator

# Page configuration
st.set_page_config(
    page_title="Carbon Budget Calculator V2",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'allocation_calc' not in st.session_state:
    st.session_state.allocation_calc = AllocationCalculator()

# Sidebar navigation
st.sidebar.title("🌍 Carbon Budget Tool V2")
st.sidebar.markdown("**NEW**: Customizable Allocation Factors!")

page = st.sidebar.radio(
    "Select Module:",
    ["⚙️ Configure Allocation Factors", "📊 Calculate Budget", "📈 Generate Pathways", "📋 Summary Report"]
)

st.sidebar.markdown("---")

# ==================== ALLOCATION FACTOR CONFIGURATION ====================
if page == "⚙️ Configure Allocation Factors":
    st.title("⚙️ Configure Allocation Factors")
    st.markdown("""
    Customize the allocation factors by adjusting **numerators** (country values) and
    **denominators** (global values). Our verified research values are loaded as defaults.
    """)

    # Country preset selector
    col1, col2 = st.columns([2, 1])
    with col1:
        country_preset = st.selectbox(
            "Load Country Preset",
            ["KOR (Korea - Default)", "USA (United States)", "CHN (China)",
             "IND (India)", "DEU (Germany)", "JPN (Japan)", "Custom"]
        )

    with col2:
        if st.button("Load Preset"):
            country_code = country_preset.split()[0]
            if country_code != "Custom":
                preset = st.session_state.allocation_calc.get_preset_country(country_code)
                st.session_state.country_preset = preset
                st.success(f"✅ Loaded {country_code} preset!")

    st.markdown("---")

    # Three columns for three factors
    st.subheader("Customize Allocation Factors")

    tab1, tab2, tab3 = st.tabs(["📜 Responsibility", "💰 Capability", "👥 Equality"])

    # ===== RESPONSIBILITY TAB =====
    with tab1:
        st.markdown("### Responsibility Factor (Historical Emissions)")
        st.info("**Formula**: Country Cumulative Emissions / World Cumulative Emissions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Numerator (Country)")
            country_cumulative = st.number_input(
                "Cumulative Emissions (Gt CO₂, 1850-2021)",
                min_value=0.1,
                max_value=1000.0,
                value=27.0,
                step=0.1,
                help="Korea default: 27.0 Gt (Source: Statista 2021)"
            )
            st.caption("🔍 Source: Statista 2021, Global Carbon Project")

        with col2:
            st.markdown("#### Denominator (World)")
            world_cumulative = st.number_input(
                "World Cumulative Emissions (Gt CO₂, 1850-2021)",
                min_value=1000.0,
                max_value=5000.0,
                value=2500.0,
                step=10.0,
                help="Default: 2,500 Gt (Source: Carbon Brief 2021)"
            )
            st.caption("🔍 Source: Carbon Brief 2021, IPCC AR6")

        # Calculate share
        resp_share = country_cumulative / world_cumulative

        st.markdown("### Calculated Share")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Responsibility Share", f"{resp_share:.4%}")
        with col2:
            st.metric("Your Value", f"{country_cumulative:.1f} Gt")
        with col3:
            st.metric("World Total", f"{world_cumulative:.0f} Gt")

        # Comparison with verified
        verified_resp = 0.0109
        diff = (resp_share - verified_resp) / verified_resp * 100
        if abs(diff) < 1:
            st.success(f"✅ Matches verified research value (1.09%)")
        else:
            st.warning(f"⚠️ Differs from verified value by {diff:+.1f}% (verified: {verified_resp:.4%})")

    # ===== CAPABILITY TAB =====
    with tab2:
        st.markdown("### Capability Factor (Economic Capacity)")
        st.info("**Formula**: Country GDP / World GDP (PPP-adjusted recommended)")

        gdp_method = st.radio("GDP Method", ["PPP-adjusted (Recommended)", "Nominal"], horizontal=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Numerator (Country)")
            country_gdp = st.number_input(
                "GDP (Trillion USD, 2023)",
                min_value=0.1,
                max_value=50.0,
                value=2.7,
                step=0.1,
                help="Korea GDP PPP: 2.7 trillion USD (Source: IMF 2023)"
            )
            st.caption("🔍 Source: IMF World Economic Outlook 2023")

        with col2:
            st.markdown("#### Denominator (World)")
            world_gdp = st.number_input(
                "World GDP (Trillion USD, 2023)",
                min_value=100.0,
                max_value=300.0,
                value=184.26 if "PPP" in gdp_method else 105.69,
                step=1.0,
                help="World GDP PPP: 184.26 trillion USD (Source: IMF 2023)"
            )
            st.caption("🔍 Source: IMF World Economic Outlook 2023")

        # Calculate share
        cap_share = country_gdp / world_gdp

        st.markdown("### Calculated Share")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Capability Share", f"{cap_share:.4%}")
        with col2:
            st.metric("Country GDP", f"${country_gdp:.2f}T")
        with col3:
            st.metric("World GDP", f"${world_gdp:.2f}T")

        # Comparison
        verified_cap = 0.0147
        diff = (cap_share - verified_cap) / verified_cap * 100
        if abs(diff) < 1:
            st.success(f"✅ Matches verified research value (1.47% PPP)")
        else:
            st.warning(f"⚠️ Differs from verified value by {diff:+.1f}% (verified: {verified_cap:.4%})")

    # ===== EQUALITY TAB =====
    with tab3:
        st.markdown("### Equality Factor (Population)")
        st.info("**Formula**: Country Population / World Population")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Numerator (Country)")
            country_pop = st.number_input(
                "Population (Millions, 2024)",
                min_value=0.1,
                max_value=2000.0,
                value=51.7,
                step=0.1,
                help="Korea population: 51.7 million (Source: Worldometer 2024)"
            )
            st.caption("🔍 Source: UN Population Division, Worldometer 2024")

        with col2:
            st.markdown("#### Denominator (World)")
            world_pop = st.number_input(
                "World Population (Billions, 2024)",
                min_value=6.0,
                max_value=10.0,
                value=8.0,
                step=0.1,
                help="World population: 8.0 billion (Source: UN 2024)"
            )
            st.caption("🔍 Source: UN World Population Prospects 2024")

        # Calculate share
        eq_share = country_pop / (world_pop * 1000)  # Convert billions to millions

        st.markdown("### Calculated Share")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Equality Share", f"{eq_share:.4%}")
        with col2:
            st.metric("Country Pop", f"{country_pop:.1f}M")
        with col3:
            st.metric("World Pop", f"{world_pop:.2f}B")

        # Comparison
        verified_eq = 0.00646
        diff = (eq_share - verified_eq) / verified_eq * 100
        if abs(diff) < 1:
            st.success(f"✅ Matches verified research value (0.646%)")
        else:
            st.warning(f"⚠️ Differs from verified value by {diff:+.1f}% (verified: {verified_eq:.4%})")

    # Store in session state
    st.session_state.custom_factors = {
        'responsibility': resp_share,
        'capability': cap_share,
        'equality': eq_share,
        'country_values': {
            'cumulative_emissions_gt': country_cumulative,
            'gdp_ppp_trillion': country_gdp,
            'population_million': country_pop
        },
        'world_values': {
            'cumulative_emissions_gt': world_cumulative,
            'gdp_ppp_trillion': world_gdp,
            'population_billion': world_pop
        }
    }

    # Summary panel
    st.markdown("---")
    st.subheader("📊 Summary of Custom Allocation Factors")

    summary_df = pd.DataFrame({
        'Factor': ['Responsibility', 'Capability', 'Equality'],
        'Your Share (%)': [resp_share * 100, cap_share * 100, eq_share * 100],
        'Verified Share (%)': [1.09, 1.47, 0.646],
        'Difference (%)': [
            (resp_share - 0.0109) / 0.0109 * 100,
            (cap_share - 0.0147) / 0.0147 * 100,
            (eq_share - 0.00646) / 0.00646 * 100
        ]
    })

    st.dataframe(summary_df.style.format({
        'Your Share (%)': '{:.3f}',
        'Verified Share (%)': '{:.3f}',
        'Difference (%)': '{:+.1f}'
    }), use_container_width=True)

    # Export configuration
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Configuration"):
            config_json = json.dumps(st.session_state.custom_factors, indent=2)
            st.download_button(
                "Download Config JSON",
                config_json,
                "custom_allocation_config.json",
                "application/json"
            )

    with col2:
        if st.button("🔄 Reset to Verified Defaults"):
            st.session_state.custom_factors = {
                'responsibility': 0.0109,
                'capability': 0.0147,
                'equality': 0.00646
            }
            st.rerun()

# ==================== CALCULATE BUDGET ====================
elif page == "📊 Calculate Budget":
    st.title("📊 Calculate Carbon Budget")

    # Check if custom factors exist
    if 'custom_factors' not in st.session_state:
        st.warning("⚠️ Please configure allocation factors first (go to '⚙️ Configure Allocation Factors')")
        st.info("Using verified default values: R=1.09%, C=1.47%, E=0.646%")
        custom_factors = {
            'responsibility': 0.0109,
            'capability': 0.0147,
            'equality': 0.00646
        }
    else:
        custom_factors = st.session_state.custom_factors
        st.success("✅ Using your custom allocation factors!")

    # Display current factors
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Responsibility", f"{custom_factors['responsibility']:.4%}")
    with col2:
        st.metric("Capability", f"{custom_factors['capability']:.4%}")
    with col3:
        st.metric("Equality", f"{custom_factors['equality']:.4%}")

    # Budget calculation settings
    st.sidebar.header("⚙️ Budget Settings")
    n_draws = st.sidebar.slider("Monte Carlo Draws", 100, 5000, 1000)

    # Weights
    st.sidebar.subheader("BKIR Weights")
    weight_r = st.sidebar.slider("Responsibility Weight", 0.0, 1.0, 0.3, 0.05)
    weight_c = st.sidebar.slider("Capability Weight", 0.0, 1.0, 0.4, 0.05)
    weight_e = st.sidebar.slider("Equality Weight", 0.0, 1.0, 0.3, 0.05)

    total_weight = weight_r + weight_c + weight_e
    if abs(total_weight - 1.0) > 0.01:
        st.sidebar.warning(f"Weights sum to {total_weight:.2f}, will be normalized")

    # Run calculation
    if st.sidebar.button("🚀 Calculate Budget", type="primary"):
        config = {
            'seed': 123,
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
                'responsibility': {
                    'low': custom_factors['responsibility'] * 0.87,
                    'mid': custom_factors['responsibility'],
                    'high': custom_factors['responsibility'] * 1.17
                },
                'capability': {
                    'mu': custom_factors['capability'],
                    'sd_pct': 0.05
                },
                'equality': {
                    'mu': custom_factors['equality'],
                    'sd_pct': 0.03
                }
            },
            'industry_fraction': 0.37,
            'petrochem_fraction': 0.10
        }

        with st.spinner("Running Monte Carlo simulation..."):
            calculator = KoreaBudgetCalculator(config)
            results = calculator.run_monte_carlo(n_draws=n_draws, scenario_mode='mixed')

        st.session_state.budget_results = results
        st.success("✅ Budget calculated with your custom factors!")

    # Display results
    if 'budget_results' in st.session_state:
        results = st.session_state.budget_results
        stats = results['statistics']

        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Korea Budget", f"{stats['korea_total']['median']/1e9:.2f} Gt CO₂")
        with col2:
            st.metric("Industry", f"{stats['industry']['median']/1e9:.2f} Gt CO₂")
        with col3:
            st.metric("Petrochemical", f"{stats['petrochem']['median']/1e6:.0f} Mt CO₂")

        # Scenario breakdown
        if 'by_scenario' in stats:
            st.subheader("Budget by Climate Scenario")

            scenario_data = []
            for scenario, data in stats['by_scenario'].items():
                scenario_data.append({
                    'Scenario': '1.5°C' if scenario == '1p5C' else '2.0°C',
                    'Korea (Gt)': data['korea_total']['median'] / 1e9,
                    'Industry (Gt)': data['industry']['median'] / 1e9,
                    'Petrochemical (Mt)': data['petrochem']['median'] / 1e6
                })

            st.dataframe(pd.DataFrame(scenario_data), use_container_width=True)

        # Visualization
        budgets_df = pd.DataFrame({
            'Industry': results['budgets']['industry'] / 1e9,
            'Scenario': results['samples']['scenarios']
        })

        fig = px.box(budgets_df, x='Scenario', y='Industry',
                    color='Scenario',
                    title="Industry Budget Distribution by Scenario")
        st.plotly_chart(fig, use_container_width=True)

# ==================== GENERATE PATHWAYS ====================
elif page == "📈 Generate Pathways":
    st.title("📈 Generate Annual Emission Pathways")
    st.markdown("""
    Convert your calculated budget into year-by-year emission pathways (2024-2050).
    Choose from 8 different curve types or compare them all.
    """)

    # Check if budget exists
    if 'budget_results' not in st.session_state:
        st.warning("⚠️ Please calculate budget first (go to '📊 Calculate Budget')")
        st.info("You can also enter a budget manually below.")
        use_manual = True
    else:
        use_manual = st.checkbox("Use manual budget instead of calculated budget")

    # Budget selection
    if use_manual:
        st.sidebar.header("⚙️ Manual Budget Input")
        budget = st.sidebar.number_input(
            "Budget (Gt CO₂)",
            min_value=0.5,
            max_value=10.0,
            value=2.14,
            step=0.1
        ) * 1e9
        st.info(f"Using manual budget: {budget/1e9:.2f} Gt CO₂")
    else:
        stats = st.session_state.budget_results['statistics']
        scenario_choice = st.radio(
            "Select Scenario for Pathway",
            ["1.5°C Scenario", "2.0°C Scenario", "Mixed (Median)"],
            horizontal=True
        )

        if scenario_choice == "1.5°C Scenario":
            budget = stats['by_scenario']['1p5C']['industry']['median']
        elif scenario_choice == "2.0°C Scenario":
            budget = stats['by_scenario']['2p0C']['industry']['median']
        else:
            budget = stats['industry']['median']

        st.success(f"✅ Using budget from calculation: {budget/1e9:.2f} Gt CO₂")

    # Pathway settings
    st.sidebar.header("⚙️ Pathway Settings")

    curve_type = st.sidebar.selectbox(
        "Curve Type",
        ["exponential", "logarithmic", "s_curve", "linear", "plateau",
         "convex", "early_action", "delayed_action"],
        help="Select emission reduction curve shape"
    )

    compare_curves = st.sidebar.checkbox("Compare All Curves")

    start_year = st.sidebar.number_input("Start Year", 2020, 2030, 2024)
    end_year = st.sidebar.number_input("End Year", 2040, 2060, 2050)
    start_emission = st.sidebar.number_input(
        "Start Emission (Mt CO₂/year)",
        50.0, 500.0, 185.0, 1.0
    ) * 1e6

    # Generate pathway
    if st.sidebar.button("📈 Generate Pathway", type="primary"):
        with st.spinner("Generating emission pathway..."):
            allocator = PathwayAllocator(
                start_year=start_year,
                end_year=end_year,
                start_emission=start_emission,
                config={'max_annual_reduction_rate': 0.20, 'end_epsilon': 1.0}
            )

            if compare_curves:
                comparison = allocator.compare_curves(budget)
                st.session_state.pathway_comparison = comparison
                st.session_state.allocator = allocator
                st.session_state.pathway_budget = budget
            else:
                result = allocator.allocate_budget(budget, curve_type, validate=True)
                st.session_state.pathway_result = result
                st.session_state.allocator = allocator

        st.success("✅ Pathway generated!")

    # Display results
    if compare_curves and 'pathway_comparison' in st.session_state:
        st.subheader("📊 Curve Comparison")

        comparison = st.session_state.pathway_comparison
        st.dataframe(comparison.style.format({
            'budget_error_pct': '{:.2f}%',
            'initial_reduction_pct': '{:.2f}%',
            'avg_annual_reduction_pct': '{:.2f}%',
            'final_emission_tco2': '{:.0f}',
            'cumulative_tco2': '{:.2e}'
        }), use_container_width=True)

        # Plot all pathways
        st.subheader("📈 Pathway Comparison Chart")

        fig = go.Figure()
        allocator = st.session_state.allocator
        budget = st.session_state.pathway_budget

        for _, row in comparison.iterrows():
            if row['budget_error_pct'] < 20:  # Only show reasonable matches
                try:
                    result = allocator.allocate_budget(budget, row['curve_type'], validate=False)
                    fig.add_trace(go.Scatter(
                        x=result['years'],
                        y=result['pathway'] / 1e6,
                        name=row['curve_type'],
                        mode='lines',
                        line=dict(width=2)
                    ))
                except:
                    pass

        fig.update_layout(
            title="Emission Pathways - All Curve Types",
            xaxis_title="Year",
            yaxis_title="Emissions (Mt CO₂/year)",
            hovermode='x unified',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    elif 'pathway_result' in st.session_state:
        result = st.session_state.pathway_result

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Budget", f"{result['budget_allocated']/1e9:.2f} Gt")
        with col2:
            st.metric("Start (2024)", f"{result['pathway'][0]/1e6:.0f} Mt/yr")
        with col3:
            st.metric("End (2050)", f"{result['pathway'][-1]/1e3:.1f} kt/yr")
        with col4:
            reduction = (1 - result['pathway'][-1]/result['pathway'][0]) * 100
            st.metric("Reduction", f"{reduction:.1f}%")

        # Validation
        validation = result['validation']
        if validation['is_valid']:
            st.success("✅ Pathway is feasible and valid")
        else:
            with st.expander("⚠️ Validation Issues (click to expand)", expanded=False):
                for issue in validation['issues']:
                    st.warning(f"• {issue}")

        # Main pathway chart
        st.subheader(f"📊 {result['curve_type'].title()} Pathway")

        fig = go.Figure()

        # Annual emissions
        fig.add_trace(go.Scatter(
            x=result['years'],
            y=result['pathway'] / 1e6,
            name='Annual Emissions',
            mode='lines',
            line=dict(color='#2171b5', width=3),
            fill='tozeroy',
            fillcolor='rgba(33, 113, 181, 0.2)'
        ))

        fig.update_layout(
            title=f"Annual Emission Pathway ({result['curve_type'].title()} Curve)",
            xaxis_title="Year",
            yaxis_title="Emissions (Mt CO₂/year)",
            hovermode='x',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Cumulative chart
        st.subheader("📈 Cumulative Emissions")

        cumulative = np.cumsum(result['pathway'])
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=result['years'],
            y=cumulative / 1e9,
            name='Cumulative Emissions',
            mode='lines',
            line=dict(color='#238b45', width=3),
            fill='tozeroy',
            fillcolor='rgba(35, 139, 69, 0.2)'
        ))

        # Budget line
        fig2.add_hline(
            y=result['budget_allocated'] / 1e9,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Budget: {result['budget_allocated']/1e9:.2f} Gt",
            annotation_position="right"
        )

        fig2.update_layout(
            title="Cumulative Emissions vs Budget",
            xaxis_title="Year",
            yaxis_title="Cumulative Emissions (Gt CO₂)",
            hovermode='x',
            height=400
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Key milestones
        st.subheader("🎯 Key Milestones")

        milestones = []
        pathway = result['pathway']
        years = result['years']

        # Find when emissions drop below certain thresholds
        for threshold, label in [(100e6, "100 Mt/yr"), (50e6, "50 Mt/yr"), (10e6, "10 Mt/yr"), (1e6, "1 Mt/yr")]:
            idx = np.where(pathway <= threshold)[0]
            if len(idx) > 0:
                year = years[idx[0]]
                milestones.append({
                    'Milestone': f'Below {label}',
                    'Year': int(year),
                    'Emission': f'{pathway[idx[0]]/1e6:.1f} Mt/yr'
                })

        # 2035 midpoint
        idx_2035 = 11  # 2035 is 11 years from 2024
        if idx_2035 < len(pathway):
            milestones.append({
                'Milestone': '2035 (Midpoint)',
                'Year': 2035,
                'Emission': f'{pathway[idx_2035]/1e6:.1f} Mt/yr'
            })

        st.dataframe(pd.DataFrame(milestones), use_container_width=True, hide_index=True)

        # Export pathway
        st.subheader("💾 Export Pathway")

        pathway_df = pd.DataFrame({
            'year': result['years'],
            'emissions_tco2': result['pathway'],
            'emissions_mtco2': result['pathway'] / 1e6,
            'cumulative_gtco2': np.cumsum(result['pathway']) / 1e9
        })

        col1, col2 = st.columns(2)
        with col1:
            csv = pathway_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"pathway_{result['curve_type']}.csv",
                "text/csv"
            )

        with col2:
            json_str = json.dumps({
                'pathway': result['pathway'].tolist(),
                'years': result['years'].tolist(),
                'budget': result['budget_allocated'],
                'curve_type': result['curve_type'],
                'validation': result['validation']
            }, indent=2)

            st.download_button(
                "📥 Download JSON",
                json_str,
                f"pathway_{result['curve_type']}.json",
                "application/json"
            )

# ==================== SUMMARY REPORT ====================
else:  # Summary Report
    st.title("📋 Summary Report")
    st.markdown("Complete analysis summary combining all modules.")

    # Check what's available
    has_factors = 'custom_factors' in st.session_state
    has_budget = 'budget_results' in st.session_state
    has_pathway = 'pathway_result' in st.session_state

    if not (has_factors or has_budget or has_pathway):
        st.info("👈 Complete the workflow first:\n1. Configure Allocation Factors\n2. Calculate Budget\n3. Generate Pathways")
    else:
        # Section 1: Allocation Factors
        if has_factors:
            st.subheader("1️⃣ Allocation Factors")
            factors = st.session_state.custom_factors

            summary_df = pd.DataFrame({
                'Factor': ['Responsibility', 'Capability', 'Equality'],
                'Your Value': [
                    f"{factors['responsibility']:.4%}",
                    f"{factors['capability']:.4%}",
                    f"{factors['equality']:.4%}"
                ],
                'Verified Default': ['1.09%', '1.47%', '0.646%']
            })

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Section 2: Budget Results
        if has_budget:
            st.subheader("2️⃣ Carbon Budget Allocation")
            stats = st.session_state.budget_results['statistics']

            budget_summary = pd.DataFrame({
                'Sector': ['Korea Total', 'Industry', 'Petrochemical'],
                'Median (Gt CO₂)': [
                    stats['korea_total']['median'] / 1e9,
                    stats['industry']['median'] / 1e9,
                    stats['petrochem']['median'] / 1e9
                ],
                '90% CI Lower': [
                    stats['korea_total']['p05'] / 1e9,
                    stats['industry']['p05'] / 1e9,
                    stats['petrochem']['p05'] / 1e9
                ],
                '90% CI Upper': [
                    stats['korea_total']['p95'] / 1e9,
                    stats['industry']['p95'] / 1e9,
                    stats['petrochem']['p95'] / 1e9
                ]
            })

            st.dataframe(budget_summary.style.format({
                'Median (Gt CO₂)': '{:.2f}',
                '90% CI Lower': '{:.2f}',
                '90% CI Upper': '{:.2f}'
            }), use_container_width=True, hide_index=True)

        # Section 3: Pathway Results
        if has_pathway:
            st.subheader("3️⃣ Emission Pathway")
            result = st.session_state.pathway_result

            pathway_summary = pd.DataFrame({
                'Metric': [
                    'Curve Type',
                    'Budget',
                    'Start Emission (2024)',
                    'End Emission (2050)',
                    'Total Reduction',
                    'Budget Match'
                ],
                'Value': [
                    result['curve_type'].title(),
                    f"{result['budget_allocated']/1e9:.2f} Gt CO₂",
                    f"{result['pathway'][0]/1e6:.0f} Mt/yr",
                    f"{result['pathway'][-1]/1e3:.1f} kt/yr",
                    f"{(1 - result['pathway'][-1]/result['pathway'][0])*100:.1f}%",
                    f"{result['validation']['budget_error_pct']:.2f}%"
                ]
            })

            st.dataframe(pathway_summary, use_container_width=True, hide_index=True)

        # Generate full report
        if has_factors and has_budget and has_pathway:
            st.markdown("---")
            st.subheader("📄 Export Complete Report")

            if st.button("📥 Generate Full Report (JSON)"):
                full_report = {
                    'allocation_factors': st.session_state.custom_factors,
                    'budget_statistics': st.session_state.budget_results['statistics'],
                    'pathway': {
                        'years': st.session_state.pathway_result['years'].tolist(),
                        'emissions': st.session_state.pathway_result['pathway'].tolist(),
                        'curve_type': st.session_state.pathway_result['curve_type'],
                        'validation': st.session_state.pathway_result['validation']
                    },
                    'metadata': {
                        'version': '2.0',
                        'date': pd.Timestamp.now().isoformat()
                    }
                }

                report_json = json.dumps(full_report, indent=2, default=str)
                st.download_button(
                    "📥 Download Complete Report",
                    report_json,
                    "complete_analysis_report.json",
                    "application/json"
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Version 2.0**
Customizable allocation factors
Data: Oct 2025
""")
