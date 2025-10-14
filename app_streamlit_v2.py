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

if 'nav_page' not in st.session_state:
    st.session_state.nav_page = "⚙️ Configure Allocation Factors"

# Sidebar navigation
st.sidebar.title("🌍 Carbon Budget Tool V2")
st.sidebar.markdown("**NEW**: Customizable Allocation Factors!")

page = st.sidebar.radio(
    "Select Module:",
    ["⚙️ Configure Allocation Factors", "📊 Calculate Budget", "📈 Generate Pathways", "📋 Summary Report"],
    key="nav_page"
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
             "IND (India)", "DEU (Germany)", "JPN (Japan)", "Custom"],
            key="country_preset_select"
        )

    with col2:
        if st.button("Load Preset"):
            country_code = country_preset.split()[0]
            if country_code != "Custom":
                preset = st.session_state.allocation_calc.get_preset_country(country_code)
                st.session_state.country_cumulative = preset.get('cumulative_emissions_gt', st.session_state.country_cumulative)
                active_method = st.session_state.get('gdp_method', "PPP-adjusted (Recommended)")
                if "PPP" in active_method:
                    st.session_state.country_gdp = preset.get('gdp_ppp_trillion', st.session_state.country_gdp)
                    st.session_state.world_gdp = defaults_world['gdp_ppp_trillion']
                else:
                    st.session_state.country_gdp = preset.get('gdp_nominal_trillion', st.session_state.country_gdp)
                    st.session_state.world_gdp = defaults_world['gdp_nominal_trillion']
                st.session_state.country_population = preset.get('population_million', st.session_state.country_population)
                st.session_state.country_preset = preset
                st.success(f"✅ Loaded {country_code} preset!")
                st.rerun()

    st.markdown("---")

    # Three columns for three factors
    st.subheader("Customize Allocation Factors")

    defaults_country = AllocationCalculator.DEFAULT_VALUES['korea']
    defaults_world = AllocationCalculator.DEFAULT_VALUES['world']

    base_state = {
        'country_cumulative': defaults_country['cumulative_emissions_gt'],
        'world_cumulative': defaults_world['cumulative_emissions_gt'],
        'country_gdp': defaults_country['gdp_ppp_trillion'],
        'world_gdp': defaults_world['gdp_ppp_trillion'],
        'country_population': defaults_country['population_million'],
        'world_population': defaults_world['population_billion'],
    }

    for key, value in base_state.items():
        st.session_state.setdefault(key, value)

    st.session_state.setdefault('gdp_method', "PPP-adjusted (Recommended)")

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
                value=st.session_state.country_cumulative,
                step=0.1,
                help="Korea default: 27.0 Gt (Source: Statista 2021)",
                key="country_cumulative"
            )
            st.caption("🔍 Source: Statista 2021, Global Carbon Project")

        with col2:
            st.markdown("#### Denominator (World)")
            world_cumulative = st.number_input(
                "World Cumulative Emissions (Gt CO₂, 1850-2021)",
                min_value=1000.0,
                max_value=5000.0,
                value=st.session_state.world_cumulative,
                step=10.0,
                help="Default: 2,500 Gt (Source: Carbon Brief 2021)",
                key="world_cumulative"
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

        gdp_method = st.radio(
            "GDP Method",
            ["PPP-adjusted (Recommended)", "Nominal"],
            horizontal=True,
            key="gdp_method"
        )

        prev_method = st.session_state.get('_last_gdp_method')
        if prev_method != gdp_method:
            if "PPP" in gdp_method:
                st.session_state.country_gdp = defaults_country['gdp_ppp_trillion']
                st.session_state.world_gdp = defaults_world['gdp_ppp_trillion']
            else:
                st.session_state.country_gdp = defaults_country['gdp_nominal_trillion']
                st.session_state.world_gdp = defaults_world['gdp_nominal_trillion']
            st.session_state['_last_gdp_method'] = gdp_method

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Numerator (Country)")
            country_gdp = st.number_input(
                "GDP (Trillion USD, 2023)",
                min_value=0.1,
                max_value=50.0,
                value=st.session_state.country_gdp,
                step=0.1,
                help="Korea GDP PPP: 2.7 trillion USD (Source: IMF 2023)",
                key="country_gdp"
            )
            st.caption("🔍 Source: IMF World Economic Outlook 2023")

        with col2:
            st.markdown("#### Denominator (World)")
            world_gdp = st.number_input(
                "World GDP (Trillion USD, 2023)",
                min_value=100.0,
                max_value=300.0,
                value=st.session_state.world_gdp,
                step=1.0,
                help="World GDP PPP: 184.26 trillion USD (Source: IMF 2023)",
                key="world_gdp"
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
                value=st.session_state.country_population,
                step=0.1,
                help="Korea population: 51.7 million (Source: Worldometer 2024)",
                key="country_population"
            )
            st.caption("🔍 Source: UN Population Division, Worldometer 2024")

        with col2:
            st.markdown("#### Denominator (World)")
            world_pop = st.number_input(
                "World Population (Billions, 2024)",
                min_value=6.0,
                max_value=10.0,
                value=st.session_state.world_population,
                step=0.1,
                help="World population: 8.0 billion (Source: UN 2024)",
                key="world_population"
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
    custom_factors = {
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
        },
        'gdp_method': gdp_method
    }
    st.session_state.custom_factors = custom_factors

    factor_signature = (
        round(resp_share, 8),
        round(cap_share, 8),
        round(eq_share, 8),
        round(country_cumulative, 4),
        round(world_cumulative, 4),
        round(country_gdp, 4),
        round(world_gdp, 4),
        round(country_pop, 4),
        round(world_pop, 4),
        gdp_method
    )

    previous_signature = st.session_state.get('_factor_signature')
    if previous_signature is not None and previous_signature != factor_signature:
        st.session_state.pop('budget_results', None)
        st.session_state.pop('pathway_result', None)
        st.session_state.pop('pathway_comparison', None)
    st.session_state['_factor_signature'] = factor_signature

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

    if st.button("Next: Calculate Budget", type="primary"):
        st.session_state.nav_page = "📊 Calculate Budget"
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

    if total_weight <= 0:
        st.sidebar.error("At least one BKIR weight must be greater than zero.")
        normalized_weights = None
    else:
        normalized_weights = {
            'responsibility': weight_r / total_weight,
            'capability': weight_c / total_weight,
            'equality': weight_e / total_weight
        }

        weight_signature = tuple(round(normalized_weights[k], 6) for k in ['responsibility', 'capability', 'equality'])
        previous_weight_signature = st.session_state.get('_weight_signature')
        if previous_weight_signature is not None and previous_weight_signature != weight_signature:
            st.session_state.pop('budget_results', None)
            st.session_state.pop('pathway_result', None)
            st.session_state.pop('pathway_comparison', None)

        st.session_state['_weight_signature'] = weight_signature
        st.session_state.bkir_weights = normalized_weights

    # Run calculation
    if st.sidebar.button("🚀 Calculate Budget", type="primary"):
        if normalized_weights is None:
            st.warning("⚠️ Unable to calculate budget. Please adjust BKIR weights.")
            st.stop()

        config = {
            'seed': 123,
            'n_draws': n_draws,
            'global_budget': {
                '1p5C': {'low': 4e11, 'mid': 5e11, 'high': 6.7e11},
                '2p0C': {'low': 1.05e12, 'mid': 1.15e12, 'high': 1.29e12}
            },
            'user_weights': {
                'responsibility': normalized_weights['responsibility'],
                'capability': normalized_weights['capability'],
                'equality': normalized_weights['equality']
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
        st.session_state.last_budget_inputs = {
            'custom_factors': custom_factors,
            'weights': normalized_weights,
            'n_draws': n_draws
        }
        st.session_state.pop('pathway_result', None)
        st.session_state.pop('pathway_comparison', None)
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

        if st.button("Next: Generate Pathways", type="primary"):
            st.session_state.nav_page = "📈 Generate Pathways"
            st.rerun()

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
            budget = stats['by_scenario']['1p5C']['korea_total']['median']
            industry_preview = stats['by_scenario']['1p5C']['industry']['median']
        elif scenario_choice == "2.0°C Scenario":
            budget = stats['by_scenario']['2p0C']['korea_total']['median']
            industry_preview = stats['by_scenario']['2p0C']['industry']['median']
        else:
            budget = stats['korea_total']['median']
            industry_preview = stats['industry']['median']

        st.success(
            "✅ Using national budget from calculation: "
            f"{budget/1e9:.2f} Gt CO₂\n\n"
            f"Industry share preview: {industry_preview/1e9:.2f} Gt CO₂"
        )

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

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏭 Sector Allocation")
    st.sidebar.markdown("Set the fraction of emissions for each sector:")

    # Industry fraction (as % of national)
    industry_fraction_pct = st.sidebar.slider(
        "Industry (% of National)",
        min_value=10.0,
        max_value=80.0,
        value=37.0,
        step=1.0,
        help="Industry sector's share of national emissions. Default: 37% based on Korea 2024 data"
    )
    industry_fraction = industry_fraction_pct / 100

    # Petrochemical fraction (as % of industry)
    petrochem_fraction_pct = st.sidebar.slider(
        "Petrochemical (% of Industry)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Petrochemical sector's share of industry emissions. Default: 10%"
    )
    petrochem_fraction = petrochem_fraction_pct / 100

    # Show calculated percentages
    st.sidebar.info(
        f"**Sector Shares:**\n"
        f"- Industry: {industry_fraction_pct:.0f}% of National\n"
        f"- Petrochemical: {petrochem_fraction_pct:.0f}% of Industry\n"
        f"- Petrochemical: {industry_fraction_pct * petrochem_fraction_pct / 100:.1f}% of National"
    )

    # Generate pathway
    if st.sidebar.button("📈 Generate Pathway", type="primary"):
        with st.spinner("Generating emission pathway..."):
            # Determine initial emission for national level
            # Assuming Korea's total emission in 2024 is around 600 Mt CO2/year
            national_start_emission = 600e6  # tCO2/year

            allocator = PathwayAllocator(
                start_year=start_year,
                end_year=end_year,
                start_emission=national_start_emission,
                config={'max_annual_reduction_rate': 0.20, 'end_epsilon': 1.0}
            )

            if compare_curves:
                comparison = allocator.compare_curves(
                    budget,
                    tier='three_tier',
                    industry_fraction=industry_fraction,
                    petrochem_fraction=petrochem_fraction
                )
                st.session_state.pathway_comparison = comparison
                st.session_state.allocator = allocator
                st.session_state.pathway_budget = budget
                st.session_state.industry_fraction = industry_fraction
                st.session_state.petrochem_fraction = petrochem_fraction
            else:
                result = allocator.allocate_budget(
                    budget, curve_type, validate=True,
                    tier='three_tier',
                    industry_fraction=industry_fraction,
                    petrochem_fraction=petrochem_fraction
                )
                st.session_state.pathway_result = result
                st.session_state.allocator = allocator

        st.success("✅ Pathway generated!")

    # Display results
    if compare_curves and 'pathway_comparison' in st.session_state:
        st.subheader("📊 Curve Comparison")

        comparison = st.session_state.pathway_comparison

        # Format columns dynamically
        format_dict = {
            'budget_error_pct': '{:.2f}%',
            'initial_reduction_pct': '{:.2f}%',
            'avg_annual_reduction_pct': '{:.2f}%',
            'final_emission_tco2': '{:.0f}',
            'cumulative_tco2': '{:.2e}'
        }

        # Add formatting for milestone columns (in Mt CO2/year)
        for year in [2035, 2040, 2045, 2050]:
            if f'emission_{year}' in comparison.columns:
                format_dict[f'emission_{year}'] = '{:.1f}'  # Mt CO2/year
            if f'industry_{year}' in comparison.columns:
                format_dict[f'industry_{year}'] = '{:.1f}'  # Mt CO2/year
            if f'petrochem_{year}' in comparison.columns:
                format_dict[f'petrochem_{year}'] = '{:.1f}'  # Mt CO2/year

        # Convert milestone values to Mt for display
        comparison_display = comparison.copy()
        for col in comparison_display.columns:
            if col.startswith('emission_') or col.startswith('industry_') or col.startswith('petrochem_'):
                comparison_display[col] = comparison_display[col] / 1e6  # tCO2 -> Mt

        st.dataframe(comparison_display.style.format(format_dict), use_container_width=True)

        # Show milestone table separately for clarity
        st.subheader("📅 Milestone Emissions by Curve Type")

        st.markdown("**🌍 National Level (Mt CO₂/year)**")
        milestone_cols = ['curve_type'] + [f'emission_{year}' for year in [2035, 2040, 2045, 2050]
                                           if f'emission_{year}' in comparison_display.columns]
        if len(milestone_cols) > 1:
            st.dataframe(
                comparison_display[milestone_cols].rename(columns={
                    'curve_type': 'Curve Type',
                    'emission_2035': '2035',
                    'emission_2040': '2040',
                    'emission_2045': '2045',
                    'emission_2050': '2050'
                }).style.format({col: '{:.1f}' for col in milestone_cols[1:]}),
                use_container_width=True
            )

        st.markdown("**🏭 Industry Level (Mt CO₂/year)**")
        industry_milestone_cols = ['curve_type'] + [f'industry_{year}' for year in [2035, 2040, 2045, 2050]
                                                     if f'industry_{year}' in comparison_display.columns]
        if len(industry_milestone_cols) > 1:
            st.dataframe(
                comparison_display[industry_milestone_cols].rename(columns={
                    'curve_type': 'Curve Type',
                    'industry_2035': '2035',
                    'industry_2040': '2040',
                    'industry_2045': '2045',
                    'industry_2050': '2050'
                }).style.format({col: '{:.1f}' for col in industry_milestone_cols[1:]}),
                use_container_width=True
            )

        st.markdown("**⚗️ Petrochemical Level (Mt CO₂/year)**")
        petrochem_milestone_cols = ['curve_type'] + [f'petrochem_{year}' for year in [2035, 2040, 2045, 2050]
                                                      if f'petrochem_{year}' in comparison_display.columns]
        if len(petrochem_milestone_cols) > 1:
            st.dataframe(
                comparison_display[petrochem_milestone_cols].rename(columns={
                    'curve_type': 'Curve Type',
                    'petrochem_2035': '2035',
                    'petrochem_2040': '2040',
                    'petrochem_2045': '2045',
                    'petrochem_2050': '2050'
                }).style.format({col: '{:.1f}' for col in petrochem_milestone_cols[1:]}),
                use_container_width=True
            )

        # Plot all pathways
        st.subheader("📈 Pathway Comparison Chart")

        # Create three tabs: National, Industry, and Petrochemical
        tab_nat, tab_ind, tab_pet = st.tabs(["🌍 National Level", "🏭 Industry Level", "⚗️ Petrochemical Level"])

        allocator = st.session_state.allocator
        budget = st.session_state.pathway_budget
        industry_fraction = st.session_state.industry_fraction
        petrochem_fraction = st.session_state.petrochem_fraction

        with tab_nat:
            fig_nat = go.Figure()
            for _, row in comparison.iterrows():
                if row['budget_error_pct'] < 20:  # Only show reasonable matches
                    try:
                        result = allocator.allocate_budget(
                            budget, row['curve_type'], validate=False,
                            tier='three_tier', industry_fraction=industry_fraction,
                            petrochem_fraction=petrochem_fraction
                        )
                        fig_nat.add_trace(go.Scatter(
                            x=result['years'],
                            y=result['pathway_national'] / 1e6,
                            name=row['curve_type'],
                            mode='lines',
                            line=dict(width=2)
                        ))
                    except:
                        pass

            fig_nat.update_layout(
                title="National Emission Pathways - All Curve Types",
                xaxis_title="Year",
                yaxis_title="Emissions (Mt CO₂/year)",
                hovermode='x unified',
                height=600
            )
            st.plotly_chart(fig_nat, use_container_width=True)

        with tab_ind:
            fig_ind = go.Figure()
            for _, row in comparison.iterrows():
                if row['budget_error_pct'] < 20:  # Only show reasonable matches
                    try:
                        result = allocator.allocate_budget(
                            budget, row['curve_type'], validate=False,
                            tier='three_tier', industry_fraction=industry_fraction,
                            petrochem_fraction=petrochem_fraction
                        )
                        fig_ind.add_trace(go.Scatter(
                            x=result['years'],
                            y=result['pathway_industry'] / 1e6,
                            name=row['curve_type'],
                            mode='lines',
                            line=dict(width=2)
                        ))
                    except:
                        pass

            fig_ind.update_layout(
                title="Industry Emission Pathways - All Curve Types",
                xaxis_title="Year",
                yaxis_title="Emissions (Mt CO₂/year)",
                hovermode='x unified',
                height=600
            )
            st.plotly_chart(fig_ind, use_container_width=True)

        with tab_pet:
            fig_pet = go.Figure()
            for _, row in comparison.iterrows():
                if row['budget_error_pct'] < 20:  # Only show reasonable matches
                    try:
                        result = allocator.allocate_budget(
                            budget, row['curve_type'], validate=False,
                            tier='three_tier', industry_fraction=industry_fraction,
                            petrochem_fraction=petrochem_fraction
                        )
                        fig_pet.add_trace(go.Scatter(
                            x=result['years'],
                            y=result['pathway_petrochem'] / 1e6,
                            name=row['curve_type'],
                            mode='lines',
                            line=dict(width=2)
                        ))
                    except:
                        pass

            fig_pet.update_layout(
                title="Petrochemical Emission Pathways - All Curve Types",
                xaxis_title="Year",
                yaxis_title="Emissions (Mt CO₂/year)",
                hovermode='x unified',
                height=600
            )
            st.plotly_chart(fig_pet, use_container_width=True)

    elif 'pathway_result' in st.session_state:
        result = st.session_state.pathway_result

        # Check tier type
        is_three_tier = result.get('tier') == 'three_tier'
        is_two_tier = result.get('tier') == 'two_tier'

        # Metrics
        if is_three_tier:
            st.subheader("📊 Budget Allocation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌍 National Budget", f"{result['budget_national']/1e9:.2f} Gt CO₂")
            with col2:
                st.metric("🏭 Industry Budget",
                         f"{result['budget_industry']/1e9:.2f} Gt CO₂",
                         f"{result['industry_fraction']:.0%} of National")
            with col3:
                st.metric("⚗️ Petrochemical Budget",
                         f"{result['budget_petrochem']/1e6:.0f} Mt CO₂",
                         f"{result['petrochem_fraction']:.0%} of Industry")

            st.subheader("📍 Baseline Emissions (2024)")
            nat_start = result['pathway_national'][0]
            ind_start = result['pathway_industry'][0]
            pet_start = result['pathway_petrochem'][0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌍 National Start", f"{nat_start/1e6:.0f} Mt/yr")
            with col2:
                st.metric("🏭 Industry Start", f"{ind_start/1e6:.0f} Mt/yr")
            with col3:
                st.metric("⚗️ Petrochemical Start", f"{pet_start/1e6:.1f} Mt/yr")

            st.markdown("---")

            # Milestone table
            st.subheader("📅 Milestone Emissions")

            milestones = result['milestones']
            milestone_df = pd.DataFrame({
                'Year': milestones['years'],
                'National (Mt/yr)': [e/1e6 if e is not None else None for e in milestones['national']],
                'Industry (Mt/yr)': [e/1e6 if e is not None else None for e in milestones['industry']],
                'Petrochemical (Mt/yr)': [e/1e6 if e is not None else None for e in milestones['petrochem']]
            })

            st.dataframe(
                milestone_df.style.format({
                    'National (Mt/yr)': '{:.1f}',
                    'Industry (Mt/yr)': '{:.1f}',
                    'Petrochemical (Mt/yr)': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")

        elif is_two_tier:
            st.subheader("📊 Budget Allocation")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("National Budget", f"{result['budget_national']/1e9:.2f} Gt CO₂")
            with col2:
                st.metric("Industry Budget", f"{result['budget_industry']/1e9:.2f} Gt CO₂ ({result['industry_fraction']:.0%})")

            st.subheader("📍 Baseline Emissions (2024)")
            nat_start = result['pathway_national'][0]
            ind_start = result['pathway_industry'][0]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌍 National Start", f"{nat_start/1e6:.0f} Mt/yr")
            with col2:
                st.metric("🏭 Industry Start", f"{ind_start/1e6:.0f} Mt/yr")

            st.markdown("---")

            # Milestone table
            st.subheader("📅 Milestone Emissions")

            milestones = result['milestones']
            milestone_df = pd.DataFrame({
                'Year': milestones['years'],
                'National (Mt/yr)': [e/1e6 if e is not None else None for e in milestones['national']],
                'Industry (Mt/yr)': [e/1e6 if e is not None else None for e in milestones['industry']]
            })

            st.dataframe(
                milestone_df.style.format({
                    'National (Mt/yr)': '{:.1f}',
                    'Industry (Mt/yr)': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")

        else:
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
        if is_three_tier:
            validation_nat = result['validation']['national']
            validation_ind = result['validation']['industry']
            validation_pet = result['validation']['petrochem']

            col1, col2, col3 = st.columns(3)
            with col1:
                if validation_nat['is_valid']:
                    st.success("✅ National pathway is feasible")
                else:
                    with st.expander("⚠️ National Issues", expanded=False):
                        for issue in validation_nat['issues']:
                            st.warning(f"• {issue}")
            with col2:
                if validation_ind['is_valid']:
                    st.success("✅ Industry pathway is feasible")
                else:
                    with st.expander("⚠️ Industry Issues", expanded=False):
                        for issue in validation_ind['issues']:
                            st.warning(f"• {issue}")
            with col3:
                if validation_pet['is_valid']:
                    st.success("✅ Petrochemical pathway is feasible")
                else:
                    with st.expander("⚠️ Petrochemical Issues", expanded=False):
                        for issue in validation_pet['issues']:
                            st.warning(f"• {issue}")

        elif is_two_tier:
            validation_nat = result['validation']['national']
            validation_ind = result['validation']['industry']

            col1, col2 = st.columns(2)
            with col1:
                if validation_nat['is_valid']:
                    st.success("✅ National pathway is feasible")
                else:
                    with st.expander("⚠️ National Validation Issues", expanded=False):
                        for issue in validation_nat['issues']:
                            st.warning(f"• {issue}")
            with col2:
                if validation_ind['is_valid']:
                    st.success("✅ Industry pathway is feasible")
                else:
                    with st.expander("⚠️ Industry Validation Issues", expanded=False):
                        for issue in validation_ind['issues']:
                            st.warning(f"• {issue}")
        else:
            validation = result['validation']
            if validation['is_valid']:
                st.success("✅ Pathway is feasible and valid")
            else:
                with st.expander("⚠️ Validation Issues (click to expand)", expanded=False):
                    for issue in validation['issues']:
                        st.warning(f"• {issue}")

        # Main pathway chart
        st.subheader(f"📊 {result['curve_type'].title()} Pathway")

        if is_three_tier:
            # Three charts in tabs
            tab1, tab2, tab3 = st.tabs(["🌍 National Pathway", "🏭 Industry Pathway", "⚗️ Petrochemical Pathway"])

            with tab1:
                fig_nat = go.Figure()
                fig_nat.add_trace(go.Scatter(
                    x=result['years'],
                    y=result['pathway_national'] / 1e6,
                    name='National Emissions',
                    mode='lines',
                    line=dict(color='#2171b5', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(33, 113, 181, 0.2)'
                ))
                fig_nat.update_layout(
                    title=f"National Emission Pathway ({result['curve_type'].title()} Curve)",
                    xaxis_title="Year",
                    yaxis_title="Emissions (Mt CO₂/year)",
                    hovermode='x',
                    height=500
                )
                st.plotly_chart(fig_nat, use_container_width=True)

            with tab2:
                fig_ind = go.Figure()
                fig_ind.add_trace(go.Scatter(
                    x=result['years'],
                    y=result['pathway_industry'] / 1e6,
                    name='Industry Emissions',
                    mode='lines',
                    line=dict(color='#238b45', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(35, 139, 69, 0.2)'
                ))
                fig_ind.update_layout(
                    title=f"Industry Emission Pathway ({result['curve_type'].title()} Curve)",
                    xaxis_title="Year",
                    yaxis_title="Emissions (Mt CO₂/year)",
                    hovermode='x',
                    height=500
                )
                st.plotly_chart(fig_ind, use_container_width=True)

            with tab3:
                fig_pet = go.Figure()
                fig_pet.add_trace(go.Scatter(
                    x=result['years'],
                    y=result['pathway_petrochem'] / 1e6,
                    name='Petrochemical Emissions',
                    mode='lines',
                    line=dict(color='#d95f02', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(217, 95, 2, 0.2)'
                ))
                fig_pet.update_layout(
                    title=f"Petrochemical Emission Pathway ({result['curve_type'].title()} Curve)",
                    xaxis_title="Year",
                    yaxis_title="Emissions (Mt CO₂/year)",
                    hovermode='x',
                    height=500
                )
                st.plotly_chart(fig_pet, use_container_width=True)

        elif is_two_tier:
            # Two charts side by side or in tabs
            tab1, tab2 = st.tabs(["🌍 National Pathway", "🏭 Industry Pathway"])

            with tab1:
                fig_nat = go.Figure()
                fig_nat.add_trace(go.Scatter(
                    x=result['years'],
                    y=result['pathway_national'] / 1e6,
                    name='National Emissions',
                    mode='lines',
                    line=dict(color='#2171b5', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(33, 113, 181, 0.2)'
                ))
                fig_nat.update_layout(
                    title=f"National Emission Pathway ({result['curve_type'].title()} Curve)",
                    xaxis_title="Year",
                    yaxis_title="Emissions (Mt CO₂/year)",
                    hovermode='x',
                    height=500
                )
                st.plotly_chart(fig_nat, use_container_width=True)

            with tab2:
                fig_ind = go.Figure()
                fig_ind.add_trace(go.Scatter(
                    x=result['years'],
                    y=result['pathway_industry'] / 1e6,
                    name='Industry Emissions',
                    mode='lines',
                    line=dict(color='#238b45', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(35, 139, 69, 0.2)'
                ))
                fig_ind.update_layout(
                    title=f"Industry Emission Pathway ({result['curve_type'].title()} Curve)",
                    xaxis_title="Year",
                    yaxis_title="Emissions (Mt CO₂/year)",
                    hovermode='x',
                    height=500
                )
                st.plotly_chart(fig_ind, use_container_width=True)

        else:
            fig = go.Figure()
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

        if is_three_tier:
            tab1, tab2, tab3 = st.tabs(["🌍 National", "🏭 Industry", "⚗️ Petrochemical"])

            with tab1:
                cumulative_nat = np.cumsum(result['pathway_national'])
                fig2_nat = go.Figure()
                fig2_nat.add_trace(go.Scatter(
                    x=result['years'],
                    y=cumulative_nat / 1e9,
                    name='Cumulative National Emissions',
                    mode='lines',
                    line=dict(color='#238b45', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(35, 139, 69, 0.2)'
                ))
                fig2_nat.add_hline(
                    y=result['budget_national'] / 1e9,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Budget: {result['budget_national']/1e9:.2f} Gt",
                    annotation_position="right"
                )
                fig2_nat.update_layout(
                    title="National Cumulative Emissions vs Budget",
                    xaxis_title="Year",
                    yaxis_title="Cumulative Emissions (Gt CO₂)",
                    hovermode='x',
                    height=400
                )
                st.plotly_chart(fig2_nat, use_container_width=True)

            with tab2:
                cumulative_ind = np.cumsum(result['pathway_industry'])
                fig2_ind = go.Figure()
                fig2_ind.add_trace(go.Scatter(
                    x=result['years'],
                    y=cumulative_ind / 1e9,
                    name='Cumulative Industry Emissions',
                    mode='lines',
                    line=dict(color='#d95f02', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(217, 95, 2, 0.2)'
                ))
                fig2_ind.add_hline(
                    y=result['budget_industry'] / 1e9,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Budget: {result['budget_industry']/1e9:.2f} Gt",
                    annotation_position="right"
                )
                fig2_ind.update_layout(
                    title="Industry Cumulative Emissions vs Budget",
                    xaxis_title="Year",
                    yaxis_title="Cumulative Emissions (Gt CO₂)",
                    hovermode='x',
                    height=400
                )
                st.plotly_chart(fig2_ind, use_container_width=True)

            with tab3:
                cumulative_pet = np.cumsum(result['pathway_petrochem'])
                fig2_pet = go.Figure()
                fig2_pet.add_trace(go.Scatter(
                    x=result['years'],
                    y=cumulative_pet / 1e6,  # Show in Mt for smaller scale
                    name='Cumulative Petrochemical Emissions',
                    mode='lines',
                    line=dict(color='#7570b3', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(117, 112, 179, 0.2)'
                ))
                fig2_pet.add_hline(
                    y=result['budget_petrochem'] / 1e6,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Budget: {result['budget_petrochem']/1e6:.0f} Mt",
                    annotation_position="right"
                )
                fig2_pet.update_layout(
                    title="Petrochemical Cumulative Emissions vs Budget",
                    xaxis_title="Year",
                    yaxis_title="Cumulative Emissions (Mt CO₂)",
                    hovermode='x',
                    height=400
                )
                st.plotly_chart(fig2_pet, use_container_width=True)

        elif is_two_tier:
            tab1, tab2 = st.tabs(["🌍 National", "🏭 Industry"])

            with tab1:
                cumulative_nat = np.cumsum(result['pathway_national'])
                fig2_nat = go.Figure()
                fig2_nat.add_trace(go.Scatter(
                    x=result['years'],
                    y=cumulative_nat / 1e9,
                    name='Cumulative National Emissions',
                    mode='lines',
                    line=dict(color='#238b45', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(35, 139, 69, 0.2)'
                ))
                fig2_nat.add_hline(
                    y=result['budget_national'] / 1e9,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Budget: {result['budget_national']/1e9:.2f} Gt",
                    annotation_position="right"
                )
                fig2_nat.update_layout(
                    title="National Cumulative Emissions vs Budget",
                    xaxis_title="Year",
                    yaxis_title="Cumulative Emissions (Gt CO₂)",
                    hovermode='x',
                    height=400
                )
                st.plotly_chart(fig2_nat, use_container_width=True)

            with tab2:
                cumulative_ind = np.cumsum(result['pathway_industry'])
                fig2_ind = go.Figure()
                fig2_ind.add_trace(go.Scatter(
                    x=result['years'],
                    y=cumulative_ind / 1e9,
                    name='Cumulative Industry Emissions',
                    mode='lines',
                    line=dict(color='#d95f02', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(217, 95, 2, 0.2)'
                ))
                fig2_ind.add_hline(
                    y=result['budget_industry'] / 1e9,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Budget: {result['budget_industry']/1e9:.2f} Gt",
                    annotation_position="right"
                )
                fig2_ind.update_layout(
                    title="Industry Cumulative Emissions vs Budget",
                    xaxis_title="Year",
                    yaxis_title="Cumulative Emissions (Gt CO₂)",
                    hovermode='x',
                    height=400
                )
                st.plotly_chart(fig2_ind, use_container_width=True)

        else:
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

        # Additional milestones (threshold-based) - only for single tier
        if not is_two_tier and not is_three_tier:
            st.subheader("🎯 Additional Milestones")

            milestones_extra = []
            pathway = result['pathway']
            years = result['years']

            # Find when emissions drop below certain thresholds
            for threshold, label in [(100e6, "100 Mt/yr"), (50e6, "50 Mt/yr"), (10e6, "10 Mt/yr"), (1e6, "1 Mt/yr")]:
                idx = np.where(pathway <= threshold)[0]
                if len(idx) > 0:
                    year = years[idx[0]]
                    milestones_extra.append({
                        'Milestone': f'Below {label}',
                        'Year': int(year),
                        'Emission': f'{pathway[idx[0]]/1e6:.1f} Mt/yr'
                    })

            if milestones_extra:
                st.dataframe(pd.DataFrame(milestones_extra), use_container_width=True, hide_index=True)

        # Export pathway
        st.subheader("💾 Export Pathway")

        if is_three_tier:
            # Export all three pathways
            pathway_df = pd.DataFrame({
                'year': result['years'],
                'national_emissions_tco2': result['pathway_national'],
                'national_emissions_mtco2': result['pathway_national'] / 1e6,
                'national_cumulative_gtco2': np.cumsum(result['pathway_national']) / 1e9,
                'industry_emissions_tco2': result['pathway_industry'],
                'industry_emissions_mtco2': result['pathway_industry'] / 1e6,
                'industry_cumulative_gtco2': np.cumsum(result['pathway_industry']) / 1e9,
                'petrochem_emissions_tco2': result['pathway_petrochem'],
                'petrochem_emissions_mtco2': result['pathway_petrochem'] / 1e6,
                'petrochem_cumulative_mtco2': np.cumsum(result['pathway_petrochem']) / 1e6
            })

            col1, col2 = st.columns(2)
            with col1:
                csv = pathway_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV (All 3 Pathways)",
                    csv,
                    f"pathway_threetier_{result['curve_type']}.csv",
                    "text/csv"
                )

            with col2:
                json_str = json.dumps({
                    'pathway_national': result['pathway_national'].tolist(),
                    'pathway_industry': result['pathway_industry'].tolist(),
                    'pathway_petrochem': result['pathway_petrochem'].tolist(),
                    'years': result['years'].tolist(),
                    'budget_national': result['budget_national'],
                    'budget_industry': result['budget_industry'],
                    'budget_petrochem': result['budget_petrochem'],
                    'industry_fraction': result['industry_fraction'],
                    'petrochem_fraction': result['petrochem_fraction'],
                    'curve_type': result['curve_type'],
                    'validation': {
                        'national': result['validation']['national'],
                        'industry': result['validation']['industry'],
                        'petrochem': result['validation']['petrochem']
                    },
                    'milestones': result['milestones']
                }, indent=2, default=str)

                st.download_button(
                    "📥 Download JSON (Complete)",
                    json_str,
                    f"pathway_threetier_{result['curve_type']}.json",
                    "application/json"
                )

        elif is_two_tier:
            # Export both national and industry pathways
            pathway_df = pd.DataFrame({
                'year': result['years'],
                'national_emissions_tco2': result['pathway_national'],
                'national_emissions_mtco2': result['pathway_national'] / 1e6,
                'national_cumulative_gtco2': np.cumsum(result['pathway_national']) / 1e9,
                'industry_emissions_tco2': result['pathway_industry'],
                'industry_emissions_mtco2': result['pathway_industry'] / 1e6,
                'industry_cumulative_gtco2': np.cumsum(result['pathway_industry']) / 1e9
            })

            col1, col2 = st.columns(2)
            with col1:
                csv = pathway_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV (Both Pathways)",
                    csv,
                    f"pathway_twotier_{result['curve_type']}.csv",
                    "text/csv"
                )

            with col2:
                json_str = json.dumps({
                    'pathway_national': result['pathway_national'].tolist(),
                    'pathway_industry': result['pathway_industry'].tolist(),
                    'years': result['years'].tolist(),
                    'budget_national': result['budget_national'],
                    'budget_industry': result['budget_industry'],
                    'industry_fraction': result['industry_fraction'],
                    'curve_type': result['curve_type'],
                    'validation': {
                        'national': result['validation']['national'],
                        'industry': result['validation']['industry']
                    },
                    'milestones': result['milestones']
                }, indent=2, default=str)

                st.download_button(
                    "📥 Download JSON (Complete)",
                    json_str,
                    f"pathway_twotier_{result['curve_type']}.json",
                    "application/json"
                )

        else:
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
                }, indent=2, default=str)

                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    f"pathway_{result['curve_type']}.json",
                    "application/json"
                )

        if st.button("Next: Summary Report", type="primary"):
            st.session_state.nav_page = "📋 Summary Report"
            st.rerun()

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

            if 'bkir_weights' in st.session_state:
                weights = st.session_state.bkir_weights
                weight_df = pd.DataFrame({
                    'Weight': ['Responsibility', 'Capability', 'Equality'],
                    'Share (%)': [
                        weights['responsibility'] * 100,
                        weights['capability'] * 100,
                        weights['equality'] * 100
                    ]
                })
                st.markdown("**BKIR Weighting**")
                st.dataframe(
                    weight_df.style.format({'Share (%)': '{:.2f}'}),
                    use_container_width=True,
                    hide_index=True
                )

        # Section 3: Pathway Results
        if has_pathway:
            st.subheader("3️⃣ Emission Pathway")
            result = st.session_state.pathway_result
            tier = result.get('tier', 'single')
            validation = result.get('validation', {})

            primary_path = result.get('pathway_national', result['pathway'])
            primary_budget = result.get('budget_national', result.get('budget_allocated'))
            start_primary = float(primary_path[0])
            end_primary = float(primary_path[-1])
            reduction_primary = (1 - end_primary / start_primary) * 100 if start_primary else 0.0

            def _format_pct(value):
                try:
                    return f"{float(value):.2f}%"
                except (TypeError, ValueError):
                    return "N/A"

            if tier in ('two_tier', 'three_tier'):
                budget_match_pct = validation.get('national', {}).get('budget_error_pct')
            else:
                budget_match_pct = validation.get('budget_error_pct')

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
                    "N/A" if primary_budget is None else f"{float(primary_budget)/1e9:.2f} Gt CO₂",
                    f"{start_primary/1e6:.0f} Mt/yr",
                    f"{end_primary/1e3:.1f} kt/yr",
                    f"{reduction_primary:.1f}%",
                    _format_pct(budget_match_pct)
                ]
            })

            st.dataframe(pathway_summary, use_container_width=True, hide_index=True)

            if tier in ('two_tier', 'three_tier'):
                st.markdown("**Sector Breakdown**")

                sector_rows = []
                sector_defs = [
                    ('🌍 National', result.get('pathway_national'), result.get('budget_national'),
                     validation.get('national', {}))
                ]

                if tier in ('two_tier', 'three_tier'):
                    sector_defs.append((
                        '🏭 Industry',
                        result.get('pathway_industry'),
                        result.get('budget_industry'),
                        validation.get('industry', {})
                    ))

                if tier == 'three_tier':
                    sector_defs.append((
                        '⚗️ Petrochemical',
                        result.get('pathway_petrochem'),
                        result.get('budget_petrochem'),
                        validation.get('petrochem', {})
                    ))

                for label, pathway, budget_value, val in sector_defs:
                    if pathway is None:
                        continue
                    start_val = float(pathway[0])
                    end_val = float(pathway[-1])
                    reduction = (1 - end_val / start_val) * 100 if start_val else 0.0
                    try:
                        budget_gt = float(budget_value) / 1e9 if budget_value is not None else np.nan
                    except (TypeError, ValueError):
                        budget_gt = np.nan

                    sector_rows.append({
                        'Sector': label,
                        'Budget (Gt CO₂)': budget_gt,
                        'Start (Mt/yr)': start_val / 1e6,
                        'End (Mt/yr)': end_val / 1e6,
                        'Reduction (%)': reduction,
                        'Budget Match (%)': val.get('budget_error_pct')
                    })

                if sector_rows:
                    sector_df = pd.DataFrame(sector_rows)
                    st.dataframe(
                        sector_df.style.format({
                            'Budget (Gt CO₂)': '{:.3f}',
                            'Start (Mt/yr)': '{:.2f}',
                            'End (Mt/yr)': '{:.2f}',
                            'Reduction (%)': '{:.1f}',
                            'Budget Match (%)': '{:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

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
