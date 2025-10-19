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

    defaults_country = AllocationCalculator.DEFAULT_VALUES['korea']
    defaults_world = AllocationCalculator.DEFAULT_VALUES['world']

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
    st.session_state.setdefault('_last_gdp_method', st.session_state['gdp_method'])

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
        active_preset = st.session_state.get('country_preset', defaults_country)
        if prev_method != gdp_method:
            if "PPP" in gdp_method:
                st.session_state.country_gdp = active_preset.get('gdp_ppp_trillion', defaults_country['gdp_ppp_trillion'])
                st.session_state.world_gdp = defaults_world['gdp_ppp_trillion']
            else:
                st.session_state.country_gdp = active_preset.get('gdp_nominal_trillion', defaults_country['gdp_nominal_trillion'])
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
            st.session_state.country_cumulative = defaults_country['cumulative_emissions_gt']
            st.session_state.world_cumulative = defaults_world['cumulative_emissions_gt']
            st.session_state.country_gdp = defaults_country['gdp_ppp_trillion']
            st.session_state.world_gdp = defaults_world['gdp_ppp_trillion']
            st.session_state.country_population = defaults_country['population_million']
            st.session_state.world_population = defaults_world['population_billion']
            st.session_state.country_preset = defaults_country
            st.session_state['gdp_method'] = "PPP-adjusted (Recommended)"
            st.session_state['_last_gdp_method'] = "PPP-adjusted (Recommended)"
            st.session_state['weight_responsibility'] = 0.3
            st.session_state['weight_capability'] = 0.4
            st.session_state['weight_equality'] = 0.3
            st.session_state['budget_n_draws'] = 1000

            st.session_state.custom_factors = {
                'responsibility': 0.0109,
                'capability': 0.0147,
                'equality': 0.00646
            }
            for key in ['_factor_signature', 'budget_results', 'pathway_result', 'pathway_comparison',
                        '_weight_signature', 'bkir_weights', 'pathway_budget',
                        'industry_fraction', 'petrochem_fraction', '_trigger_budget_run',
                        '_trigger_pathway_run']:
                st.session_state.pop(key, None)
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
    n_draws = st.sidebar.slider(
        "Monte Carlo Draws",
        100,
        5000,
        value=st.session_state.get('budget_n_draws', 1000),
        key="budget_n_draws"
    )

    # Weights
    st.sidebar.subheader("BKIR Weights")
    weight_r = st.sidebar.slider(
        "Responsibility Weight",
        0.0,
        1.0,
        value=st.session_state.get('weight_responsibility', 0.3),
        step=0.05,
        key="weight_responsibility"
    )
    weight_c = st.sidebar.slider(
        "Capability Weight",
        0.0,
        1.0,
        value=st.session_state.get('weight_capability', 0.4),
        step=0.05,
        key="weight_capability"
    )
    weight_e = st.sidebar.slider(
        "Equality Weight",
        0.0,
        1.0,
        value=st.session_state.get('weight_equality', 0.3),
        step=0.05,
        key="weight_equality"
    )

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

    def run_budget_calculation(auto_trigger: bool = False) -> None:
        """Execute Monte Carlo budget calculation."""
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
            }
        }

        spinner_msg = "Auto-running Monte Carlo simulation..." if auto_trigger else "Running Monte Carlo simulation..."
        with st.spinner(spinner_msg):
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
        st.session_state.pop('_trigger_budget_run', None)

        if auto_trigger:
            st.info("✅ Budget automatically recalculated with the latest settings.")
        else:
            st.success("✅ Budget calculated with your custom factors!")

    # Run calculation (manual trigger)
    if st.sidebar.button("🚀 Calculate Budget", type="primary"):
        if normalized_weights is None:
            st.warning("⚠️ Unable to calculate budget. Please adjust BKIR weights.")
        else:
            run_budget_calculation(auto_trigger=False)

    # Automatic calculation trigger (e.g., after completing previous step)
    if st.session_state.get('_trigger_budget_run') and normalized_weights is not None:
        run_budget_calculation(auto_trigger=True)
    elif st.session_state.get('_trigger_budget_run'):
        # Clear trigger if weights are not yet valid
        st.session_state.pop('_trigger_budget_run', None)

    # Display results
    if 'budget_results' in st.session_state:
        results = st.session_state.budget_results
        stats = results['statistics']

        # Key metrics
        industry_stats = stats.get('industry')
        petrochem_stats = stats.get('petrochem')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Korea Budget", f"{stats['korea_total']['median']/1e9:.2f} Gt CO₂")
        with col2:
            if industry_stats:
                st.metric("Industry", f"{industry_stats['median']/1e9:.2f} Gt CO₂")
            else:
                st.metric("Industry", "N/A")
        with col3:
            if petrochem_stats:
                st.metric("Petrochemical", f"{petrochem_stats['median']/1e9:.2f} Gt CO₂")
            else:
                st.metric("Petrochemical", "N/A")

        # Scenario breakdown
        if 'by_scenario' in stats and stats['by_scenario']:
            st.subheader("Budget by Climate Scenario")

            scenario_data = []
            for scenario, data in stats['by_scenario'].items():
                row = {
                    'Scenario': '1.5°C' if scenario == '1p5C' else '2.0°C',
                    'Korea (Gt)': data['korea_total']['median'] / 1e9
                }
                if data.get('industry'):
                    row['Industry (Gt)'] = data['industry']['median'] / 1e9
                if data.get('petrochem'):
                    row['Petrochemical (Gt)'] = data['petrochem']['median'] / 1e9
                scenario_data.append(row)

            scenario_df = pd.DataFrame(scenario_data)
            st.dataframe(scenario_df, use_container_width=True)

        # Visualization
        industry_budgets = results['budgets'].get('industry')
        if industry_budgets is not None:
            industry_budgets = np.asarray(industry_budgets)
            finite_mask = np.isfinite(industry_budgets)
            if np.any(finite_mask):
                budgets_df = pd.DataFrame({
                    'Industry (Gt)': industry_budgets[finite_mask] / 1e9,
                    'Scenario': np.array(results['samples']['scenarios'])[finite_mask]
                })

                fig = px.box(
                    budgets_df,
                    x='Scenario',
                    y='Industry (Gt)',
                    color='Scenario',
                    title="Industry Budget Distribution by Scenario"
                )
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
        st.session_state['pathway_use_manual'] = True
        use_manual = True
    else:
        use_manual = st.checkbox(
            "Use manual budget instead of calculated budget",
            value=st.session_state.get('pathway_use_manual', False),
            key="pathway_use_manual"
        )

    # Budget selection
    if use_manual:
        st.sidebar.header("⚙️ Manual Budget Input")
        budget = st.sidebar.number_input(
            "Budget (Gt CO₂)",
            min_value=0.5,
            max_value=10.0,
            value=st.session_state.get('manual_budget_gt', 2.14),
            step=0.1,
            key="manual_budget_gt"
        ) * 1e9
        st.info(f"Using manual budget: {budget/1e9:.2f} Gt CO₂")
    else:
        stats = st.session_state.budget_results['statistics']
        st.session_state.setdefault('pathway_scenario', "Mixed (Median)")
        scenario_choice = st.radio(
            "Select Scenario for Pathway",
            ["1.5°C Scenario", "2.0°C Scenario", "Mixed (Median)"],
            horizontal=True,
            key="pathway_scenario"
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

    curve_options = ["exponential", "logarithmic", "s_curve", "linear", "plateau",
                     "convex", "early_action", "delayed_action"]
    default_curve = st.session_state.get('pathway_curve_type', curve_options[0])
    curve_type = st.sidebar.selectbox(
        "Curve Type",
        curve_options,
        index=curve_options.index(default_curve) if default_curve in curve_options else 0,
        help="Select emission reduction curve shape",
        key="pathway_curve_type"
    )

    compare_curves = st.sidebar.checkbox(
        "Compare All Curves",
        value=st.session_state.get('pathway_compare_curves', False),
        key="pathway_compare_curves"
    )
    start_year = st.sidebar.number_input(
        "Start Year",
        2020,
        2030,
        value=st.session_state.get('pathway_start_year', 2024),
        key="pathway_start_year"
    )
    end_year = st.sidebar.number_input(
        "End Year",
        2040,
        2060,
        value=st.session_state.get('pathway_end_year', 2050),
        key="pathway_end_year"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏭 Sector Configuration")

    st.sidebar.markdown("**Step 1: Set Starting Emissions (2024)**")

    # Starting emissions for each sector (in Mt CO2/year)
    national_start_mt = st.sidebar.number_input(
        "National Total (Mt CO₂/year)",
        min_value=100.0,
        max_value=1000.0,
        value=st.session_state.get('national_start_mt', 600.0),
        step=10.0,
        help="Total national emissions in 2024",
        key="national_start_mt"
    )

    transformation_start_mt = st.sidebar.number_input(
        "Transformation Sector (Mt CO₂/year)",
        min_value=0.0,
        max_value=national_start_mt,
        value=st.session_state.get('transformation_start_mt', 90.0),
        step=5.0,
        help="Starting emissions for the transformation sector",
        key="transformation_start_mt"
    )

    other_start_mt = st.sidebar.number_input(
        "Other Sectors (Mt CO₂/year)",
        min_value=0.0,
        max_value=national_start_mt,
        value=st.session_state.get('other_start_mt', 30.0),
        step=5.0,
        help="Starting emissions for all other non-industry sectors",
        key="other_start_mt"
    )

    transformation_other_start_mt = transformation_start_mt + other_start_mt
    if transformation_other_start_mt > national_start_mt:
        st.sidebar.error("⚠️ Transformation + Other exceeds national total. Adjust values.")
        industry_start_mt = 0.0
    else:
        industry_start_mt = national_start_mt - transformation_other_start_mt

    st.session_state['industry_start_mt'] = industry_start_mt
    st.sidebar.write(f"🏭 Computed Industry Start: **{industry_start_mt:.1f} Mt CO₂/year**")

    petrochem_start_mt = st.sidebar.number_input(
        "Petrochemical (Mt CO₂/year)",
        min_value=0.0,
        max_value=industry_start_mt,
        value=st.session_state.get('petrochem_start_mt', min(22.0, industry_start_mt)),
        step=1.0,
        help="Starting emissions for petrochemical sector",
        key="petrochem_start_mt"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Step 2: Set Budgets**")

    # Sector budgets
    transformation_budget_gt = st.sidebar.number_input(
        "Transformation Budget (Gt CO₂)",
        min_value=0.0,
        max_value=budget/1e9,
        value=st.session_state.get('transformation_budget_gt', max(budget/1e9 * 0.12, 0.0)),
        step=0.05,
        help="Total carbon budget allocated to the transformation sector",
        key="transformation_budget_gt"
    )
    transformation_budget = transformation_budget_gt * 1e9

    other_budget_gt = st.sidebar.number_input(
        "Other Sectors Budget (Gt CO₂)",
        min_value=0.0,
        max_value=max(budget/1e9 - transformation_budget_gt, 0.0),
        value=st.session_state.get('other_budget_gt', max(budget/1e9 * 0.08, 0.0)),
        step=0.05,
        help="Total carbon budget allocated to other non-industry sectors",
        key="other_budget_gt"
    )
    other_budget = other_budget_gt * 1e9

    # Industry budget = Total - Transformation - Other
    industry_budget = budget - transformation_budget - other_budget
    if industry_budget < 0:
        st.sidebar.error("⚠️ Transformation + Other budgets exceed national total. Adjust values.")
        industry_budget = 0.0

    industry_budget_gt = industry_budget / 1e9
    st.session_state['industry_budget_gt'] = industry_budget_gt
    st.sidebar.write(f"🏭 Computed Industry Budget: **{industry_budget_gt:.2f} Gt CO₂**")

    # Petrochemical budget (user can allocate from industry)
    petrochem_budget_gt = st.sidebar.number_input(
        "Petrochemical Budget (Gt CO₂)",
        min_value=0.0,
        max_value=industry_budget_gt,
        value=st.session_state.get('petrochem_budget_gt', min(industry_budget_gt * 0.1, industry_budget_gt)),
        step=0.05,
        help="Total carbon budget for petrochemical sector",
        key="petrochem_budget_gt"
    )
    petrochem_budget = petrochem_budget_gt * 1e9

    # Convert to tCO2 for calculations
    national_start_emission = national_start_mt * 1e6
    transformation_start_emission = transformation_start_mt * 1e6
    other_start_emission = other_start_mt * 1e6
    transformation_other_start_emission = transformation_other_start_mt * 1e6
    industry_start_emission = industry_start_mt * 1e6
    petrochem_start_emission = petrochem_start_mt * 1e6

    # Show summary
    st.sidebar.info(
        f"**Summary:**\n"
        f"**Starting Emissions (2024):**\n"
        f"- National: {national_start_mt:.0f} Mt/yr\n"
        f"- Transformation: {transformation_start_mt:.0f} Mt/yr\n"
        f"- Other: {other_start_mt:.0f} Mt/yr\n"
        f"- Industry: {industry_start_mt:.0f} Mt/yr\n"
        f"- Petrochemical: {petrochem_start_mt:.0f} Mt/yr\n\n"
        f"**Budgets (2024-2050):**\n"
        f"- National: {budget/1e9:.2f} Gt\n"
        f"- Transformation: {transformation_budget_gt:.2f} Gt\n"
        f"- Other: {other_budget_gt:.2f} Gt\n"
        f"- Industry: {industry_budget/1e9:.2f} Gt\n"
        f"- Petrochemical: {petrochem_budget/1e9:.2f} Gt"
    )

    # Generate pathway
    def run_pathway_generation(auto_trigger: bool = False) -> None:
        """Generate pathways for industry, transformation, other, and petrochemical sectors."""
        if compare_curves:
            st.warning("⚠️ Curve comparison is temporarily unavailable for the multi-sector configuration.")
            st.session_state.pathway_comparison = None
            st.session_state.pathway_result = None
            st.session_state.pop('_trigger_pathway_run', None)
            return

        spinner_msg = "Auto-generating emission pathway..." if auto_trigger else "Generating emission pathway..."
        with st.spinner(spinner_msg):
            path_config = {'max_annual_reduction_rate': 0.20}
            sector_inputs = {
                'industry': {'start': industry_start_emission, 'budget': industry_budget},
                'transformation': {'start': transformation_start_emission, 'budget': transformation_budget},
                'other': {'start': other_start_emission, 'budget': other_budget}
            }

            sector_results = {}
            for key, info in sector_inputs.items():
                allocator = PathwayAllocator(
                    start_year=start_year,
                    end_year=end_year,
                    start_emission=info['start'],
                    config=path_config
                )
                sector_results[key] = allocator.allocate_budget(
                    info['budget'],
                    curve_type,
                    validate=True,
                    tier='single',
                    national_start_emission=info['start']
                )

            result_petrochem = None
            if petrochem_budget > 0 and petrochem_start_emission > 0:
                allocator_petrochem = PathwayAllocator(
                    start_year=start_year,
                    end_year=end_year,
                    start_emission=petrochem_start_emission,
                    config=path_config
                )
                result_petrochem = allocator_petrochem.allocate_budget(
                    petrochem_budget,
                    curve_type,
                    validate=True,
                    tier='single',
                    national_start_emission=petrochem_start_emission
                )

            years = sector_results['industry']['years']
            pathway_industry = sector_results['industry']['pathway']
            pathway_transformation = sector_results['transformation']['pathway']
            pathway_other = sector_results['other']['pathway']

            pathway_national = pathway_industry + pathway_transformation + pathway_other
            cumulative_national = np.cumsum(pathway_national)
            budget_match_pct = ((cumulative_national[-1] - budget) / budget * 100) if budget > 0 else 0.0
            national_validation = {
                'is_valid': abs(budget_match_pct) <= 0.5,
                'budget_error_pct': budget_match_pct
            }

            milestone_years = sector_results['industry']['milestones']['years']
            industry_milestones = sector_results['industry']['milestones']['national']
            transformation_milestones = sector_results['transformation']['milestones']['national']
            other_milestones = sector_results['other']['milestones']['national']

            national_milestones = []
            for i in range(len(milestone_years)):
                total_value = 0.0
                for component in (industry_milestones, transformation_milestones, other_milestones):
                    value = component[i] if component[i] is not None else 0.0
                    total_value += value
                national_milestones.append(total_value)

            milestones = {
                'years': milestone_years,
                'national': national_milestones,
                'industry': industry_milestones,
                'transformation': transformation_milestones,
                'other': other_milestones
            }
            if result_petrochem:
                milestones['petrochem'] = result_petrochem['milestones']['national']

            combined_result = {
                'curve_type': curve_type,
                'years': years,
                'pathway': pathway_national,
                'pathway_national': pathway_national,
                'pathway_industry': pathway_industry,
                'pathway_transformation': pathway_transformation,
                'pathway_other': pathway_other,
                'pathway_petrochem': result_petrochem['pathway'] if result_petrochem else None,
                'budget_allocated': budget,
                'budget_national': budget,
                'budget_industry': industry_budget,
                'budget_transformation': transformation_budget,
                'budget_other': other_budget,
                'budget_petrochem': petrochem_budget if result_petrochem else 0.0,
                'cumulative_emissions': cumulative_national[-1],
                'validation': {
                    'national': national_validation,
                    'industry': sector_results['industry']['validation'],
                    'transformation': sector_results['transformation']['validation'],
                    'other': sector_results['other']['validation'],
                    'petrochem': result_petrochem['validation'] if result_petrochem else None
                },
                'milestones': milestones,
                'sector_results': sector_results,
                'tier': 'multi_sector',
                'industry_fraction': st.session_state.get('industry_fraction', 0.0),
                'petrochem_fraction': st.session_state.get('petrochem_fraction', 0.0)
            }
            if result_petrochem:
                combined_result['sector_results']['petrochem'] = result_petrochem

            st.session_state.pathway_result = combined_result
            st.session_state.pathway_comparison = None

            st.session_state.pathway_budget = budget
            st.session_state.transformation_budget = transformation_budget
            st.session_state.other_budget = other_budget
            st.session_state.industry_budget = industry_budget
            st.session_state.petrochem_budget = petrochem_budget
            st.session_state.transformation_other_budget = transformation_budget + other_budget

            if budget > 0:
                st.session_state.transformation_fraction = transformation_budget / budget
                st.session_state.other_fraction = other_budget / budget
                st.session_state.industry_fraction = industry_budget / budget
                st.session_state.transformation_other_fraction = (transformation_budget + other_budget) / budget
            if industry_budget > 0:
                st.session_state.petrochem_fraction = petrochem_budget / industry_budget if industry_budget else 0.0

        st.session_state.pop('_trigger_pathway_run', None)

        if auto_trigger:
            st.info("✅ Pathway automatically generated with updated sector inputs.")
        else:
            st.success("✅ Pathway generated successfully!")

    if st.sidebar.button("📈 Generate Pathway", type="primary"):
        run_pathway_generation(auto_trigger=False)

    if st.session_state.get('_trigger_pathway_run'):
        run_pathway_generation(auto_trigger=True)

    # Display results
    pathway_result = st.session_state.get('pathway_result')
    if pathway_result and pathway_result.get('tier') == 'multi_sector':
        transformation_budget_val = pathway_result['budget_transformation']
        other_budget_val = pathway_result['budget_other']
        industry_budget_val = pathway_result['budget_industry']
        national_budget_val = pathway_result['budget_national']
        petrochem_budget_val = pathway_result.get('budget_petrochem', 0.0)

        transformation_fraction = st.session_state.get('transformation_fraction', 0.0)
        other_fraction = st.session_state.get('other_fraction', 0.0)
        industry_fraction = st.session_state.get('industry_fraction', 0.0)

        info_lines = [
            f"**Budget Allocation:**",
            f"- National: {national_budget_val/1e9:.2f} Gt CO₂",
            f"- Industry: {industry_budget_val/1e9:.2f} Gt CO₂ ({industry_fraction*100:.1f}%)",
            f"- Transformation: {transformation_budget_val/1e9:.2f} Gt CO₂ ({transformation_fraction*100:.1f}%)",
            f"- Other: {other_budget_val/1e9:.2f} Gt CO₂ ({other_fraction*100:.1f}%)"
        ]
        if petrochem_budget_val:
            info_lines.append(f"- Petrochemical: {petrochem_budget_val/1e9:.2f} Gt CO₂")
        st.info("\n".join(info_lines))

        # Budget metrics
        metric_cols = st.columns(5)
        metric_cols[0].metric("National Budget", f"{national_budget_val/1e9:.2f} Gt")
        metric_cols[1].metric("Industry Budget", f"{industry_budget_val/1e9:.2f} Gt")
        metric_cols[2].metric("Transformation Budget", f"{transformation_budget_val/1e9:.2f} Gt")
        metric_cols[3].metric("Other Budget", f"{other_budget_val/1e9:.2f} Gt")
        metric_cols[4].metric("Petrochemical Budget", f"{petrochem_budget_val/1e9:.2f} Gt")

        # Baseline emissions
        start_cols = st.columns(4)
        start_cols[0].metric("National Start", f"{pathway_result['pathway_national'][0]/1e6:.0f} Mt/yr")
        start_cols[1].metric("Industry Start", f"{pathway_result['pathway_industry'][0]/1e6:.0f} Mt/yr")
        start_cols[2].metric("Transformation Start", f"{pathway_result['pathway_transformation'][0]/1e6:.0f} Mt/yr")
        start_cols[3].metric("Other Start", f"{pathway_result['pathway_other'][0]/1e6:.0f} Mt/yr")

        # Validation summary
        validation = pathway_result['validation']
        val_cols = st.columns(4)
        for col, key, label in zip(val_cols, ['national', 'industry', 'transformation', 'other'],
                                   ['National', 'Industry', 'Transformation', 'Other']):
            result_val = validation.get(key)
            if result_val and result_val.get('is_valid', True):
                col.success(f"✅ {label} pathway valid")
            else:
                issues = (result_val or {}).get('issues', [])
                with col.expander(f"⚠️ {label} Issues", expanded=False):
                    if issues:
                        for issue in issues:
                            st.warning(f"• {issue}")
                    else:
                        st.warning("• Validation flagged issues, please review inputs.")

        if validation.get('petrochem'):
            pet_col = st.columns(1)[0]
            pet_val = validation['petrochem']
            if pet_val.get('is_valid', True):
                pet_col.success("✅ Petrochemical pathway valid")
            else:
                issues = pet_val.get('issues', [])
                with pet_col.expander("⚠️ Petrochemical Issues", expanded=False):
                    if issues:
                        for issue in issues:
                            st.warning(f"• {issue}")
                    else:
                        st.warning("• Validation flagged issues, please review inputs.")

        # Emission pathways
        st.subheader("📈 Emission Pathways")
        tabs = ["🌍 National", "🏭 Industry", "🔄 Transformation", "📦 Other"]
        petrochem_path = pathway_result.get('pathway_petrochem')
        if petrochem_path is not None:
            tabs.append("⚗️ Petrochemical")

        tab_objects = st.tabs(tabs)
        sector_paths = [
            ("National Emissions", pathway_result['pathway_national']),
            ("Industry Emissions", pathway_result['pathway_industry']),
            ("Transformation Emissions", pathway_result['pathway_transformation']),
            ("Other Emissions", pathway_result['pathway_other'])
        ]
        if petrochem_path is not None:
            sector_paths.append(("Petrochemical Emissions", petrochem_path))

        for tab, (label, pathway) in zip(tab_objects, sector_paths):
            with tab:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pathway_result['years'],
                    y=pathway / 1e6,
                    name=label,
                    mode='lines',
                    line=dict(width=3)
                ))
                fig.update_layout(
                    title=f"{label} Pathway ({pathway_result['curve_type'].title()} Curve)",
                    xaxis_title="Year",
                    yaxis_title="Emissions (Mt CO₂/year)",
                    hovermode='x',
                    height=480
                )
                st.plotly_chart(fig, use_container_width=True)

        # Cumulative emissions chart
        st.subheader("📉 Cumulative Emissions vs Budget")
        cumulative = np.cumsum(pathway_result['pathway_national']) / 1e9
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=pathway_result['years'],
            y=cumulative,
            name='Cumulative Emissions',
            mode='lines',
            line=dict(color='#2171b5', width=3),
            fill='tozeroy',
            fillcolor='rgba(33, 113, 181, 0.2)'
        ))
        fig_cum.add_hline(
            y=pathway_result['budget_national'] / 1e9,
            line_dash='dash',
            line_color='red',
            annotation_text=f"Budget: {pathway_result['budget_national']/1e9:.2f} Gt",
            annotation_position='right'
        )
        fig_cum.update_layout(
            xaxis_title='Year',
            yaxis_title='Cumulative Emissions (Gt CO₂)',
            hovermode='x',
            height=420
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Milestones table
        milestones = pathway_result['milestones']
        milestone_years = milestones['years']
        milestone_rows = []
        for label, key in [
            ('National', 'national'),
            ('Industry', 'industry'),
            ('Transformation', 'transformation'),
            ('Other', 'other')
        ]:
            values = milestones.get(key)
            if values:
                row = {'Sector': label}
                for year, value in zip(milestone_years, values):
                    row[str(year)] = value / 1e6 if value is not None else None
                milestone_rows.append(row)
        if milestones.get('petrochem'):
            row = {'Sector': 'Petrochemical'}
            for year, value in zip(milestone_years, milestones['petrochem']):
                row[str(year)] = value / 1e6 if value is not None else None
            milestone_rows.append(row)

        if milestone_rows:
            milestone_df = pd.DataFrame(milestone_rows)
            st.subheader("📅 Milestone Emissions (Mt CO₂/year)")
            st.dataframe(milestone_df.style.format({col: '{:.1f}' for col in milestone_df.columns if col != 'Sector'}),
                         use_container_width=True, hide_index=True)

        # Export
        st.subheader("💾 Export Pathway")
        pathway_df = pd.DataFrame({
            'year': pathway_result['years'],
            'national_emissions_tco2': pathway_result['pathway_national'],
            'national_emissions_mtco2': pathway_result['pathway_national'] / 1e6,
            'industry_emissions_tco2': pathway_result['pathway_industry'],
            'industry_emissions_mtco2': pathway_result['pathway_industry'] / 1e6,
            'transformation_emissions_tco2': pathway_result['pathway_transformation'],
            'transformation_emissions_mtco2': pathway_result['pathway_transformation'] / 1e6,
            'other_emissions_tco2': pathway_result['pathway_other'],
            'other_emissions_mtco2': pathway_result['pathway_other'] / 1e6
        })
        if petrochem_path is not None:
            pathway_df['petrochem_emissions_tco2'] = petrochem_path
            pathway_df['petrochem_emissions_mtco2'] = petrochem_path / 1e6

        csv_data = pathway_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV (All Sectors)",
            csv_data,
            f"pathway_multisector_{pathway_result['curve_type']}.csv",
            "text/csv"
        )

        export_payload = {
            'curve_type': pathway_result['curve_type'],
            'years': pathway_result['years'].tolist(),
            'pathway_national': pathway_result['pathway_national'].tolist(),
            'pathway_industry': pathway_result['pathway_industry'].tolist(),
            'pathway_transformation': pathway_result['pathway_transformation'].tolist(),
            'pathway_other': pathway_result['pathway_other'].tolist(),
            'budget_national': pathway_result['budget_national'],
            'budget_industry': industry_budget_val,
            'budget_transformation': transformation_budget_val,
            'budget_other': other_budget_val,
            'validation': pathway_result['validation'],
            'milestones': pathway_result['milestones']
        }
        if petrochem_path is not None:
            export_payload['pathway_petrochem'] = petrochem_path.tolist()
            export_payload['budget_petrochem'] = petrochem_budget_val

        json_data = json.dumps(export_payload, indent=2, default=float)
        st.download_button(
            "📥 Download JSON",
            json_data,
            f"pathway_multisector_{pathway_result['curve_type']}.json",
            "application/json"
        )

    elif pathway_result:
        st.warning("The loaded pathway uses a legacy structure. Please regenerate the pathway to apply the new sector model.")
    elif compare_curves:
        st.info("Curve comparison is temporarily disabled for the multi-sector configuration.")

# ==================== SUMMARY REPORT ====================
else:  # Summary Report
    st.title("📋 Summary Report")
    st.markdown("Complete analysis summary combining all modules.")

    has_factors = 'custom_factors' in st.session_state
    has_budget = 'budget_results' in st.session_state
    pathway_result = st.session_state.get('pathway_result')
    multi_sector_pathway = pathway_result and pathway_result.get('tier') == 'multi_sector'

    if not (has_factors or has_budget or multi_sector_pathway):
        st.info("👈 Complete the workflow first:\n1. Configure Allocation Factors\n2. Calculate Budget\n3. Generate Pathways")
    else:
        if has_factors:
            st.subheader("1️⃣ Allocation Factors")
            factors = st.session_state.custom_factors
            summary_df = pd.DataFrame({
                'Factor': ['Responsibility', 'Capability', 'Equality'],
                'Your Value (%)': [
                    factors['responsibility'] * 100,
                    factors['capability'] * 100,
                    factors['equality'] * 100
                ],
                'Verified Default (%)': [1.09, 1.47, 0.646]
            })
            st.dataframe(
                summary_df.style.format({'Your Value (%)': '{:.3f}', 'Verified Default (%)': '{:.3f}'}),
                use_container_width=True,
                hide_index=True
            )

        if has_budget:
            st.subheader("2️⃣ Carbon Budget Allocation")
            stats = st.session_state.budget_results['statistics']
            rows = [
                {
                    'Total Budget (Gt CO₂)': stats['korea_total']['median'] / 1e9,
                    'p05': stats['korea_total']['p05'] / 1e9,
                    'p95': stats['korea_total']['p95'] / 1e9
                }
            ]
            budget_summary = pd.DataFrame(rows, index=['Korea Total'])
            st.dataframe(
                budget_summary.style.format({'Total Budget (Gt CO₂)': '{:.2f}', 'p05': '{:.2f}', 'p95': '{:.2f}'}),
                use_container_width=True
            )

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
                st.dataframe(weight_df.style.format({'Share (%)': '{:.2f}'}), use_container_width=True, hide_index=True)

        if multi_sector_pathway:
            st.subheader("3️⃣ Sector Pathway Summary")
            result = pathway_result
            national_budget = result['budget_national'] / 1e9
            industry_budget = result['budget_industry'] / 1e9
            transformation_budget = result['budget_transformation'] / 1e9
            other_budget = result['budget_other'] / 1e9
            petrochem_budget = result.get('budget_petrochem', 0.0) / 1e9

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("National Budget", f"{national_budget:.2f} Gt")
            col2.metric("Industry", f"{industry_budget:.2f} Gt")
            col3.metric("Transformation", f"{transformation_budget:.2f} Gt")
            col4.metric("Other", f"{other_budget:.2f} Gt")
            col5.metric("Petrochemical", f"{petrochem_budget:.2f} Gt")

            start_cols = st.columns(4)
            start_cols[0].metric("National Start", f"{result['pathway_national'][0]/1e6:.0f} Mt/yr")
            start_cols[1].metric("Industry Start", f"{result['pathway_industry'][0]/1e6:.0f} Mt/yr")
            start_cols[2].metric("Transformation Start", f"{result['pathway_transformation'][0]/1e6:.0f} Mt/yr")
            start_cols[3].metric("Other Start", f"{result['pathway_other'][0]/1e6:.0f} Mt/yr")

            validation = result['validation']
            validation_rows = []
            for key, label in [
                ('national', 'National'),
                ('industry', 'Industry'),
                ('transformation', 'Transformation'),
                ('other', 'Other')
            ]:
                val = validation.get(key)
                validation_rows.append({
                    'Sector': label,
                    'Valid': 'Yes' if val and val.get('is_valid', True) else 'No',
                    'Budget Error (%)': val.get('budget_error_pct') if val else None
                })
            if validation.get('petrochem'):
                val = validation['petrochem']
                validation_rows.append({
                    'Sector': 'Petrochemical',
                    'Valid': 'Yes' if val.get('is_valid', True) else 'No',
                    'Budget Error (%)': val.get('budget_error_pct')
                })
            st.dataframe(
                pd.DataFrame(validation_rows).style.format({'Budget Error (%)': '{:.2f}'}),
                use_container_width=True,
                hide_index=True
            )

            milestones = result['milestones']
            milestone_years = milestones['years']
            milestone_rows = []
            for label, key in [
                ('National', 'national'),
                ('Industry', 'industry'),
                ('Transformation', 'transformation'),
                ('Other', 'other')
            ]:
                values = milestones.get(key)
                if values:
                    row = {'Sector': label}
                    for year, value in zip(milestone_years, values):
                        row[str(year)] = value / 1e6 if value is not None else None
                    milestone_rows.append(row)
            if milestones.get('petrochem'):
                row = {'Sector': 'Petrochemical'}
                for year, value in zip(milestone_years, milestones['petrochem']):
                    row[str(year)] = value / 1e6 if value is not None else None
                milestone_rows.append(row)

            if milestone_rows:
                st.markdown("**Milestone Emissions (Mt CO₂/year)**")
                milestone_df = pd.DataFrame(milestone_rows)
                st.dataframe(
                    milestone_df.style.format({col: '{:.1f}' for col in milestone_df.columns if col != 'Sector'}),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("**Cumulative Budget Check**")
            cumulative_gt = np.cumsum(result['pathway_national']) / 1e9
            st.line_chart(pd.DataFrame({'Cumulative Emissions (Gt)': cumulative_gt}, index=result['years']))
            st.caption(
                f"Budget match error: {validation['national']['budget_error_pct']:.2f}%"
                if validation['national'] else "Budget validation unavailable"
            )

        elif pathway_result:
            st.warning("Pathway data uses a legacy structure. Please regenerate the pathway to view the summary.")
