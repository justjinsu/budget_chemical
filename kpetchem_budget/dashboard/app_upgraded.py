"""
Upgraded Streamlit dashboard for Korean petrochemical carbon budget analysis.

This upgraded dashboard features:
- Composite pathway visualization with percentile ribbons
- Full 768-case Monte Carlo system integration
- Real-time uncertainty quantification
- Enhanced KPI cards with milestone tracking
- Interactive filtering and parameter exploration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import Dict, List, Optional, Tuple
import io
import sys
import os
from pathlib import Path

# Setup imports for both local and deployment environments
def setup_imports():
    """Setup imports to work in different environments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

setup_imports()

try:
    from data_layer import load_global_budget, load_demo_industry_data, get_timeline_years
    from parameter_space import ParameterGrid, MonteCarloSampler, get_budget_line_params
    from pathway import PathwayGenerator, BudgetOverflowError, mark_milestones, batch_generate_pathways
    from simulator import HighPerformanceSimulator, VectorizedBudgetAllocator
    from datastore import SimulationDataStore, PercentileResult, quick_percentile_analysis
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data_layer import load_global_budget, load_demo_industry_data, get_timeline_years
    from parameter_space import ParameterGrid, MonteCarloSampler, get_budget_line_params
    from pathway import PathwayGenerator, BudgetOverflowError, mark_milestones, batch_generate_pathways
    from simulator import HighPerformanceSimulator, VectorizedBudgetAllocator
    from datastore import SimulationDataStore, PercentileResult, quick_percentile_analysis


# Configure Streamlit page
st.set_page_config(
    page_title="K-PetChem Carbon Budget Toolkit v2.0",
    page_icon="🏭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stMetric {
        background-color: #1e2530;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 1rem;
    }
    .stSelectbox > div > div {
        background-color: #262730;
    }
    .milestone-marker {
        font-size: 14px;
        font-weight: bold;
        color: #ff6b6b;
    }
    .uncertainty-band {
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'current_emissions': 50.0,
        'allocation_method': 'population',
        'budget_scenario': '1.5C-50%',
        'net_zero_year': 2050,
        'start_year': 2023,
        'enable_monte_carlo': True,
        'enable_percentiles': True,
        'n_mc_samples': 100,
        'selected_pathways': ['linear', 'constant_rate', 'logistic', 'iea_proxy'],
        'simulation_cache': None,
        'last_sim_params': None,
        'percentile_cache': None,
        'show_individual_pathways': True,
        'confidence_level': 90
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_enhanced_sidebar():
    """Create enhanced sidebar with comprehensive controls."""
    st.sidebar.header("🏭 K-PetChem Budget Toolkit v2.0")
    st.sidebar.markdown("**768-case Monte Carlo system**")
    
    with st.sidebar.expander("📊 Basic Parameters", expanded=True):
        st.session_state.current_emissions = st.number_input(
            "2023 baseline emissions (Mt CO₂e/year)",
            min_value=1.0,
            max_value=200.0,
            value=st.session_state.current_emissions,
            step=1.0,
            help="Korean petrochemical sector baseline emissions"
        )
        
        allocation_options = {
            'Population share (~0.66%)': 'population',
            'GDP share (~1.8%)': 'gdp', 
            'Historical GHG (~1.4%)': 'national_ghg',
            'IEA sector pathway': 'iea_sector'
        }
        
        selected_allocation = st.selectbox(
            "Budget allocation method",
            list(allocation_options.keys()),
            index=0,
            help="Method for allocating global carbon budget to Korea"
        )
        st.session_state.allocation_method = allocation_options[selected_allocation]
        
        st.session_state.budget_scenario = st.selectbox(
            "Global budget scenario",
            ['1.5C-67%', '1.5C-50%', '1.7C-50%', '2.0C-67%'],
            index=1,
            help="Temperature target and probability level"
        )
        
        st.session_state.net_zero_year = st.selectbox(
            "Net-zero target year",
            [2045, 2050, 2055],
            index=1,
            help="Year to reach net-zero emissions"
        )
    
    with st.sidebar.expander("🎯 Pathway Selection", expanded=True):
        pathway_options = {
            'Linear decline': 'linear',
            'Constant rate': 'constant_rate', 
            'Logistic decline': 'logistic',
            'IEA proxy': 'iea_proxy'
        }
        
        selected_pathways = st.multiselect(
            "Pathway families to visualize",
            list(pathway_options.keys()),
            default=list(pathway_options.keys()),
            help="Select which pathway families to display"
        )
        st.session_state.selected_pathways = [pathway_options[p] for p in selected_pathways]
        
        st.session_state.show_individual_pathways = st.checkbox(
            "Show individual pathways",
            value=st.session_state.show_individual_pathways,
            help="Display individual pathway lines along with uncertainty bands"
        )
    
    with st.sidebar.expander("🎲 Monte Carlo Options", expanded=False):
        st.session_state.enable_monte_carlo = st.checkbox(
            "Enable Monte Carlo simulation",
            value=st.session_state.enable_monte_carlo,
            help="Run full Monte Carlo uncertainty analysis"
        )
        
        if st.session_state.enable_monte_carlo:
            st.session_state.n_mc_samples = st.selectbox(
                "MC samples per case",
                [10, 25, 50, 100],
                index=3,
                help="Number of Monte Carlo samples per deterministic case"
            )
            
            st.session_state.confidence_level = st.slider(
                "Confidence level (%)",
                min_value=80,
                max_value=95,
                value=st.session_state.confidence_level,
                step=5,
                help="Confidence level for uncertainty bands"
            )
            
            st.session_state.enable_percentiles = st.checkbox(
                "Show percentile ribbons",
                value=st.session_state.enable_percentiles,
                help="Display 5th-95th percentile uncertainty bands"
            )
    
    # Quick actions
    with st.sidebar.expander("⚡ Quick Actions", expanded=False):
        if st.button("🚀 Run Full Simulation"):
            run_full_monte_carlo_simulation()
        
        if st.button("📊 Load Example Results"):
            load_example_simulation_data()
        
        if st.button("🗑️ Clear Cache"):
            clear_simulation_cache()
    
    # File uploader
    with st.sidebar.expander("📁 Data Upload", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload custom emissions data",
            type=['csv'],
            help="CSV with columns: year, production_Mt, direct_CO2_Mt"
        )
        
        if uploaded_file is not None:
            handle_file_upload(uploaded_file)


def run_full_monte_carlo_simulation():
    """Run full Monte Carlo simulation with progress tracking."""
    with st.spinner("🚀 Running Monte Carlo simulation..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create simulator
            simulator = HighPerformanceSimulator(n_workers=4)
            
            # Update progress
            progress_bar.progress(10)
            status_text.text("Initializing parameter space...")
            
            # Run filtered simulation based on current settings
            results = simulator.run_filtered_simulation(
                budget_lines=[st.session_state.budget_scenario],
                allocation_rules=[st.session_state.allocation_method],
                net_zero_years=[st.session_state.net_zero_year],
                pathway_families=st.session_state.selected_pathways,
                n_samples=st.session_state.n_mc_samples
            )
            
            progress_bar.progress(70)
            status_text.text("Computing percentiles...")
            
            # Store results in datastore
            store = SimulationDataStore()
            store.save_results(results)
            
            # Compute percentiles
            years = list(range(2023, 2051))
            percentiles = store.compute_percentiles(years)
            
            # Cache results
            st.session_state.simulation_cache = results
            st.session_state.percentile_cache = percentiles
            
            progress_bar.progress(100)
            status_text.text("✅ Simulation complete!")
            
            st.success(f"Completed {len(results):,} simulations with {sum(1 for r in results if r.success)/len(results)*100:.1f}% success rate")
            
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()


def load_example_simulation_data():
    """Load example simulation data for demonstration."""
    try:
        # Try to load from existing datastore
        analysis = quick_percentile_analysis()
        if analysis.get('status') != 'no_data':
            st.success("📊 Loaded existing simulation data")
            # Load into session state
            store = SimulationDataStore()
            years = list(range(2023, 2051))
            st.session_state.percentile_cache = store.compute_percentiles(years)
        else:
            st.info("No example data available. Run a simulation first.")
    except Exception as e:
        st.warning(f"Could not load example data: {str(e)}")


def clear_simulation_cache():
    """Clear cached simulation results."""
    st.session_state.simulation_cache = None
    st.session_state.percentile_cache = None
    st.session_state.last_sim_params = None
    
    # Clear datastore cache
    try:
        store = SimulationDataStore()
        store.clear_cache()
        st.success("🗑️ Cache cleared successfully")
    except Exception as e:
        st.warning(f"Cache clear warning: {str(e)}")


def handle_file_upload(uploaded_file):
    """Handle custom CSV file upload."""
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['year', 'production_Mt', 'direct_CO2_Mt']
        
        if all(col in df.columns for col in required_cols):
            if df['year'].min() >= 2023:
                st.session_state.uploaded_data = df
                st.success("✅ Custom data uploaded successfully!")
            else:
                st.error("❌ Data must start from 2023 or later")
        else:
            st.error(f"❌ Missing required columns: {required_cols}")
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")


def calculate_deterministic_budget():
    """Calculate budget allocation for current settings."""
    allocator = VectorizedBudgetAllocator()
    
    try:
        budget = allocator.allocate_budget_vectorized(
            allocation_rule=st.session_state.allocation_method,
            budget_line=st.session_state.budget_scenario
        )
        return budget, None
    except Exception as e:
        return 0.0, str(e)


def generate_deterministic_pathways(allocated_budget: float) -> Dict[str, pd.DataFrame]:
    """Generate deterministic pathways for selected families."""
    generator = PathwayGenerator(
        baseline_emissions=st.session_state.current_emissions,
        allocated_budget=allocated_budget,
        start_year=st.session_state.start_year,
        net_zero_year=st.session_state.net_zero_year
    )
    
    pathways = {}
    pathway_generators = {
        'linear': lambda: generator.linear_to_zero(),
        'constant_rate': lambda: generator.constant_rate(5.0),  # Default 5% rate
        'logistic': lambda: generator.logistic_decline(),
        'iea_proxy': lambda: generator.iea_proxy()
    }
    
    for family in st.session_state.selected_pathways:
        if family in pathway_generators:
            try:
                pathways[family] = pathway_generators[family]()
            except Exception as e:
                st.warning(f"Failed to generate {family} pathway: {str(e)}")
    
    return pathways


def create_composite_visualization(pathways: Dict[str, pd.DataFrame], 
                                 percentiles: Optional[PercentileResult] = None) -> go.Figure:
    """Create composite pathway visualization with uncertainty bands."""
    fig = go.Figure()
    
    # Color scheme for pathway families
    colors = {
        'linear': '#ff6b6b',
        'constant_rate': '#4ecdc4', 
        'logistic': '#45b7d1',
        'iea_proxy': '#96ceb4'
    }
    
    pathway_names = {
        'linear': 'Linear decline',
        'constant_rate': 'Constant rate',
        'logistic': 'Logistic decline', 
        'iea_proxy': 'IEA proxy'
    }
    
    # Add percentile uncertainty bands if available
    if percentiles is not None and st.session_state.enable_percentiles:
        # 95% confidence band (5th-95th percentile)
        fig.add_trace(go.Scatter(
            x=list(percentiles.years) + list(percentiles.years[::-1]),
            y=list(percentiles.p5) + list(percentiles.p95[::-1]),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% confidence',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # 80% confidence band (10th-90th percentile)
        fig.add_trace(go.Scatter(
            x=list(percentiles.years) + list(percentiles.years[::-1]),
            y=list(percentiles.p10) + list(percentiles.p90[::-1]),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% confidence',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # 50% confidence band (25th-75th percentile)
        fig.add_trace(go.Scatter(
            x=list(percentiles.years) + list(percentiles.years[::-1]),
            y=list(percentiles.p25) + list(percentiles.p75[::-1]),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            name='50% confidence',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Median line
        fig.add_trace(go.Scatter(
            x=percentiles.years,
            y=percentiles.p50,
            mode='lines',
            line=dict(color='white', width=3, dash='dot'),
            name='Median (Monte Carlo)',
            showlegend=True
        ))
    
    # Add individual pathway lines if enabled
    if st.session_state.show_individual_pathways:
        for family, pathway_df in pathways.items():
            if family in colors:
                fig.add_trace(go.Scatter(
                    x=pathway_df['year'],
                    y=pathway_df['emission'],
                    mode='lines+markers',
                    line=dict(color=colors[family], width=3),
                    marker=dict(size=6),
                    name=pathway_names.get(family, family),
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{pathway_names.get(family, family)}</b><br>"
                        "Year: %{x}<br>"
                        "Emissions: %{y:.1f} Mt CO₂e<br>"
                        "<extra></extra>"
                    )
                ))
    
    # Add milestone markers
    milestone_years = [2035, 2050]
    for year in milestone_years:
        fig.add_vline(
            x=year,
            line=dict(color='orange', width=2, dash='dash'),
            annotation=dict(
                text=f"{year} target",
                textangle=90,
                yanchor='bottom'
            )
        )
    
    # Update layout for dark theme
    fig.update_layout(
        title={
            'text': "Korean Petrochemical Carbon Budget Pathways (2023-2050)",
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis=dict(
            title="Year",
            gridcolor='rgba(128, 128, 128, 0.2)',
            color='white'
        ),
        yaxis=dict(
            title="Annual Emissions (Mt CO₂e/year)",
            gridcolor='rgba(128, 128, 128, 0.2)',
            color='white'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='rgba(128, 128, 128, 0.5)',
            borderwidth=1
        ),
        hovermode='x unified',
        height=600
    )
    
    return fig


def create_kpi_dashboard(pathways: Dict[str, pd.DataFrame], 
                        allocated_budget: float,
                        percentiles: Optional[PercentileResult] = None) -> None:
    """Create KPI cards dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Allocated Budget",
            value=f"{allocated_budget:.0f} Mt CO₂e",
            help="Total carbon budget for 2023-2050 period"
        )
    
    with col2:
        if percentiles is not None:
            # Use median values from Monte Carlo
            emissions_2035_idx = list(percentiles.years).index(2035)
            median_2035 = percentiles.p50[emissions_2035_idx]
            st.metric(
                label="🏁 2035 Target (Median)",
                value=f"{median_2035:.1f} Mt CO₂e",
                help="Median emission level in 2035 from Monte Carlo"
            )
        elif pathways:
            # Use first available pathway
            first_pathway = next(iter(pathways.values()))
            emissions_2035 = first_pathway[first_pathway['year'] == 2035]['emission'].iloc[0]
            st.metric(
                label="🏁 2035 Emissions",
                value=f"{emissions_2035:.1f} Mt CO₂e",
                help="Emission level in 2035"
            )
    
    with col3:
        if percentiles is not None:
            emissions_2050_idx = list(percentiles.years).index(2050)
            median_2050 = percentiles.p50[emissions_2050_idx]
            st.metric(
                label="🌟 2050 Target (Median)",
                value=f"{median_2050:.1f} Mt CO₂e",
                help="Median emission level in 2050 from Monte Carlo"
            )
        elif pathways:
            first_pathway = next(iter(pathways.values()))
            emissions_2050 = first_pathway[first_pathway['year'] == 2050]['emission'].iloc[0]
            st.metric(
                label="🌟 2050 Emissions",
                value=f"{emissions_2050:.1f} Mt CO₂e",
                help="Emission level in 2050"
            )
    
    with col4:
        if percentiles is not None:
            # Calculate uncertainty range at 2035
            uncertainty_2035 = percentiles.p95[emissions_2035_idx] - percentiles.p5[emissions_2035_idx]
            st.metric(
                label="📊 Uncertainty (2035)",
                value=f"±{uncertainty_2035/2:.1f} Mt CO₂e",
                help="Half-width of 95% confidence interval in 2035"
            )
        elif pathways and len(pathways) > 1:
            # Show range across pathways
            emissions_2035_all = [
                pathway[pathway['year'] == 2035]['emission'].iloc[0] 
                for pathway in pathways.values()
            ]
            range_2035 = max(emissions_2035_all) - min(emissions_2035_all)
            st.metric(
                label="📈 Pathway Range (2035)",
                value=f"{range_2035:.1f} Mt CO₂e",
                help="Range across pathway families in 2035"
            )


def create_summary_statistics(pathways: Dict[str, pd.DataFrame],
                             percentiles: Optional[PercentileResult] = None) -> None:
    """Create summary statistics table."""
    st.subheader("📋 Pathway Summary Statistics")
    
    if percentiles is not None:
        # Monte Carlo summary
        summary_data = []
        years_of_interest = [2023, 2035, 2050]
        
        for year in years_of_interest:
            year_idx = list(percentiles.years).index(year)
            summary_data.append({
                'Year': year,
                'Median (Mt CO₂e)': f"{percentiles.p50[year_idx]:.1f}",
                '5th percentile': f"{percentiles.p5[year_idx]:.1f}",
                '95th percentile': f"{percentiles.p95[year_idx]:.1f}",
                'Mean': f"{percentiles.mean[year_idx]:.1f}",
                'Std Dev': f"{percentiles.std[year_idx]:.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Additional Monte Carlo statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Simulations", f"{percentiles.n_simulations:,}")
        with col2:
            computation_time = percentiles.metadata.get('computation_time', 0)
            st.metric("Computation Time", f"{computation_time:.2f}s")
    
    elif pathways:
        # Deterministic pathway summary
        summary_data = []
        
        for name, pathway_df in pathways.items():
            total_emissions = pathway_df['emission'].sum()
            emissions_2035 = pathway_df[pathway_df['year'] == 2035]['emission'].iloc[0]
            emissions_2050 = pathway_df[pathway_df['year'] == 2050]['emission'].iloc[0]
            
            summary_data.append({
                'Pathway': name.replace('_', ' ').title(),
                'Total Emissions (Mt CO₂e)': f"{total_emissions:.1f}",
                '2035 Emissions': f"{emissions_2035:.1f}",
                '2050 Emissions': f"{emissions_2050:.1f}",
                'Peak Emission': f"{pathway_df['emission'].max():.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)


def create_download_section(pathways: Dict[str, pd.DataFrame],
                           percentiles: Optional[PercentileResult] = None) -> None:
    """Create download section for results."""
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if pathways:
            # Download pathway data
            combined_pathways = []
            for name, pathway_df in pathways.items():
                pathway_df_copy = pathway_df.copy()
                pathway_df_copy['pathway_family'] = name
                combined_pathways.append(pathway_df_copy)
            
            if combined_pathways:
                combined_df = pd.concat(combined_pathways, ignore_index=True)
                csv_pathways = combined_df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Pathways (CSV)",
                    data=csv_pathways,
                    file_name="kpetchem_pathways.csv",
                    mime="text/csv"
                )
    
    with col2:
        if percentiles is not None:
            # Download percentile data
            percentile_df = pd.DataFrame({
                'year': percentiles.years,
                'p5': percentiles.p5,
                'p10': percentiles.p10,
                'p25': percentiles.p25,
                'p50': percentiles.p50,
                'p75': percentiles.p75,
                'p90': percentiles.p90,
                'p95': percentiles.p95,
                'mean': percentiles.mean,
                'std': percentiles.std
            })
            
            csv_percentiles = percentile_df.to_csv(index=False)
            
            st.download_button(
                label="🎲 Download Percentiles (CSV)",
                data=csv_percentiles,
                file_name="kpetchem_percentiles.csv",
                mime="text/csv"
            )
    
    with col3:
        # Download configuration
        config_data = {
            'baseline_emissions': st.session_state.current_emissions,
            'allocation_method': st.session_state.allocation_method,
            'budget_scenario': st.session_state.budget_scenario,
            'net_zero_year': st.session_state.net_zero_year,
            'selected_pathways': st.session_state.selected_pathways,
            'monte_carlo_enabled': st.session_state.enable_monte_carlo,
            'n_mc_samples': st.session_state.n_mc_samples,
            'confidence_level': st.session_state.confidence_level
        }
        
        config_json = pd.Series(config_data).to_json(indent=2)
        
        st.download_button(
            label="⚙️ Download Config (JSON)",
            data=config_json,
            file_name="kpetchem_config.json",
            mime="application/json"
        )


def main():
    """Main dashboard application."""
    st.title("🏭 Korean Petrochemical Carbon Budget Toolkit v2.0")
    st.markdown("**Advanced Monte Carlo simulation for emission pathway planning (2023-2050)**")
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    create_enhanced_sidebar()
    
    # Calculate budget allocation
    allocated_budget, budget_error = calculate_deterministic_budget()
    
    if budget_error:
        st.error(f"Budget calculation error: {budget_error}")
        return
    
    # Generate deterministic pathways
    pathways = generate_deterministic_pathways(allocated_budget)
    
    if not pathways:
        st.warning("No valid pathways could be generated with current settings.")
        return
    
    # Main content area
    # KPI Dashboard
    create_kpi_dashboard(pathways, allocated_budget, st.session_state.percentile_cache)
    
    # Main visualization
    st.subheader("📈 Composite Pathway Visualization")
    
    fig = create_composite_visualization(pathways, st.session_state.percentile_cache)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Summary Statistics", "🎯 Milestone Analysis", "📥 Downloads"])
    
    with tab1:
        create_summary_statistics(pathways, st.session_state.percentile_cache)
    
    with tab2:
        st.subheader("🎯 Milestone Analysis")
        if st.session_state.percentile_cache is not None:
            # Monte Carlo milestone analysis
            percentiles = st.session_state.percentile_cache
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**2035 Milestone Distribution**")
                year_2035_idx = list(percentiles.years).index(2035)
                emissions_2035 = {
                    '5th percentile': percentiles.p5[year_2035_idx],
                    '25th percentile': percentiles.p25[year_2035_idx],
                    'Median': percentiles.p50[year_2035_idx],
                    '75th percentile': percentiles.p75[year_2035_idx],
                    '95th percentile': percentiles.p95[year_2035_idx]
                }
                
                for label, value in emissions_2035.items():
                    st.write(f"• {label}: {value:.1f} Mt CO₂e")
            
            with col2:
                st.markdown("**2050 Milestone Distribution**")
                year_2050_idx = list(percentiles.years).index(2050)
                emissions_2050 = {
                    '5th percentile': percentiles.p5[year_2050_idx],
                    '25th percentile': percentiles.p25[year_2050_idx],
                    'Median': percentiles.p50[year_2050_idx],
                    '75th percentile': percentiles.p75[year_2050_idx],
                    '95th percentile': percentiles.p95[year_2050_idx]
                }
                
                for label, value in emissions_2050.items():
                    st.write(f"• {label}: {value:.1f} Mt CO₂e")
        else:
            st.info("Run Monte Carlo simulation to see milestone uncertainty analysis.")
    
    with tab3:
        create_download_section(pathways, st.session_state.percentile_cache)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **K-PetChem Carbon Budget Toolkit v2.0** | 
    768-case parameter grid | 
    Monte Carlo uncertainty quantification | 
    2023-2050 timeline
    """)


if __name__ == "__main__":
    main()