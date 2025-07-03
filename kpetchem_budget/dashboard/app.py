"""
Interactive Streamlit dashboard for Korean petrochemical carbon budget analysis.

This dashboard provides a comprehensive interface for exploring carbon budget
allocation scenarios and emission pathways (2023-2050) with Monte Carlo
uncertainty visualization.

UPGRADE NOTICE: For the full 768-case Monte Carlo system with composite 
visualization and percentile ribbons, use app_upgraded.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

# Setup imports for both local and deployment environments
def setup_imports():
    """Setup imports to work in different environments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

setup_imports()

# Import modules with robust error handling
try:
    # Try importing from parent directory first
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from data_layer import load_global_budget, load_demo_industry_data, get_timeline_years
    from parameter_space import ParameterGrid, MonteCarloSampler, get_budget_line_params
    from pathway import PathwayGenerator, BudgetOverflowError, mark_milestones
    from simulator import VectorizedBudgetAllocator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are available. Run from the main kpetchem_budget directory.")
    st.stop()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'current_emissions' not in st.session_state:
        st.session_state.current_emissions = 50.0
    if 'allocation_method' not in st.session_state:
        st.session_state.allocation_method = 'Population'
    if 'reduction_rate' not in st.session_state:
        st.session_state.reduction_rate = 5.0
    if 'net_zero_year' not in st.session_state:
        st.session_state.net_zero_year = 2050
    if 'budget_scenario' not in st.session_state:
        st.session_state.budget_scenario = '1.5C-50%'
    if 'enable_monte_carlo' not in st.session_state:
        st.session_state.enable_monte_carlo = False
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None


def create_sidebar():
    """Create sidebar with user controls."""
    st.sidebar.header("🏭 Configuration")
    
    # Current emissions input
    st.session_state.current_emissions = st.sidebar.number_input(
        "Current annual emissions (Mt CO₂e)",
        min_value=0.0,
        max_value=200.0,
        value=st.session_state.current_emissions,
        step=1.0,
        help="Korean petrochemical sector baseline emissions for 2023"
    )
    
    # Allocation method selection
    allocation_options = ['Population', 'GDP', 'National GHG', 'IEA Sector']
    st.session_state.allocation_method = st.sidebar.radio(
        "Allocation rule",
        allocation_options,
        index=allocation_options.index(st.session_state.allocation_method),
        help="Method for allocating global carbon budget to Korea"
    )
    
    # Budget scenario selection
    budget_options = ['1.5C-67%', '1.5C-50%', '2.0C-67%']
    st.session_state.budget_scenario = st.sidebar.selectbox(
        "Budget scenario",
        budget_options,
        index=budget_options.index(st.session_state.budget_scenario),
        help="Global carbon budget scenario (temperature target and probability)"
    )
    
    # Net-zero year selection
    st.session_state.net_zero_year = st.sidebar.selectbox(
        "Net-zero target year",
        [2045, 2050, 2055],
        index=1,  # Default to 2050
        help="Year to reach net-zero emissions"
    )
    
    # Reduction rate slider
    st.session_state.reduction_rate = st.sidebar.slider(
        "Constant reduction rate (%)",
        min_value=1.0,
        max_value=20.0,
        value=st.session_state.reduction_rate,
        step=0.5,
        help="Annual emission reduction rate for constant rate pathway"
    )
    
    # File uploader for custom data
    uploaded_file = st.sidebar.file_uploader(
        "Upload custom CSV (2023+)",
        type=['csv'],
        help="CSV with columns: year, production_Mt, direct_CO2_Mt (must start from 2023)"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'year' in df.columns and df['year'].min() >= 2023:
                st.session_state.uploaded_data = df
                st.sidebar.success("✅ Custom data uploaded!")
            else:
                st.sidebar.error("❌ Data must include 'year' column starting from 2023")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
    
    # Monte Carlo visualization toggle
    st.session_state.enable_monte_carlo = st.sidebar.checkbox(
        "Enable Monte Carlo cloud",
        value=st.session_state.enable_monte_carlo,
        help="Show uncertainty bands from Monte Carlo simulations"
    )


def calculate_budget_allocation():
    """Calculate budget allocation based on current settings."""
    allocator = VectorizedBudgetAllocator()
    
    # Map UI method to internal method
    method_map = {
        'Population': 'population',
        'GDP': 'gdp', 
        'National GHG': 'national_ghg',
        'IEA Sector': 'iea_sector'
    }
    
    method = method_map[st.session_state.allocation_method]
    
    try:
        budget = allocator.allocate_budget_vectorized(
            allocation_rule=method,
            budget_line=st.session_state.budget_scenario
        )
        return budget, None
    except Exception as e:
        return 0.0, str(e)


def generate_all_pathways(allocated_budget: float):
    """Generate all four pathway families."""
    generator = PathwayGenerator(
        baseline_emissions=st.session_state.current_emissions,
        allocated_budget=allocated_budget,
        start_year=2023,
        net_zero_year=st.session_state.net_zero_year
    )
    
    pathways = {}
    errors = {}
    
    # Generate all four pathway families
    pathway_configs = [
        ('Linear to Zero', lambda: generator.linear_to_zero()),
        ('Constant Rate', lambda: generator.constant_rate(st.session_state.reduction_rate)),
        ('Logistic Decline', lambda: generator.logistic_decline()),
        ('IEA Proxy', lambda: generator.iea_proxy())
    ]
    
    for name, pathway_func in pathway_configs:
        try:
            pathway_df = pathway_func()
            pathways[name] = pathway_df
        except BudgetOverflowError as e:
            errors[name] = f"Budget overflow: {str(e)}"
        except Exception as e:
            errors[name] = f"Error: {str(e)}"
    
    return pathways, errors


def generate_monte_carlo_cloud(allocated_budget: float, n_samples: int = 50):
    """Generate Monte Carlo uncertainty cloud for visualization."""
    if not st.session_state.enable_monte_carlo:
        return None
    
    # Create Monte Carlo sampler with fewer samples for real-time visualization
    sampler = MonteCarloSampler(n_samples=n_samples, random_seed=42)
    
    # Generate perturbations
    mc_pathways = []
    
    for i in range(n_samples):
        # Generate one MC sample
        mc_samples = list(sampler.generate_samples(case_id=0))
        if mc_samples:
            mc_sample = mc_samples[i % len(mc_samples)]
            
            # Create perturbed allocator
            allocator = BudgetAllocator()
            temp, prob = get_budget_line_params(st.session_state.budget_scenario)
            
            method_map = {
                'Population': 'population',
                'GDP': 'gdp', 
                'National GHG': 'national_ghg',
                'IEA Sector': 'iea_sector'
            }
            method = method_map[st.session_state.allocation_method]
            
            try:
                perturbed_budget = allocator.allocate_budget(
                    method, 
                    temp, 
                    prob,
                    mc_sample.global_budget_factor,
                    mc_sample.production_share_error
                )
                
                # Generate pathway with perturbed budget
                generator = PathwayGenerator(
                    baseline_emissions=st.session_state.current_emissions,
                    allocated_budget=perturbed_budget,
                    net_zero_year=st.session_state.net_zero_year
                )
                
                # Use linear pathway for MC cloud
                pathway = generator.linear_to_zero()
                mc_pathways.append(pathway['emission'].values)
                
            except Exception:
                continue  # Skip failed samples
    
    return mc_pathways


def display_kpi_cards(allocated_budget: float, pathways: Dict[str, pd.DataFrame]):
    """Display KPI cards with milestone information."""
    if not pathways:
        st.warning("No valid pathways to display KPIs")
        return
    
    # Get first available pathway for milestone extraction
    first_pathway = next(iter(pathways.values()))
    
    # Extract milestone data
    data_2035 = first_pathway[first_pathway['year'] == 2035].iloc[0]
    data_2050 = first_pathway[first_pathway['year'] == 2050].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Remaining Budget 2023-50",
            f"{allocated_budget:.0f} Mt CO₂e",
            help="Total carbon budget allocated for 2023-2050 period"
        )
    
    with col2:
        reduction_2035 = ((first_pathway.iloc[0]['emission'] - data_2035['emission']) / 
                         first_pathway.iloc[0]['emission'] * 100)
        st.metric(
            "2035 Milestone",
            f"{data_2035['emission']:.1f} Mt/yr",
            delta=f"{reduction_2035:.1f}% reduction vs 2023",
            help=f"Emission in 2035: {data_2035['emission']:.1f} Mt/yr\nCumulative by 2035: {data_2035['cumulative']:.0f} Mt"
        )
    
    with col3:
        reduction_2050 = ((first_pathway.iloc[0]['emission'] - data_2050['emission']) / 
                         first_pathway.iloc[0]['emission'] * 100)
        st.metric(
            "2050 Milestone", 
            f"{data_2050['emission']:.1f} Mt/yr",
            delta=f"{reduction_2050:.1f}% reduction vs 2023",
            help=f"Emission in 2050: {data_2050['emission']:.1f} Mt/yr\nCumulative by 2050: {data_2050['cumulative']:.0f} Mt"
        )


def create_composite_pathway_chart(pathways: Dict[str, pd.DataFrame], 
                                  mc_pathways: Optional[List] = None):
    """Create composite line chart with all pathways."""
    if not pathways:
        st.warning("No pathways to display")
        return None
    
    # Create Plotly figure with dark background
    fig = go.Figure()
    
    # Color scheme for pathways
    colors = {
        'Linear to Zero': '#1f77b4',
        'Constant Rate': '#ff7f0e', 
        'Logistic Decline': '#2ca02c',
        'IEA Proxy': '#d62728'
    }
    
    # Add Monte Carlo uncertainty cloud first (background)
    if mc_pathways and st.session_state.enable_monte_carlo:
        years = get_timeline_years()
        
        # Calculate percentiles for uncertainty band
        mc_array = np.array(mc_pathways)
        if len(mc_array) > 0:
            p10 = np.percentile(mc_array, 10, axis=0)
            p90 = np.percentile(mc_array, 90, axis=0)
            
            # Add uncertainty band
            fig.add_trace(go.Scatter(
                x=years,
                y=p90,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=p10,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(128,128,128,0.2)',
                name='MC Uncertainty (10-90%)',
                hoverinfo='skip'
            ))
    
    # Add pathway lines with milestone markers
    for name, pathway_df in pathways.items():
        color = colors.get(name, '#333333')
        
        # Main pathway line
        fig.add_trace(go.Scatter(
            x=pathway_df['year'],
            y=pathway_df['emission'],
            mode='lines',
            name=name,
            line=dict(color=color, width=3),
            hovertemplate=f'<b>{name}</b><br>' +
                         'Year: %{x}<br>' +
                         'Emissions: %{y:.1f} Mt CO₂e/yr<br>' +
                         '<extra></extra>'
        ))
        
        # Add milestone markers (2035 and 2050)
        milestone_data = pathway_df[pathway_df['year'].isin([2035, 2050])]
        
        fig.add_trace(go.Scatter(
            x=milestone_data['year'],
            y=milestone_data['emission'],
            mode='markers',
            name=f'{name} Milestones',
            marker=dict(
                size=10,
                color=color,
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            showlegend=False,
            hovertemplate=f'<b>{name} - Milestone</b><br>' +
                         'Year: %{x}<br>' +
                         'Emissions: %{y:.1f} Mt CO₂e/yr<br>' +
                         '<extra></extra>'
        ))
    
    # Update layout with dark theme
    fig.update_layout(
        title={
            'text': 'Korean Petrochemical Emission Pathways (2023-2050)',
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title='Year',
        yaxis_title='Annual Emissions (Mt CO₂e)',
        plot_bgcolor='rgb(17, 17, 17)',
        paper_bgcolor='rgb(17, 17, 17)',
        font_color='white',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        ),
        hovermode='x unified',
        height=600
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.2)',
        zerolinecolor='rgba(255,255,255,0.2)',
        tickfont=dict(color='white'),
        titlefont=dict(color='white')
    )
    
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.2)',
        zerolinecolor='rgba(255,255,255,0.2)',
        tickfont=dict(color='white'),
        titlefont=dict(color='white')
    )
    
    return fig


def create_data_download_section(pathways: Dict[str, pd.DataFrame]):
    """Create download section for data and charts."""
    if not pathways:
        return
    
    st.subheader("📥 Downloads")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV download - all pathways combined
        if pathways:
            combined_df = pd.DataFrame()
            for name, pathway_df in pathways.items():
                pathway_copy = pathway_df.copy()
                pathway_copy['pathway'] = name
                combined_df = pd.concat([combined_df, pathway_copy], ignore_index=True)
            
            csv_buffer = io.StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"kpetchem_pathways_{st.session_state.allocation_method.lower()}.csv",
                mime="text/csv"
            )
    
    with col2:
        # Summary statistics download
        if pathways:
            generator = PathwayGenerator(
                st.session_state.current_emissions, 
                1000.0  # Dummy budget for summary
            )
            
            summary_data = []
            for name, pathway_df in pathways.items():
                summary = generator.get_pathway_summary(pathway_df)
                summary['pathway'] = name
                summary_data.append(summary)
            
            summary_df = pd.DataFrame(summary_data)
            
            summary_buffer = io.StringIO()
            summary_df.to_csv(summary_buffer, index=False)
            summary_data = summary_buffer.getvalue()
            
            st.download_button(
                label="📊 Download Summary",
                data=summary_data,
                file_name="pathway_summary.csv",
                mime="text/csv"
            )
    
    with col3:
        # Configuration export
        config_data = {
            'current_emissions': st.session_state.current_emissions,
            'allocation_method': st.session_state.allocation_method,
            'budget_scenario': st.session_state.budget_scenario,
            'net_zero_year': st.session_state.net_zero_year,
            'reduction_rate': st.session_state.reduction_rate,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        config_df = pd.DataFrame([config_data])
        config_buffer = io.StringIO()
        config_df.to_csv(config_buffer, index=False)
        config_csv = config_buffer.getvalue()
        
        st.download_button(
            label="⚙️ Download Config",
            data=config_csv,
            file_name="analysis_config.csv",
            mime="text/csv"
        )


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="K-PetChem Carbon Budget",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🏭 Korean Petrochemical Carbon Budget Toolkit")
    st.markdown("""
    **Advanced Monte Carlo simulation engine for carbon budget allocation and 
    emission pathway optimization (2023-2050)**
    
    Explore four allocation criteria and pathway families with real-time uncertainty quantification.
    """)
    
    # Create sidebar
    create_sidebar()
    
    # Main content area
    if st.session_state.current_emissions > 0:
        # Calculate budget allocation
        allocated_budget, error = calculate_budget_allocation()
        
        if error:
            st.error(f"❌ Error calculating budget: {error}")
            return
        
        if allocated_budget > 0:
            # Generate pathways
            with st.spinner("🔄 Generating emission pathways..."):
                pathways, pathway_errors = generate_all_pathways(allocated_budget)
                
                # Generate Monte Carlo cloud if enabled
                mc_pathways = None
                if st.session_state.enable_monte_carlo:
                    mc_pathways = generate_monte_carlo_cloud(allocated_budget)
            
            # Display errors if any
            if pathway_errors:
                with st.expander("⚠️ Pathway Generation Warnings", expanded=False):
                    for pathway_name, error_msg in pathway_errors.items():
                        st.warning(f"**{pathway_name}**: {error_msg}")
            
            if pathways:
                # Display KPI cards
                display_kpi_cards(allocated_budget, pathways)
                
                st.markdown("---")
                
                # Create composite chart
                fig = create_composite_pathway_chart(pathways, mc_pathways)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📈 Pathway Analysis", "📋 Data Table", "📥 Downloads"])
                
                with tab1:
                    # Pathway comparison analysis
                    st.subheader("Pathway Comparison")
                    
                    generator = PathwayGenerator(
                        st.session_state.current_emissions,
                        allocated_budget,
                        net_zero_year=st.session_state.net_zero_year
                    )
                    
                    comparison_df = generator.compare_pathways(pathways)
                    
                    # Display comparison table
                    st.dataframe(
                        comparison_df.round(2),
                        use_container_width=True
                    )
                
                with tab2:
                    # Display pathway data
                    st.subheader("Detailed Pathway Data")
                    
                    # Pathway selector
                    selected_pathway = st.selectbox(
                        "Select pathway to view:",
                        list(pathways.keys())
                    )
                    
                    if selected_pathway and selected_pathway in pathways:
                        pathway_df = pathways[selected_pathway]
                        
                        # Show milestone highlights
                        milestone_data = pathway_df[pathway_df['year'].isin([2035, 2050])]
                        
                        st.markdown("**🎯 Milestone Years**")
                        st.dataframe(
                            milestone_data[['year', 'emission', 'cumulative', 'budget_left']].round(2),
                            use_container_width=True
                        )
                        
                        st.markdown("**📊 Complete Pathway Data**")
                        st.dataframe(
                            pathway_df.round(2),
                            use_container_width=True,
                            height=400
                        )
                
                with tab3:
                    # Download section
                    create_data_download_section(pathways)
            
            else:
                st.error("❌ No valid pathways could be generated. Try adjusting the parameters.")
        
        else:
            st.error("❌ No budget allocated. Check your settings.")
    
    else:
        st.warning("⚠️ Please set current emissions to a positive value.")


if __name__ == "__main__":
    main()