"""
Streamlit application for Korean petrochemical carbon budget visualization.

This module provides an interactive web interface for exploring carbon budget
allocation scenarios and emission pathways for the Korean petrochemical sector.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io
import base64
from typing import Optional, Dict, Any

from .data_layer import load_global_budget, load_demo_industry_data
from .allocator import BudgetAllocator
from .pathway import PathwayGenerator, BudgetOverflowError


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'current_emissions' not in st.session_state:
        st.session_state.current_emissions = 50.0
    if 'allocation_method' not in st.session_state:
        st.session_state.allocation_method = 'Population'
    if 'reduction_rate' not in st.session_state:
        st.session_state.reduction_rate = 5.0
    if 'show_historical' not in st.session_state:
        st.session_state.show_historical = True
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None


def create_sidebar():
    """Create sidebar with user controls."""
    st.sidebar.header("Configuration")
    
    # Current emissions input
    st.session_state.current_emissions = st.sidebar.number_input(
        "Current annual emissions (Mt CO₂e)",
        min_value=0.0,
        max_value=200.0,
        value=st.session_state.current_emissions,
        step=1.0,
        help="Current Korean petrochemical sector emissions"
    )
    
    # Allocation method selection
    allocation_options = ['Population', 'GDP', 'Historical GHG', 'IEA Sector']
    st.session_state.allocation_method = st.sidebar.radio(
        "Allocation rule",
        allocation_options,
        index=allocation_options.index(st.session_state.allocation_method)
    )
    
    # Reduction rate slider (only shown for constant rate pathway)
    st.session_state.reduction_rate = st.sidebar.slider(
        "Constant reduction rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=st.session_state.reduction_rate,
        step=0.5,
        help="Annual emission reduction rate for constant rate pathway"
    )
    
    # File uploader for custom data
    uploaded_file = st.sidebar.file_uploader(
        "Upload custom 5-year CSV",
        type=['csv'],
        help="CSV with columns: year, production_Mt, direct_CO2_Mt"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.sidebar.success("Custom data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
    
    # Historical data checkbox
    st.session_state.show_historical = st.sidebar.checkbox(
        "Show historical 2019-2023 bars",
        value=st.session_state.show_historical
    )


def calculate_budget_allocation():
    """Calculate budget allocation based on current settings."""
    allocator = BudgetAllocator(st.session_state.current_emissions)
    
    # Map UI method to internal method
    method_map = {
        'Population': 'population',
        'GDP': 'gdp', 
        'Historical GHG': 'historical_ghg',
        'IEA Sector': 'iea_sector'
    }
    
    method = method_map[st.session_state.allocation_method]
    
    try:
        budget = allocator.allocate_budget(method, temp=1.5, probability=0.5)
        return budget, None
    except Exception as e:
        return 0.0, str(e)


def generate_pathways(allocated_budget: float):
    """Generate all pathway scenarios."""
    generator = PathwayGenerator(st.session_state.current_emissions, allocated_budget)
    
    pathways = {}
    errors = {}
    
    # Generate different pathways
    try:
        pathways['Linear to Zero'] = generator.linear_to_zero()
    except BudgetOverflowError as e:
        errors['Linear to Zero'] = str(e)
    
    try:
        pathways['Constant Rate'] = generator.constant_rate(st.session_state.reduction_rate)
    except BudgetOverflowError as e:
        errors['Constant Rate'] = str(e)
    
    try:
        pathways['IEA Proxy'] = generator.iea_proxy()
    except BudgetOverflowError as e:
        errors['IEA Proxy'] = str(e)
    
    return pathways, errors


def display_kpi_cards(allocated_budget: float, pathways: Dict[str, pd.DataFrame]):
    """Display KPI cards with key metrics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Remaining Budget",
            f"{allocated_budget:.1f} Mt CO₂e",
            help="Total carbon budget for 2035-2050"
        )
    
    with col2:
        # Find overshoot year across all pathways
        overshoot_year = None
        for pathway_name, pathway_df in pathways.items():
            generator = PathwayGenerator(st.session_state.current_emissions, allocated_budget)
            summary = generator.get_pathway_summary(pathway_df)
            if summary['overshoot_year'] is not None:
                overshoot_year = summary['overshoot_year']
                break
        
        if overshoot_year:
            st.metric(
                "Overshoot Year",
                str(overshoot_year),
                help="Year when budget is exceeded"
            )
        else:
            st.metric(
                "Overshoot Year",
                "None",
                help="No budget overshoot detected"
            )
    
    with col3:
        # Calculate peak-to-zero reduction for first available pathway
        if pathways:
            first_pathway = next(iter(pathways.values()))
            peak_emission = first_pathway['emission'].max()
            final_emission = first_pathway['emission'].iloc[-1]
            reduction_pct = (peak_emission - final_emission) / peak_emission * 100
            
            st.metric(
                "Peak-to-Zero Drop",
                f"{reduction_pct:.1f}%",
                help="Reduction from peak to final emission"
            )


def create_pathway_chart(pathways: Dict[str, pd.DataFrame], show_historical: bool = True):
    """Create pathway visualization chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Historical data if requested
    if show_historical:
        try:
            historical_data = load_demo_industry_data()
            if not historical_data.empty:
                ax.bar(
                    historical_data['year'], 
                    historical_data['direct_CO2_Mt'],
                    alpha=0.6,
                    color='gray',
                    label='Historical (2019-2023)'
                )
        except Exception:
            pass
    
    # Plot pathways
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (name, pathway_df) in enumerate(pathways.items()):
        ax.plot(
            pathway_df['year'],
            pathway_df['emission'],
            marker='o',
            linewidth=2,
            color=colors[i % len(colors)],
            label=name
        )
    
    # Formatting
    ax.set_xlabel('Year')
    ax.set_ylabel('Emissions (Mt CO₂e/year)')
    ax.set_title('Korean Petrochemical Sector Emission Pathways')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    return fig


def create_download_panel(pathways: Dict[str, pd.DataFrame], chart_fig):
    """Create download panel for data and charts."""
    st.subheader("Downloads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        if pathways:
            # Combine all pathways into single DataFrame
            combined_df = pd.DataFrame()
            for name, pathway_df in pathways.items():
                pathway_copy = pathway_df.copy()
                pathway_copy['pathway'] = name
                combined_df = pd.concat([combined_df, pathway_copy], ignore_index=True)
            
            csv_buffer = io.StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="kpetchem_pathways.csv",
                mime="text/csv"
            )
    
    with col2:
        # PNG download
        if chart_fig:
            img_buffer = io.BytesIO()
            chart_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_data = img_buffer.getvalue()
            
            st.download_button(
                label="Download PNG",
                data=img_data,
                file_name="kpetchem_pathways.png",
                mime="image/png"
            )


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Korean Petrochemical Carbon Budget",
        page_icon="🏭",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("Korean Petrochemical Carbon Budget Allocation")
    st.markdown("""
    This application allocates global carbon budgets to the Korean petrochemical sector 
    using different allocation criteria and generates emission pathways for 2035-2050.
    """)
    
    # Create sidebar
    create_sidebar()
    
    # Main content area
    if st.session_state.current_emissions > 0:
        # Calculate budget allocation
        allocated_budget, error = calculate_budget_allocation()
        
        if error:
            st.error(f"Error calculating budget: {error}")
            return
        
        if allocated_budget > 0:
            # Generate pathways
            pathways, pathway_errors = generate_pathways(allocated_budget)
            
            # Display errors if any
            if pathway_errors:
                st.warning("Some pathways could not be generated:")
                for pathway_name, error_msg in pathway_errors.items():
                    st.write(f"- {pathway_name}: {error_msg}")
            
            if pathways:
                # Display KPI cards
                display_kpi_cards(allocated_budget, pathways)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Pathways Chart", "Data Table", "Downloads"])
                
                with tab1:
                    # Create and display chart
                    chart_fig = create_pathway_chart(
                        pathways, 
                        st.session_state.show_historical
                    )
                    st.pyplot(chart_fig)
                
                with tab2:
                    # Display pathway data
                    st.subheader("Pathway Data")
                    
                    # Create selector for pathway
                    selected_pathway = st.selectbox(
                        "Select pathway to view:",
                        list(pathways.keys())
                    )
                    
                    if selected_pathway:
                        pathway_df = pathways[selected_pathway]
                        st.dataframe(pathway_df)
                        
                        # Show summary statistics
                        generator = PathwayGenerator(st.session_state.current_emissions, allocated_budget)
                        summary = generator.get_pathway_summary(pathway_df)
                        
                        st.subheader("Summary Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Emissions", f"{summary['total_emissions']:.1f} Mt")
                            st.metric("Peak Emission", f"{summary['peak_emission']:.1f} Mt")
                            st.metric("Final Emission", f"{summary['final_emission']:.1f} Mt")
                        
                        with col2:
                            st.metric("Budget Utilization", f"{summary['budget_utilization_pct']:.1f}%")
                            st.metric("Peak-to-Final Reduction", f"{summary['peak_to_final_reduction_pct']:.1f}%")
                            if summary['overshoot_year']:
                                st.metric("Overshoot Year", summary['overshoot_year'])
                
                with tab3:
                    # Download panel
                    create_download_panel(pathways, chart_fig)
            
            else:
                st.error("No valid pathways could be generated. Try adjusting the parameters.")
        
        else:
            st.error("No budget allocated. Check your settings.")
    
    else:
        st.warning("Please set current emissions to a positive value.")


if __name__ == "__main__":
    main()