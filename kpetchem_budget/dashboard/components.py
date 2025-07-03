"""
Reusable Streamlit UI components for the carbon budget dashboard.

This module provides modular UI components that can be composed into
different dashboard layouts and applications.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import io


def create_kpi_cards(allocated_budget: float, 
                    pathways: Dict[str, pd.DataFrame],
                    layout: str = "horizontal") -> None:
    """
    Create KPI cards showing key metrics and milestones.
    
    Parameters
    ----------
    allocated_budget : float
        Total allocated budget for 2023-2050
    pathways : Dict[str, pd.DataFrame]
        Dictionary of pathway DataFrames
    layout : str
        Layout style ("horizontal" or "vertical")
        
    Examples
    --------
    >>> pathways = {"Linear": pathway_df}
    >>> create_kpi_cards(800.0, pathways)
    """
    if not pathways:
        st.warning("No pathways available for KPI display")
        return
    
    # Use first pathway for milestone calculations
    first_pathway = next(iter(pathways.values()))
    
    # Extract milestone data
    data_2035 = first_pathway[first_pathway['year'] == 2035].iloc[0]
    data_2050 = first_pathway[first_pathway['year'] == 2050].iloc[0]
    baseline_emission = first_pathway.iloc[0]['emission']
    
    if layout == "horizontal":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Budget 2023-50",
                f"{allocated_budget:.0f} Mt",
                help="Total carbon budget allocated for 2023-2050"
            )
        
        with col2:
            reduction_2035 = ((baseline_emission - data_2035['emission']) / baseline_emission * 100)
            st.metric(
                "🎯 2035 Target",
                f"{data_2035['emission']:.1f} Mt/yr",
                delta=f"-{reduction_2035:.1f}%",
                help=f"Cumulative by 2035: {data_2035['cumulative']:.0f} Mt"
            )
        
        with col3:
            reduction_2050 = ((baseline_emission - data_2050['emission']) / baseline_emission * 100)
            st.metric(
                "🏁 2050 Target",
                f"{data_2050['emission']:.1f} Mt/yr", 
                delta=f"-{reduction_2050:.1f}%",
                help=f"Cumulative by 2050: {data_2050['cumulative']:.0f} Mt"
            )
        
        with col4:
            utilization = (first_pathway['emission'].sum() / allocated_budget * 100)
            st.metric(
                "📊 Budget Use",
                f"{utilization:.1f}%",
                help="Percentage of allocated budget utilized"
            )
    
    else:  # vertical layout
        st.metric(
            "💰 Remaining Budget 2023-50",
            f"{allocated_budget:.0f} Mt CO₂e",
            help="Total carbon budget allocated for 2023-2050 period"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            reduction_2035 = ((baseline_emission - data_2035['emission']) / baseline_emission * 100)
            st.metric(
                "🎯 2035 Milestone",
                f"{data_2035['emission']:.1f} Mt/yr",
                delta=f"{reduction_2035:.1f}% reduction",
                help=f"Cumulative by 2035: {data_2035['cumulative']:.0f} Mt"
            )
        
        with col2:
            reduction_2050 = ((baseline_emission - data_2050['emission']) / baseline_emission * 100)
            st.metric(
                "🏁 2050 Milestone",
                f"{data_2050['emission']:.1f} Mt/yr",
                delta=f"{reduction_2050:.1f}% reduction", 
                help=f"Cumulative by 2050: {data_2050['cumulative']:.0f} Mt"
            )


def create_composite_chart(pathways: Dict[str, pd.DataFrame],
                          mc_pathways: Optional[List] = None,
                          theme: str = "dark",
                          height: int = 600,
                          show_milestones: bool = True) -> go.Figure:
    """
    Create composite line chart with all pathways and optional Monte Carlo cloud.
    
    Parameters
    ----------
    pathways : Dict[str, pd.DataFrame]
        Dictionary mapping pathway names to DataFrames
    mc_pathways : List, optional
        List of Monte Carlo pathway arrays for uncertainty visualization
    theme : str
        Chart theme ("dark" or "light")
    height : int
        Chart height in pixels
    show_milestones : bool
        Whether to show milestone markers
        
    Returns
    -------
    go.Figure
        Plotly figure object
        
    Examples
    --------
    >>> pathways = {"Linear": linear_df, "Constant": const_df}
    >>> fig = create_composite_chart(pathways)
    >>> fig.show()
    """
    if not pathways:
        return go.Figure()
    
    # Color scheme for pathways
    colors = {
        'Linear to Zero': '#1f77b4',
        'Constant Rate': '#ff7f0e',
        'Logistic Decline': '#2ca02c', 
        'IEA Proxy': '#d62728'
    }
    
    # Theme settings
    if theme == "dark":
        bg_color = 'rgb(17, 17, 17)'
        text_color = 'white'
        grid_color = 'rgba(255,255,255,0.2)'
    else:
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'rgba(0,0,0,0.1)'
    
    fig = go.Figure()
    
    # Add Monte Carlo uncertainty cloud (background layer)
    if mc_pathways:
        years = np.arange(2023, 2051)
        
        try:
            mc_array = np.array(mc_pathways)
            if len(mc_array) > 0:
                # Calculate percentiles
                p10 = np.percentile(mc_array, 10, axis=0)
                p50 = np.percentile(mc_array, 50, axis=0)
                p90 = np.percentile(mc_array, 90, axis=0)
                
                # Add uncertainty bands
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
                    hovertemplate='Uncertainty Band<br>Year: %{x}<br>Range: %{y:.1f} Mt<extra></extra>'
                ))
                
                # Add median line
                fig.add_trace(go.Scatter(
                    x=years,
                    y=p50,
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    name='MC Median',
                    hovertemplate='MC Median<br>Year: %{x}<br>Emissions: %{y:.1f} Mt<extra></extra>'
                ))
        except Exception:
            pass  # Skip MC visualization if it fails
    
    # Add main pathway lines
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
        
        # Add milestone markers if requested
        if show_milestones:
            milestone_years = [2035, 2050]
            milestone_data = pathway_df[pathway_df['year'].isin(milestone_years)]
            
            if not milestone_data.empty:
                fig.add_trace(go.Scatter(
                    x=milestone_data['year'],
                    y=milestone_data['emission'],
                    mode='markers',
                    name=f'{name} Milestones',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol='circle',
                        line=dict(width=2, color=text_color)
                    ),
                    showlegend=False,
                    hovertemplate=f'<b>{name} - Milestone</b><br>' +
                                 'Year: %{x}<br>' +
                                 'Emissions: %{y:.1f} Mt CO₂e/yr<br>' +
                                 '<extra></extra>'
                ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Korean Petrochemical Emission Pathways (2023-2050)',
            'x': 0.5,
            'font': {'size': 20, 'color': text_color}
        },
        xaxis_title='Year',
        yaxis_title='Annual Emissions (Mt CO₂e/year)',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor=f'rgba(255,255,255,{0.1 if theme == "dark" else 0.9})',
            bordercolor=f'rgba(255,255,255,{0.2 if theme == "dark" else 0.5})',
            borderwidth=1
        ),
        hovermode='x unified',
        height=height
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        tickfont=dict(color=text_color),
        titlefont=dict(color=text_color),
        range=[2023, 2050]
    )
    
    fig.update_yaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        tickfont=dict(color=text_color),
        titlefont=dict(color=text_color),
        rangemode='tozero'
    )
    
    return fig


def create_milestone_markers(pathway_df: pd.DataFrame, 
                           years: List[int] = [2035, 2050]) -> pd.DataFrame:
    """
    Extract milestone data from pathway DataFrame.
    
    Parameters
    ----------
    pathway_df : pd.DataFrame
        Pathway DataFrame with year and emission columns
    years : List[int]
        Milestone years to extract
        
    Returns
    -------
    pd.DataFrame
        Milestone data with enhanced information
        
    Examples
    --------
    >>> milestones = create_milestone_markers(pathway_df)
    >>> 2035 in milestones['year'].values
    True
    """
    milestone_data = pathway_df[pathway_df['year'].isin(years)].copy()
    
    if not milestone_data.empty:
        # Add reduction percentages relative to 2023
        baseline = pathway_df.iloc[0]['emission']
        milestone_data['reduction_pct'] = (
            (baseline - milestone_data['emission']) / baseline * 100
        )
        
        # Add remaining budget percentage
        total_budget = pathway_df['emission'].sum() + pathway_df.iloc[-1]['budget_left']
        milestone_data['budget_remaining_pct'] = (
            milestone_data['budget_left'] / total_budget * 100
        )
    
    return milestone_data


def create_pathway_comparison_table(pathways: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create comprehensive comparison table for pathways.
    
    Parameters
    ----------
    pathways : Dict[str, pd.DataFrame]
        Dictionary of pathway DataFrames
        
    Returns
    -------
    pd.DataFrame
        Comparison table with key metrics
        
    Examples
    --------
    >>> comparison = create_pathway_comparison_table(pathways)
    >>> 'total_emissions' in comparison.columns
    True
    """
    comparison_data = []
    
    for name, pathway_df in pathways.items():
        # Basic metrics
        total_emissions = pathway_df['emission'].sum()
        baseline = pathway_df.iloc[0]['emission']
        final = pathway_df.iloc[-1]['emission']
        
        # Milestone data
        data_2035 = pathway_df[pathway_df['year'] == 2035].iloc[0] if 2035 in pathway_df['year'].values else None
        data_2050 = pathway_df[pathway_df['year'] == 2050].iloc[0] if 2050 in pathway_df['year'].values else None
        
        metrics = {
            'pathway': name,
            'total_emissions_mt': total_emissions,
            'baseline_2023_mt': baseline,
            'emission_2035_mt': data_2035['emission'] if data_2035 is not None else np.nan,
            'emission_2050_mt': data_2050['emission'] if data_2050 is not None else np.nan,
            'cumulative_2035_mt': data_2035['cumulative'] if data_2035 is not None else np.nan,
            'cumulative_2050_mt': data_2050['cumulative'] if data_2050 is not None else np.nan,
            'reduction_2023_to_2035_pct': (
                (baseline - data_2035['emission']) / baseline * 100 
                if data_2035 is not None else np.nan
            ),
            'reduction_2023_to_2050_pct': (
                (baseline - final) / baseline * 100
            ),
            'peak_emission_mt': pathway_df['emission'].max(),
            'min_emission_mt': pathway_df['emission'].min()
        }
        
        comparison_data.append(metrics)
    
    return pd.DataFrame(comparison_data).round(2)


def create_download_buttons(pathways: Dict[str, pd.DataFrame],
                          comparison_df: Optional[pd.DataFrame] = None,
                          config_data: Optional[Dict] = None) -> None:
    """
    Create download buttons for various data exports.
    
    Parameters
    ----------
    pathways : Dict[str, pd.DataFrame]
        Pathway data to export
    comparison_df : pd.DataFrame, optional
        Comparison table to export
    config_data : Dict, optional
        Configuration data to export
        
    Examples
    --------
    >>> create_download_buttons(pathways, comparison_df, config)
    """
    st.subheader("📥 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Combined pathway data
        if pathways:
            combined_df = pd.DataFrame()
            for name, pathway_df in pathways.items():
                pathway_copy = pathway_df.copy()
                pathway_copy['pathway'] = name
                combined_df = pd.concat([combined_df, pathway_copy], ignore_index=True)
            
            csv_buffer = io.StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📄 Pathway Data (CSV)",
                data=csv_buffer.getvalue(),
                file_name="kpetchem_pathways.csv",
                mime="text/csv",
                help="Download complete pathway data for all scenarios"
            )
    
    with col2:
        # Comparison summary
        if comparison_df is not None:
            summary_buffer = io.StringIO()
            comparison_df.to_csv(summary_buffer, index=False)
            
            st.download_button(
                label="📊 Summary Stats (CSV)",
                data=summary_buffer.getvalue(),
                file_name="pathway_comparison.csv",
                mime="text/csv",
                help="Download pathway comparison summary"
            )
    
    with col3:
        # Configuration export
        if config_data:
            config_df = pd.DataFrame([config_data])
            config_buffer = io.StringIO()
            config_df.to_csv(config_buffer, index=False)
            
            st.download_button(
                label="⚙️ Configuration (CSV)",
                data=config_buffer.getvalue(),
                file_name="analysis_config.csv",
                mime="text/csv",
                help="Download current analysis configuration"
            )


def create_parameter_selector(parameter_grid: Any,
                            default_values: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create parameter selection interface.
    
    Parameters
    ----------
    parameter_grid : ParameterGrid
        Parameter grid object
    default_values : Dict, optional
        Default parameter values
        
    Returns
    -------
    Dict[str, Any]
        Selected parameter values
        
    Examples
    --------
    >>> params = create_parameter_selector(grid)
    >>> 'allocation_rule' in params
    True
    """
    defaults = default_values or {}
    
    st.subheader("🎛️ Parameter Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        allocation_rule = st.selectbox(
            "Allocation Rule",
            parameter_grid.allocation_rules,
            index=parameter_grid.allocation_rules.index(
                defaults.get('allocation_rule', parameter_grid.allocation_rules[0])
            )
        )
        
        budget_line = st.selectbox(
            "Budget Scenario",
            parameter_grid.budget_lines,
            index=parameter_grid.budget_lines.index(
                defaults.get('budget_line', parameter_grid.budget_lines[0])
            )
        )
    
    with col2:
        net_zero_year = st.selectbox(
            "Net-Zero Year",
            parameter_grid.net_zero_years,
            index=parameter_grid.net_zero_years.index(
                defaults.get('net_zero_year', parameter_grid.net_zero_years[1])
            )
        )
        
        pathway_family = st.selectbox(
            "Pathway Family",
            parameter_grid.pathway_families,
            index=parameter_grid.pathway_families.index(
                defaults.get('pathway_family', parameter_grid.pathway_families[0])
            )
        )
    
    return {
        'allocation_rule': allocation_rule,
        'budget_line': budget_line,
        'net_zero_year': net_zero_year,
        'pathway_family': pathway_family
    }


def display_uncertainty_info(mc_pathways: Optional[List],
                           n_samples: int = 100) -> None:
    """
    Display information about Monte Carlo uncertainty analysis.
    
    Parameters
    ----------
    mc_pathways : List, optional
        Monte Carlo pathway results
    n_samples : int
        Number of Monte Carlo samples
        
    Examples
    --------
    >>> display_uncertainty_info(mc_pathways, 100)
    """
    if mc_pathways:
        st.info(
            f"🎲 **Monte Carlo Analysis Active**: Showing uncertainty from {len(mc_pathways)} samples. "
            f"Gray bands represent 10th-90th percentile range from parameter uncertainties."
        )
        
        # Show uncertainty statistics
        if len(mc_pathways) > 0:
            mc_array = np.array(mc_pathways)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                median_2023 = np.median(mc_array[:, 0])
                std_2023 = np.std(mc_array[:, 0])
                st.metric("2023 Uncertainty", f"±{std_2023:.1f} Mt", f"Median: {median_2023:.1f}")
            
            with col2:
                idx_2035 = 12  # 2035 is 12 years from 2023
                if mc_array.shape[1] > idx_2035:
                    median_2035 = np.median(mc_array[:, idx_2035])
                    std_2035 = np.std(mc_array[:, idx_2035])
                    st.metric("2035 Uncertainty", f"±{std_2035:.1f} Mt", f"Median: {median_2035:.1f}")
            
            with col3:
                median_2050 = np.median(mc_array[:, -1])
                std_2050 = np.std(mc_array[:, -1])
                st.metric("2050 Uncertainty", f"±{std_2050:.1f} Mt", f"Median: {median_2050:.1f}")
    else:
        st.info("🎲 **Monte Carlo Analysis**: Enable in sidebar to show uncertainty bands from parameter variations.")