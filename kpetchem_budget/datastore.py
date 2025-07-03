"""
High-performance data warehouse for Monte Carlo simulation results.

This module provides optimized storage, querying, and percentile computation
for large-scale carbon budget simulation datasets using Parquet format.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
import gzip
import pickle

try:
    from .simulator import SimulationResult
except ImportError:
    from simulator import SimulationResult


@dataclass
class PercentileResult:
    """Percentile computation result for uncertainty visualization."""
    years: np.ndarray
    p5: np.ndarray    # 5th percentile
    p10: np.ndarray   # 10th percentile
    p25: np.ndarray   # 25th percentile
    p50: np.ndarray   # 50th percentile (median)
    p75: np.ndarray   # 75th percentile
    p90: np.ndarray   # 90th percentile
    p95: np.ndarray   # 95th percentile
    mean: np.ndarray
    std: np.ndarray
    n_simulations: int
    metadata: Dict[str, any]


class SimulationDataStore:
    """
    High-performance data warehouse for Monte Carlo simulation results.
    
    Features:
    - Parquet columnar storage with Snappy compression
    - Vectorized percentile computation for uncertainty bands
    - Efficient filtering and aggregation queries
    - Automatic data partitioning and indexing
    - Memory-mapped file access for large datasets
    
    Examples
    --------
    >>> store = SimulationDataStore()
    >>> store.save_results(simulation_results)
    >>> percentiles = store.compute_percentiles(years=range(2023, 2051))
    >>> store.get_results_summary()
    """
    
    def __init__(self, data_dir: str = "simulation_data", compression: str = "snappy"):
        """
        Initialize simulation data store.
        
        Parameters
        ----------
        data_dir : str
            Directory for data storage
        compression : str
            Parquet compression algorithm ('snappy', 'gzip', 'lz4')
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.compression = compression
        
        # File paths
        self.results_file = self.data_dir / "simulation_results.parquet"
        self.pathways_file = self.data_dir / "pathway_timeseries.parquet"
        self.metadata_file = self.data_dir / "metadata.pkl.gz"
        
        # Performance settings
        self.chunk_size = 10000
        self.memory_limit = "1GB"
        
    
    def save_results(self, results: List[SimulationResult], 
                    pathway_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """
        Save simulation results to optimized Parquet storage.
        
        Parameters
        ----------
        results : List[SimulationResult]
            Simulation results to store
        pathway_data : pd.DataFrame, optional
            Time-series pathway data
            
        Returns
        -------
        Dict[str, any]
            Storage statistics and metadata
        """
        start_time = time.time()
        
        # Convert results to DataFrame with optimized dtypes
        results_data = []
        for r in results:
            results_data.append({
                'case_id': r.case_id,
                'sample_id': r.sample_id,
                'budget_line': r.budget_line,
                'allocation_rule': r.allocation_rule,
                'start_year': r.start_year,
                'net_zero_year': r.net_zero_year,
                'pathway_family': r.pathway_family,
                'baseline_emissions': r.baseline_emissions,
                'allocated_budget': r.allocated_budget,
                'total_emissions': r.total_emissions,
                'budget_utilization': r.budget_utilization,
                'emissions_2023': r.emissions_2023,
                'emissions_2035': r.emissions_2035,
                'emissions_2050': r.emissions_2050,
                'cumulative_2035': r.cumulative_2035,
                'cumulative_2050': r.cumulative_2050,
                'peak_emission': r.peak_emission,
                'min_emission': r.min_emission,
                'success': r.success,
                'error_message': r.error_message or ""
            })
        
        df = pd.DataFrame(results_data)
        
        # Optimize data types for storage efficiency
        df = self._optimize_dtypes(df)
        
        # Save to Parquet with optimal settings
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table, 
            self.results_file,
            compression=self.compression,
            use_dictionary=True,  # Compress string columns
            row_group_size=5000,  # Optimize for query performance
            write_statistics=True
        )
        
        # Save pathway time-series if provided
        if pathway_data is not None:
            pathway_table = pa.Table.from_pandas(pathway_data)
            pq.write_table(
                pathway_table,
                self.pathways_file,
                compression=self.compression,
                row_group_size=1000
            )
        
        # Store metadata
        metadata = {
            'n_simulations': len(results),
            'n_successful': sum(1 for r in results if r.success),
            'storage_time': time.time() - start_time,
            'file_size_mb': self.results_file.stat().st_size / 1024 / 1024,
            'timestamp': pd.Timestamp.now(),
            'compression': self.compression
        }
        
        with gzip.open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"💾 Stored {len(results):,} results in {metadata['storage_time']:.1f}s")
        print(f"📁 File size: {metadata['file_size_mb']:.1f} MB")
        
        return metadata
    
    def load_results(self, filters: Optional[Dict[str, any]] = None,
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load simulation results with optional filtering.
        
        Parameters
        ----------
        filters : Dict[str, any], optional
            Column filters to apply
        columns : List[str], optional
            Specific columns to load
            
        Returns
        -------
        pd.DataFrame
            Filtered simulation results
        """
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        # Build Parquet filters for efficient reading
        parquet_filters = []
        if filters:
            for col, values in filters.items():
                if isinstance(values, (list, tuple)):
                    parquet_filters.append((col, 'in', values))
                else:
                    parquet_filters.append((col, '==', values))
        
        # Read with optimizations
        df = pd.read_parquet(
            self.results_file,
            columns=columns,
            filters=parquet_filters if parquet_filters else None,
            engine='pyarrow'
        )
        
        return df
    
    def compute_percentiles(self, 
                          years: Union[range, List[int], np.ndarray],
                          filters: Optional[Dict[str, any]] = None,
                          pathway_data: Optional[pd.DataFrame] = None) -> PercentileResult:
        """
        Compute emission percentiles for uncertainty visualization.
        
        Parameters
        ----------
        years : range, List[int], or np.ndarray
            Years to compute percentiles for
        filters : Dict[str, any], optional
            Filters to apply before computation
        pathway_data : pd.DataFrame, optional
            Pre-loaded pathway data
            
        Returns
        -------
        PercentileResult
            Percentile bands and statistics
        """
        start_time = time.time()
        
        # Load results with filters
        df = self.load_results(filters=filters)
        
        if len(df) == 0:
            raise ValueError("No simulation results match the specified filters")
        
        # If pathway data not provided, try to load from file
        if pathway_data is None and self.pathways_file.exists():
            pathway_data = pd.read_parquet(self.pathways_file)
        
        # Create emission matrix for vectorized percentile computation
        years_array = np.array(years)
        n_years = len(years_array)
        n_sims = len(df)
        
        if pathway_data is not None:
            # Use detailed pathway data if available
            emission_matrix = self._build_emission_matrix_from_pathways(
                pathway_data, df, years_array
            )
        else:
            # Interpolate from milestone data
            emission_matrix = self._interpolate_emission_matrix(df, years_array)
        
        # Vectorized percentile computation
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = np.percentile(emission_matrix, percentiles, axis=0)
        
        result = PercentileResult(
            years=years_array,
            p5=percentile_values[0],
            p10=percentile_values[1],
            p25=percentile_values[2],
            p50=percentile_values[3],
            p75=percentile_values[4],
            p90=percentile_values[5],
            p95=percentile_values[6],
            mean=np.mean(emission_matrix, axis=0),
            std=np.std(emission_matrix, axis=0),
            n_simulations=n_sims,
            metadata={
                'computation_time': time.time() - start_time,
                'filters_applied': filters,
                'data_source': 'pathways' if pathway_data is not None else 'interpolated'
            }
        )
        
        print(f"📊 Computed percentiles for {n_sims:,} simulations in {result.metadata['computation_time']:.2f}s")
        
        return result
    
    def get_results_summary(self) -> Dict[str, any]:
        """
        Get summary statistics for stored simulation results.
        
        Returns
        -------
        Dict[str, any]
            Summary statistics and metadata
        """
        if not self.results_file.exists():
            return {'status': 'no_data', 'message': 'No simulation results found'}
        
        # Load metadata
        metadata = {}
        if self.metadata_file.exists():
            with gzip.open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
        
        # Load results for summary stats
        df = self.load_results()
        
        # Compute summary statistics
        summary = {
            'total_simulations': len(df),
            'successful_simulations': sum(df['success']),
            'success_rate': sum(df['success']) / len(df) * 100,
            'unique_cases': df['case_id'].nunique(),
            'budget_lines': sorted(df['budget_line'].unique()),
            'allocation_rules': sorted(df['allocation_rule'].unique()),
            'pathway_families': sorted(df['pathway_family'].unique()),
            'net_zero_years': sorted(df['net_zero_year'].unique()),
            'budget_utilization': {
                'mean': df['budget_utilization'].mean(),
                'std': df['budget_utilization'].std(),
                'min': df['budget_utilization'].min(),
                'max': df['budget_utilization'].max()
            },
            'total_emissions': {
                'mean': df['total_emissions'].mean(),
                'std': df['total_emissions'].std(),
                'min': df['total_emissions'].min(),
                'max': df['total_emissions'].max()
            },
            'storage_info': metadata,
            'file_size_mb': self.results_file.stat().st_size / 1024 / 1024 if self.results_file.exists() else 0
        }
        
        return summary
    
    def export_filtered_results(self, 
                               output_file: str,
                               filters: Optional[Dict[str, any]] = None,
                               format: str = 'csv') -> str:
        """
        Export filtered results to various formats.
        
        Parameters
        ----------
        output_file : str
            Output file path
        filters : Dict[str, any], optional
            Filters to apply
        format : str
            Output format ('csv', 'parquet', 'excel')
            
        Returns
        -------
        str
            Path to exported file
        """
        df = self.load_results(filters=filters)
        
        if format.lower() == 'csv':
            df.to_csv(output_file, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(output_file, compression='snappy')
        elif format.lower() == 'excel':
            df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📤 Exported {len(df):,} results to {output_file}")
        return output_file
    
    def clear_cache(self) -> None:
        """Remove all cached data files."""
        for file_path in [self.results_file, self.pathways_file, self.metadata_file]:
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️ Removed {file_path.name}")
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for storage efficiency."""
        # Convert string columns to categorical
        categorical_cols = ['budget_line', 'allocation_rule', 'pathway_family']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Optimize integer columns
        int_cols = ['case_id', 'sample_id', 'start_year', 'net_zero_year']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Optimize float columns
        float_cols = [
            'baseline_emissions', 'allocated_budget', 'total_emissions',
            'budget_utilization', 'emissions_2023', 'emissions_2035', 
            'emissions_2050', 'cumulative_2035', 'cumulative_2050',
            'peak_emission', 'min_emission'
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def _build_emission_matrix_from_pathways(self, 
                                           pathway_data: pd.DataFrame,
                                           results_df: pd.DataFrame,
                                           years: np.ndarray) -> np.ndarray:
        """Build emission matrix from detailed pathway data."""
        n_sims = len(results_df)
        n_years = len(years)
        emission_matrix = np.zeros((n_sims, n_years))
        
        # Group pathway data by simulation
        pathway_grouped = pathway_data.groupby(['case_id', 'sample_id'])
        
        for i, (_, result_row) in enumerate(results_df.iterrows()):
            case_id = result_row['case_id']
            sample_id = result_row['sample_id']
            
            if (case_id, sample_id) in pathway_grouped.groups:
                pathway_sim = pathway_grouped.get_group((case_id, sample_id))
                
                # Interpolate emissions for requested years
                emission_matrix[i, :] = np.interp(
                    years, 
                    pathway_sim['year'].values,
                    pathway_sim['emission'].values
                )
        
        return emission_matrix
    
    def _interpolate_emission_matrix(self, 
                                   results_df: pd.DataFrame,
                                   years: np.ndarray) -> np.ndarray:
        """Build emission matrix by interpolating milestone data."""
        n_sims = len(results_df)
        n_years = len(years)
        emission_matrix = np.zeros((n_sims, n_years))
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            # Known points: 2023, 2035, 2050
            known_years = np.array([2023, 2035, 2050])
            known_emissions = np.array([
                row['emissions_2023'],
                row['emissions_2035'],
                row['emissions_2050']
            ])
            
            # Interpolate for all requested years
            emission_matrix[i, :] = np.interp(years, known_years, known_emissions)
        
        return emission_matrix


def create_pathway_timeseries(results: List[SimulationResult],
                            pathway_generator_func) -> pd.DataFrame:
    """
    Create detailed pathway time series for percentile computation.
    
    Parameters
    ----------
    results : List[SimulationResult]
        Simulation results
    pathway_generator_func : callable
        Function to regenerate pathways
        
    Returns
    -------
    pd.DataFrame
        Pathway time series data
    """
    timeseries_data = []
    
    print(f"🔄 Generating pathway time series for {len(results):,} simulations...")
    
    for i, result in enumerate(results):
        if i % 1000 == 0:
            print(f"   Progress: {i:,}/{len(results):,}")
        
        if not result.success:
            continue
            
        try:
            # Regenerate pathway for this simulation
            pathway_df = pathway_generator_func(result)
            
            # Add simulation identifiers
            pathway_df['case_id'] = result.case_id
            pathway_df['sample_id'] = result.sample_id
            
            timeseries_data.append(pathway_df)
            
        except Exception as e:
            warnings.warn(f"Failed to generate pathway for case {result.case_id}, sample {result.sample_id}: {e}")
            continue
    
    if not timeseries_data:
        raise ValueError("No valid pathway time series could be generated")
    
    combined_df = pd.concat(timeseries_data, ignore_index=True)
    print(f"✅ Generated {len(combined_df):,} time series points")
    
    return combined_df


def quick_percentile_analysis(data_dir: str = "simulation_data") -> Dict[str, any]:
    """
    Quick percentile analysis for debugging and validation.
    
    Parameters
    ----------
    data_dir : str
        Data directory path
        
    Returns
    -------
    Dict[str, any]
        Analysis results
    """
    store = SimulationDataStore(data_dir=data_dir)
    
    if not store.results_file.exists():
        return {'status': 'no_data', 'message': 'No simulation data found'}
    
    # Get summary
    summary = store.get_results_summary()
    
    # Compute sample percentiles
    years = range(2023, 2051)
    percentiles = store.compute_percentiles(years)
    
    # Analysis
    analysis = {
        'summary': summary,
        'percentile_bands': {
            'p5_2035': float(percentiles.p5[years.index(2035)]),
            'p50_2035': float(percentiles.p50[years.index(2035)]),
            'p95_2035': float(percentiles.p95[years.index(2035)]),
            'p5_2050': float(percentiles.p5[years.index(2050)]),
            'p50_2050': float(percentiles.p50[years.index(2050)]),
            'p95_2050': float(percentiles.p95[years.index(2050)])
        },
        'uncertainty_metrics': {
            'mean_cv_2035': float(percentiles.std[years.index(2035)] / percentiles.mean[years.index(2035)]),
            'mean_cv_2050': float(percentiles.std[years.index(2050)] / percentiles.mean[years.index(2050)]),
            'range_2035': float(percentiles.p95[years.index(2035)] - percentiles.p5[years.index(2035)]),
            'range_2050': float(percentiles.p95[years.index(2050)] - percentiles.p5[years.index(2050)])
        }
    }
    
    return analysis


if __name__ == "__main__":
    # Quick test and validation
    try:
        analysis = quick_percentile_analysis()
        print("📊 Percentile Analysis Results:")
        print(f"   Total simulations: {analysis['summary']['total_simulations']:,}")
        print(f"   Success rate: {analysis['summary']['success_rate']:.1f}%")
        print(f"   2035 median emission: {analysis['percentile_bands']['p50_2035']:.1f} Mt")
        print(f"   2050 median emission: {analysis['percentile_bands']['p50_2050']:.1f} Mt")
        print(f"   Uncertainty range 2035: {analysis['uncertainty_metrics']['range_2035']:.1f} Mt")
        print(f"   Uncertainty range 2050: {analysis['uncertainty_metrics']['range_2050']:.1f} Mt")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
