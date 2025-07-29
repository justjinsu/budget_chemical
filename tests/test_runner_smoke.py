"""
Smoke test for Monte Carlo runner.

Tests that the main runner can execute end-to-end with small n_draws
without errors and produces expected outputs.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
import sys
import os

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'model_D_MC'))

from model_D_MC.mc_runner import run_monte_carlo_analysis


class TestRunnerSmoke:
    """Smoke test suite for Monte Carlo runner."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_config = {
            'n_draws': 10,  # Small number for fast testing
            'seed': 42,
            'output_dir': 'test_outputs',
            
            'global_budget': {
                'low': 4.5e11,
                'mid': 5.0e11,
                'high': 5.5e11
            },
            
            'user_weights': {
                'responsibility': 0.30,
                'capability': 0.40,
                'equality': 0.30
            },
            
            'uncertainty': {
                'responsibility': {
                    'low': 0.0095,
                    'mid': 0.0110,
                    'high': 0.0128
                },
                'capability': {
                    'mu': 0.017,
                    'sd_pct': 0.05
                },
                'equality': {
                    'mu': 0.0067,
                    'sd_pct': 0.03
                }
            },
            
            'industry_fraction': 0.37,
            'petrochem_fraction': 0.10,
            
            'timeline': {
                'start_year': 2024,
                'mid_year': 2035,
                'end_year': 2050
            },
            
            'curve_types': ['linear', 'exp', 'log'],
            
            'lambda_dist': {
                'type': 'beta',
                'a': 2.0,
                'b': 2.0
            },
            
            'solver': {
                'tolerance': 1e-3,
                'max_iter': 60
            },
            
            'outputs': {
                'quantiles': [0.05, 0.25, 0.50, 0.75, 0.95],
                'save_raw_paths': False,
                'create_plots': False  # Disable plots for faster testing
            },
            
            'base_emissions': {
                'industry': 185.0e6,
                'petrochem': 50.0e6
            },
            
            'logging': {
                'level': 'INFO',
                'progress_interval': 5
            }
        }
    
    def test_runner_completes_successfully(self):
        """Test that runner completes without errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            # Update output directory to use temp directory
            self.test_config['output_dir'] = str(Path(temp_dir) / 'outputs')
            
            with open(config_path, 'w') as f:
                yaml.dump(self.test_config, f)
            
            # Run analysis - should not raise any exceptions
            try:
                run_monte_carlo_analysis(str(config_path))
            except Exception as e:
                pytest.fail(f"Runner failed with error: {e}")
    
    def test_required_outputs_created(self):
        """Test that runner creates all required output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            output_dir = Path(temp_dir) / 'outputs'
            
            self.test_config['output_dir'] = str(output_dir)
            
            with open(config_path, 'w') as f:
                yaml.dump(self.test_config, f)
            
            # Run analysis
            run_monte_carlo_analysis(str(config_path))
            
            # Check that output directory was created
            assert output_dir.exists(), "Output directory not created"
            
            # Check for required output files
            required_files = [
                'industry_fan_quantiles.csv',
                'petrochem_fan_quantiles.csv'
            ]
            
            for filename in required_files:
                file_path = output_dir / filename
                assert file_path.exists(), f"Required output file missing: {filename}"
                assert file_path.stat().st_size > 0, f"Output file is empty: {filename}"
            
            # Check for summary JSON file (should have timestamp in name)
            json_files = list(output_dir.glob('summary_*.json'))
            assert len(json_files) > 0, "No summary JSON file created"
            
            # Verify JSON file is valid
            with open(json_files[0], 'r') as f:
                summary = json.load(f)
                assert 'analysis_metadata' in summary
                assert 'budget_statistics' in summary
                assert 'timeline' in summary
    
    def test_quantile_files_structure(self):
        """Test that quantile CSV files have correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            output_dir = Path(temp_dir) / 'outputs'
            
            self.test_config['output_dir'] = str(output_dir)
            
            with open(config_path, 'w') as f:
                yaml.dump(self.test_config, f)
            
            run_monte_carlo_analysis(str(config_path))
            
            # Check industry quantiles
            industry_file = output_dir / 'industry_fan_quantiles.csv'
            import pandas as pd
            industry_df = pd.read_csv(industry_file)
            
            # Should have year column plus quantile columns
            expected_columns = ['year'] + [f'p{int(q*100):02d}' for q in self.test_config['outputs']['quantiles']]
            for col in expected_columns:
                assert col in industry_df.columns, f"Missing column: {col}"
            
            # Should have correct number of years
            expected_years = self.test_config['timeline']['end_year'] - self.test_config['timeline']['start_year'] + 1
            assert len(industry_df) == expected_years, f"Wrong number of years: {len(industry_df)} vs {expected_years}"
            
            # Years should be in correct range
            assert industry_df['year'].min() == self.test_config['timeline']['start_year']
            assert industry_df['year'].max() == self.test_config['timeline']['end_year']
            
            # Quantiles should be non-negative and properly ordered
            quantile_cols = [col for col in industry_df.columns if col.startswith('p')]
            for _, row in industry_df.iterrows():
                quantile_values = [row[col] for col in quantile_cols]
                assert all(v >= 0 for v in quantile_values), "Negative emissions found"
                assert quantile_values == sorted(quantile_values), "Quantiles not properly ordered"
    
    def test_summary_json_structure(self):
        """Test that summary JSON has correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            output_dir = Path(temp_dir) / 'outputs'
            
            self.test_config['output_dir'] = str(output_dir)
            
            with open(config_path, 'w') as f:
                yaml.dump(self.test_config, f)
            
            run_monte_carlo_analysis(str(config_path))
            
            # Find and load summary file
            json_files = list(output_dir.glob('summary_*.json'))
            with open(json_files[0], 'r') as f:
                summary = json.load(f)
            
            # Check required top-level keys
            required_keys = [
                'analysis_metadata',
                'timeline', 
                'budget_statistics',
                'validation'
            ]
            
            for key in required_keys:
                assert key in summary, f"Missing top-level key: {key}"
            
            # Check metadata structure
            metadata = summary['analysis_metadata']
            assert 'n_draws' in metadata
            assert 'runtime_seconds' in metadata
            assert 'success_rate' in metadata
            assert metadata['n_draws'] == self.test_config['n_draws']
            
            # Check budget statistics structure
            budget_stats = summary['budget_statistics']
            assert 'industry' in budget_stats
            assert 'petrochem' in budget_stats
            
            for sector in ['industry', 'petrochem']:
                sector_stats = budget_stats[sector]
                required_stats = ['mean', 'median', 'p05', 'p95']
                for stat in required_stats:
                    assert stat in sector_stats, f"Missing {stat} for {sector}"
                    assert isinstance(sector_stats[stat], (int, float)), f"{stat} is not numeric"
                    assert sector_stats[stat] > 0, f"{stat} should be positive"
    
    def test_different_configurations(self):
        """Test runner with different configuration parameters."""
        test_configs = [
            {'n_draws': 5, 'curve_types': ['linear']},
            {'n_draws': 8, 'curve_types': ['exp', 'log']},
            {'n_draws': 3, 'industry_fraction': 0.5, 'petrochem_fraction': 0.2}
        ]
        
        for config_override in test_configs:
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / 'test_config.yaml'
                output_dir = Path(temp_dir) / 'outputs'
                
                # Create modified config
                test_config = self.test_config.copy()
                test_config.update(config_override)
                test_config['output_dir'] = str(output_dir)
                
                with open(config_path, 'w') as f:
                    yaml.dump(test_config, f)
                
                # Should run without errors
                try:
                    run_monte_carlo_analysis(str(config_path))
                except Exception as e:
                    pytest.fail(f"Runner failed with config {config_override}: {e}")
                
                # Should create required outputs
                assert (output_dir / 'industry_fan_quantiles.csv').exists()
                assert (output_dir / 'petrochem_fan_quantiles.csv').exists()


if __name__ == "__main__":
    pytest.main([__file__])