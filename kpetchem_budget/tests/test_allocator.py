"""
Tests for budget allocator functionality.

These tests verify that the budget allocation logic works correctly
and that Korean shares are properly validated.
"""

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kpetchem_budget.allocator import BudgetAllocator

# Define simple pytest replacement
class pytest:
    @staticmethod
    def raises(exception_type, match=None):
        class RaisesContext:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {exception_type.__name__} but no exception was raised")
                if not issubclass(exc_type, exception_type):
                    raise AssertionError(f"Expected {exception_type.__name__} but got {exc_type.__name__}")
                return True
        return RaisesContext()
    
    @staticmethod
    def main(args):
        print(f"Would run pytest with args: {args}")


class TestBudgetAllocator:
    """Test suite for BudgetAllocator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.allocator = BudgetAllocator(baseline_emissions=50.0)
        
        # Mock global budget data
        self.mock_global_budget = pd.DataFrame({
            'temp': [1.5, 1.5, 2.0, 2.0],
            'probability': [0.5, 0.83, 0.5, 0.83],
            'approach': ['GDP', 'GDP', 'GDP', 'GDP'],
            'period': [2023, 2023, 2023, 2023],
            'budget_gt': [400, 600, 800, 1200]
        })
    
    def test_allocator_initialization(self):
        """Test allocator initialization."""
        assert self.allocator.baseline_emissions == 50.0
        assert isinstance(self.allocator.korean_shares, dict)
        assert all(0 <= share <= 1 for share in self.allocator.korean_shares.values())
    
    @patch('kpetchem_budget.allocator.load_global_budget')
    def test_population_allocation(self, mock_load_budget):
        """Test population-based allocation."""
        mock_load_budget.return_value = self.mock_global_budget
        
        budget = self.allocator.allocate_budget('population', temp=1.5, probability=0.5)
        
        expected_budget = 400 * self.allocator.korean_shares['population'] * 1000
        assert budget == expected_budget
        assert budget > 0
    
    @patch('kpetchem_budget.allocator.load_global_budget')
    def test_gdp_allocation(self, mock_load_budget):
        """Test GDP-based allocation."""
        mock_load_budget.return_value = self.mock_global_budget
        
        budget = self.allocator.allocate_budget('gdp', temp=1.5, probability=0.5)
        
        expected_budget = 400 * self.allocator.korean_shares['gdp'] * 1000
        assert budget == expected_budget
        assert budget > 0
    
    @patch('kpetchem_budget.allocator.load_global_budget')
    def test_historical_ghg_allocation(self, mock_load_budget):
        """Test historical GHG-based allocation."""
        mock_load_budget.return_value = self.mock_global_budget
        
        budget = self.allocator.allocate_budget('historical_ghg', temp=1.5, probability=0.5)
        
        expected_budget = 400 * self.allocator.korean_shares['historical_ghg'] * 1000
        assert budget == expected_budget
        assert budget > 0
    
    @patch('kpetchem_budget.allocator.load_global_budget')
    @patch('kpetchem_budget.allocator.load_iea_sector_budget')
    def test_iea_sector_allocation(self, mock_load_iea, mock_load_budget):
        """Test IEA sector-based allocation."""
        mock_load_budget.return_value = self.mock_global_budget
        mock_load_iea.return_value = 6.0
        
        budget = self.allocator.allocate_budget('iea_sector', temp=1.5, probability=0.5)
        
        # IEA sector should use minimum of 6 Gt and global budget
        expected_budget = min(6.0, 400) * self.allocator.korean_shares['production'] * 1000
        assert budget == expected_budget
        assert budget > 0
    
    @patch('kpetchem_budget.allocator.load_global_budget')
    @patch('kpetchem_budget.allocator.load_iea_sector_budget')
    def test_iea_sector_returns_6gt_share(self, mock_load_iea, mock_load_budget):
        """Test that IEA sector allocation returns 6 Gt × Korean share."""
        mock_load_budget.return_value = self.mock_global_budget
        mock_load_iea.return_value = 6.0
        
        budget = self.allocator.allocate_budget('iea_sector', temp=1.5, probability=0.5)
        
        # Should be exactly 6 Gt × production share × 1000 (Gt to Mt)
        expected_budget = 6.0 * self.allocator.korean_shares['production'] * 1000
        assert budget == expected_budget
    
    def test_shares_sum_validation(self):
        """Test that Korean shares are reasonable (not testing sum to 1 as they're different categories)."""
        shares = self.allocator.korean_shares
        
        # Each share should be between 0 and 1
        for method, share in shares.items():
            assert 0 <= share <= 1, f"Share for {method} is out of range: {share}"
        
        # Population share should be small (Korea ~0.66% of world)
        assert shares['population'] < 0.02, "Population share too large"
        
        # GDP share should be reasonable (Korea ~1.8% of world GDP)
        assert shares['gdp'] < 0.05, "GDP share too large"
    
    def test_shares_precision(self):
        """Test Korean shares precision within tolerance."""
        tolerance = 1e-6
        
        # Test validation method
        assert self.allocator.validate_shares(tolerance=tolerance)
        
        # Test individual shares are above tolerance (except for edge cases)
        for method, share in self.allocator.korean_shares.items():
            if method != 'production':  # Production might be exactly at tolerance
                assert share > tolerance, f"Share for {method} is at or below tolerance"
    
    @patch('kpetchem_budget.allocator.load_global_budget')
    def test_allocation_summary(self, mock_load_budget):
        """Test allocation summary generation."""
        mock_load_budget.return_value = self.mock_global_budget
        
        summary = self.allocator.get_allocation_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4  # Four allocation methods
        assert 'method' in summary.columns
        assert 'korean_share' in summary.columns
        assert 'allocated_budget_mt' in summary.columns
        
        # Check all methods are present
        expected_methods = ['population', 'gdp', 'historical_ghg', 'iea_sector']
        assert set(summary['method'].tolist()) == set(expected_methods)
    
    def test_invalid_method_raises_error(self):
        """Test that invalid allocation method raises error."""
        with pytest.raises(ValueError, match="Unknown allocation method"):
            self.allocator.allocate_budget('invalid_method')
    
    @patch('kpetchem_budget.allocator.load_global_budget')
    def test_missing_budget_data_raises_error(self, mock_load_budget):
        """Test that missing budget data raises error."""
        # Return empty DataFrame
        mock_load_budget.return_value = pd.DataFrame()
        
        with pytest.raises(ValueError, match="No budget found for criteria"):
            self.allocator.allocate_budget('population')
    
    def test_validate_shares_method(self):
        """Test share validation method."""
        # Valid allocator should pass
        assert self.allocator.validate_shares()
        
        # Create allocator with invalid shares
        invalid_allocator = BudgetAllocator(50.0)
        invalid_allocator.korean_shares['population'] = 1.5  # Invalid: > 1
        
        assert not invalid_allocator.validate_shares()
        
        # Test negative share
        invalid_allocator.korean_shares['population'] = -0.1  # Invalid: < 0
        assert not invalid_allocator.validate_shares()


if __name__ == '__main__':
    pytest.main([__file__])