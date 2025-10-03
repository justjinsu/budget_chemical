"""
Allocation Factor Calculator

This module provides flexible calculation of allocation factors with customizable
numerators and denominators. Users can adjust country-specific and global values
to explore different equity scenarios.

Key Features:
- Customizable numerators (country values)
- Customizable denominators (global values)
- Multiple calculation methods (direct, cumulative, per-capita)
- Preset configurations for different countries
- Validation and bounds checking
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AllocationCalculator:
    """
    Calculate allocation factors from customizable inputs.

    This class separates the calculation of shares from the budget allocation,
    allowing users to explore different equity assumptions.
    """

    # Verified default values (Korea, Oct 2025)
    DEFAULT_VALUES = {
        'korea': {
            'cumulative_emissions_gt': 27.0,      # Gt CO2 (1850-2021)
            'annual_emissions_mt': 640.0,          # Mt CO2/year (2022)
            'gdp_ppp_trillion': 2.7,              # Trillion USD (2023)
            'gdp_nominal_trillion': 1.71,         # Trillion USD (2023)
            'population_million': 51.7,           # Million (2024)
            'historical_start_year': 1850,
            'historical_end_year': 2021
        },
        'world': {
            'cumulative_emissions_gt': 2500.0,    # Gt CO2 (1850-2021)
            'annual_emissions_gt': 37.5,          # Gt CO2/year (2022)
            'gdp_ppp_trillion': 184.26,           # Trillion USD (2023)
            'gdp_nominal_trillion': 105.69,       # Trillion USD (2023)
            'population_billion': 8.0,            # Billion (2024)
        }
    }

    # Source metadata
    DATA_SOURCES = {
        'cumulative_emissions': 'Statista 2021, Global Carbon Project',
        'annual_emissions': 'IEA 2023',
        'gdp_ppp': 'IMF World Economic Outlook April 2023',
        'gdp_nominal': 'World Bank 2023',
        'population': 'UN World Population Prospects 2024, Worldometer'
    }

    def __init__(self):
        """Initialize allocation calculator with default values."""
        self.country_values = self.DEFAULT_VALUES['korea'].copy()
        self.world_values = self.DEFAULT_VALUES['world'].copy()

        logger.info("AllocationCalculator initialized with verified default values")

    def calculate_responsibility_share(
        self,
        country_emissions: Optional[float] = None,
        world_emissions: Optional[float] = None,
        method: str = 'cumulative'
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate responsibility share (historical emissions).

        Args:
            country_emissions: Country's emissions (Gt CO2 for cumulative, Mt/year for annual)
            world_emissions: World's emissions (same units)
            method: 'cumulative' or 'annual'

        Returns:
            Tuple of (share, metadata_dict)
        """
        if method == 'cumulative':
            numerator = country_emissions or self.country_values['cumulative_emissions_gt']
            denominator = world_emissions or self.world_values['cumulative_emissions_gt']
            units = 'Gt CO2 (1850-2021)'
        elif method == 'annual':
            numerator = (country_emissions or self.country_values['annual_emissions_mt']) / 1000  # Mt to Gt
            denominator = world_emissions or self.world_values['annual_emissions_gt']
            units = 'Gt CO2/year (2022)'
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cumulative' or 'annual'")

        share = numerator / denominator

        metadata = {
            'numerator': numerator,
            'denominator': denominator,
            'share': share,
            'share_percent': share * 100,
            'method': method,
            'units': units,
            'source': self.DATA_SOURCES['cumulative_emissions' if method == 'cumulative' else 'annual_emissions']
        }

        logger.debug(f"Responsibility share ({method}): {share:.4%} = {numerator:.2f} / {denominator:.2f}")

        return share, metadata

    def calculate_capability_share(
        self,
        country_gdp: Optional[float] = None,
        world_gdp: Optional[float] = None,
        method: str = 'ppp'
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate capability share (economic capacity).

        Args:
            country_gdp: Country's GDP (trillion USD)
            world_gdp: World's GDP (trillion USD)
            method: 'ppp' (preferred) or 'nominal'

        Returns:
            Tuple of (share, metadata_dict)
        """
        if method == 'ppp':
            numerator = country_gdp or self.country_values['gdp_ppp_trillion']
            denominator = world_gdp or self.world_values['gdp_ppp_trillion']
            label = 'GDP PPP-adjusted'
        elif method == 'nominal':
            numerator = country_gdp or self.country_values['gdp_nominal_trillion']
            denominator = world_gdp or self.world_values['gdp_nominal_trillion']
            label = 'GDP nominal'
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'ppp' or 'nominal'")

        share = numerator / denominator

        metadata = {
            'numerator': numerator,
            'denominator': denominator,
            'share': share,
            'share_percent': share * 100,
            'method': method,
            'units': f'{label} (trillion USD, 2023)',
            'source': self.DATA_SOURCES[f'gdp_{method}']
        }

        logger.debug(f"Capability share ({method}): {share:.4%} = {numerator:.2f} / {denominator:.2f}")

        return share, metadata

    def calculate_equality_share(
        self,
        country_population: Optional[float] = None,
        world_population: Optional[float] = None,
        method: str = 'current'
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate equality share (population).

        Args:
            country_population: Country's population (millions)
            world_population: World's population (billions)
            method: 'current' or 'projected' (for future year)

        Returns:
            Tuple of (share, metadata_dict)
        """
        numerator = country_population or self.country_values['population_million']
        denominator_billions = world_population or self.world_values['population_billion']
        denominator = denominator_billions * 1000  # Billion to million

        share = numerator / denominator

        metadata = {
            'numerator': numerator,
            'denominator': denominator,
            'share': share,
            'share_percent': share * 100,
            'method': method,
            'units': 'Million people (2024)',
            'source': self.DATA_SOURCES['population']
        }

        logger.debug(f"Equality share ({method}): {share:.4%} = {numerator:.2f} / {denominator:.2f}")

        return share, metadata

    def calculate_all_shares(
        self,
        custom_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Tuple[float, Dict[str, Any]]]:
        """
        Calculate all three allocation shares.

        Args:
            custom_values: Optional dict with custom country/world values

        Returns:
            Dictionary with 'responsibility', 'capability', 'equality' shares
        """
        # Update values if provided
        if custom_values:
            if 'country' in custom_values:
                self.country_values.update(custom_values['country'])
            if 'world' in custom_values:
                self.world_values.update(custom_values['world'])

        results = {
            'responsibility': self.calculate_responsibility_share(method='cumulative'),
            'capability': self.calculate_capability_share(method='ppp'),
            'equality': self.calculate_equality_share(method='current')
        }

        # Log summary
        logger.info("Allocation shares calculated:")
        for factor, (share, _) in results.items():
            logger.info(f"  {factor.capitalize()}: {share:.4%}")

        return results

    def get_preset_country(self, country_code: str) -> Dict[str, Any]:
        """
        Get preset values for different countries.

        Args:
            country_code: ISO 3-letter country code (KOR, USA, CHN, etc.)

        Returns:
            Dictionary with country values
        """
        presets = {
            'KOR': self.DEFAULT_VALUES['korea'],
            'USA': {
                'cumulative_emissions_gt': 500.0,    # ~20% of global
                'annual_emissions_mt': 5000.0,
                'gdp_ppp_trillion': 27.4,
                'gdp_nominal_trillion': 27.4,
                'population_million': 335.0,
            },
            'CHN': {
                'cumulative_emissions_gt': 275.0,    # ~11% of global
                'annual_emissions_mt': 11500.0,
                'gdp_ppp_trillion': 33.0,
                'gdp_nominal_trillion': 19.4,
                'population_million': 1425.0,
            },
            'IND': {
                'cumulative_emissions_gt': 124.0,    # ~5% of global
                'annual_emissions_mt': 2900.0,
                'gdp_ppp_trillion': 13.0,
                'gdp_nominal_trillion': 3.7,
                'population_million': 1428.0,
            },
            'DEU': {
                'cumulative_emissions_gt': 92.0,     # ~3.7% of global
                'annual_emissions_mt': 670.0,
                'gdp_ppp_trillion': 5.5,
                'gdp_nominal_trillion': 4.5,
                'population_million': 84.0,
            },
            'JPN': {
                'cumulative_emissions_gt': 67.0,     # ~2.7% of global
                'annual_emissions_mt': 1050.0,
                'gdp_ppp_trillion': 6.5,
                'gdp_nominal_trillion': 4.2,
                'population_million': 123.0,
            }
        }

        if country_code not in presets:
            logger.warning(f"Country code '{country_code}' not found, using Korea defaults")
            return self.DEFAULT_VALUES['korea']

        return presets[country_code]

    def validate_inputs(
        self,
        country_values: Dict[str, float],
        world_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validate that input values are reasonable.

        Args:
            country_values: Country input values
            world_values: World input values

        Returns:
            Validation results dictionary
        """
        issues = []
        warnings = []

        # Check country emissions don't exceed world
        if country_values.get('cumulative_emissions_gt', 0) > world_values.get('cumulative_emissions_gt', 1e9):
            issues.append("Country cumulative emissions exceed world total")

        if country_values.get('annual_emissions_mt', 0) / 1000 > world_values.get('annual_emissions_gt', 1e9):
            issues.append("Country annual emissions exceed world total")

        # Check GDP
        if country_values.get('gdp_ppp_trillion', 0) > world_values.get('gdp_ppp_trillion', 1e9):
            issues.append("Country GDP exceeds world total")

        # Check population
        country_pop_billions = country_values.get('population_million', 0) / 1000
        if country_pop_billions > world_values.get('population_billion', 1e9):
            issues.append("Country population exceeds world total")

        # Warnings for unusual values
        responsibility_share = country_values.get('cumulative_emissions_gt', 0) / world_values.get('cumulative_emissions_gt', 1)
        if responsibility_share > 0.30:
            warnings.append(f"Responsibility share very high: {responsibility_share:.1%}")

        capability_share = country_values.get('gdp_ppp_trillion', 0) / world_values.get('gdp_ppp_trillion', 1)
        if capability_share > 0.30:
            warnings.append(f"Capability share very high: {capability_share:.1%}")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def export_configuration(self) -> Dict[str, Any]:
        """
        Export current configuration for reproducibility.

        Returns:
            Dictionary with all current values and calculations
        """
        all_shares = self.calculate_all_shares()

        config = {
            'country_values': self.country_values,
            'world_values': self.world_values,
            'calculated_shares': {
                'responsibility': {
                    'share': all_shares['responsibility'][0],
                    'metadata': all_shares['responsibility'][1]
                },
                'capability': {
                    'share': all_shares['capability'][0],
                    'metadata': all_shares['capability'][1]
                },
                'equality': {
                    'share': all_shares['equality'][0],
                    'metadata': all_shares['equality'][1]
                }
            },
            'data_sources': self.DATA_SOURCES,
            'defaults_used': self.DEFAULT_VALUES
        }

        return config
