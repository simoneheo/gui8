"""
Agreement Breakdown Comparison Method

This module implements agreement breakdown analysis between two data channels,
showing agreement rates across different value ranges or percentiles using bar charts.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class AgreementBreakdownComparison(BaseComparison):
    """
    Agreement breakdown analysis comparison method.
    
    Analyzes agreement rates between two datasets across different value ranges
    or percentiles, displaying results as a bar chart.
    """
    
    name = "agreement_breakdown"
    description = "Analyze agreement rates across different value ranges using bar charts"
    category = "Agreement Analysis"
    tags = ["bar", "agreement", "breakdown", "percentile", "threshold"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "agreement_threshold", "type": "float", "default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01, "help": "Threshold for considering measurements in agreement"},
        {"name": "breakdown_type", "type": "str", "default": "percentile", "options": ["percentile", "absolute", "relative"], "help": "Type of breakdown analysis"},
        {"name": "n_bins", "type": "int", "default": 10, "min": 5, "max": 50, "help": "Number of bins for breakdown analysis"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before calculating agreement"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "options": ["iqr", "zscore"], "help": "Method for detecting outliers"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "min": 0.5, "max": 0.99, "step": 0.01, "help": "Confidence level for error bars"}
    ]
    
    # Plot configuration
    plot_type = "bar"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'agreement_threshold_line': {'default': True, 'label': 'Agreement Threshold', 'tooltip': 'Horizontal line showing agreement threshold', 'type': 'line'},
        'error_bars': {'default': True, 'label': 'Error Bars', 'tooltip': 'Show confidence intervals on bars', 'type': 'marker'},
        'statistical_results': {'default': True, 'label': 'Statistical Results', 'tooltip': 'Display agreement statistics', 'type': 'text'},
        'value_labels': {'default': False, 'label': 'Value Labels', 'tooltip': 'Show values on top of bars', 'type': 'text'}
    }
    
    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for agreement breakdown analysis
        
        This defines what gets plotted for bar chart visualization.
        
        Args:
            ref_data: Reference measurements (cleaned of NaN/infinite values)
            test_data: Test measurements (cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (categories, values, error_bars, metadata)
                categories: List of bin labels for X-axis
                values: Agreement rates for each bin
                error_bars: Confidence intervals for each bar
                metadata: Plot configuration dictionary
        """
        
        # Handle outlier removal if requested
        if params.get("remove_outliers", False):
            ref_data, test_data = self._remove_outliers(ref_data, test_data, params)
        
        # Get parameters
        agreement_threshold = params.get("agreement_threshold", 0.95)
        breakdown_type = params.get("breakdown_type", "percentile")
        n_bins = params.get("n_bins", 10)
        confidence_level = params.get("confidence_level", 0.95)
        
        # Calculate agreement for each data point
        agreement_rates = np.abs(ref_data - test_data) / np.abs(ref_data)
        agreement_binary = agreement_rates <= (1 - agreement_threshold)
        
        # Create bins based on breakdown type
        if breakdown_type == "percentile":
            # Create bins based on percentiles of reference data
            bin_edges = np.percentile(ref_data, np.linspace(0, 100, n_bins + 1))
            categories = [f"{i*100//n_bins}-{(i+1)*100//n_bins}%" for i in range(n_bins)]
        elif breakdown_type == "absolute":
            # Create bins based on absolute values
            ref_min, ref_max = np.min(ref_data), np.max(ref_data)
            bin_edges = np.linspace(ref_min, ref_max, n_bins + 1)
            categories = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(n_bins)]
        else:  # relative
            # Create bins based on relative ranges
            ref_mean = np.mean(ref_data)
            ref_std = np.std(ref_data)
            bin_edges = np.linspace(ref_mean - 2*ref_std, ref_mean + 2*ref_std, n_bins + 1)
            categories = [f"{(bin_edges[i]-ref_mean)/ref_std:.1f}σ to {(bin_edges[i+1]-ref_mean)/ref_std:.1f}σ" for i in range(n_bins)]
        
        # Calculate agreement rates for each bin
        values = []
        error_bars = []
        
        for i in range(n_bins):
            # Find data points in this bin
            mask = (ref_data >= bin_edges[i]) & (ref_data < bin_edges[i + 1])
            if i == n_bins - 1:  # Include the last edge
                mask = (ref_data >= bin_edges[i]) & (ref_data <= bin_edges[i + 1])
            
            if np.sum(mask) > 0:
                bin_agreement = agreement_binary[mask]
                agreement_rate = np.mean(bin_agreement) * 100  # Convert to percentage
                
                # Calculate confidence interval
                n_samples = len(bin_agreement)
                if n_samples > 1:
                    sem = stats.sem(bin_agreement) * 100
                    confidence_interval = stats.t.interval(confidence_level, n_samples - 1, 
                                                         loc=agreement_rate, scale=sem)
                    error_bar = (confidence_interval[1] - confidence_interval[0]) / 2
                else:
                    error_bar = 0
                
                values.append(agreement_rate)
                error_bars.append(error_bar)
            else:
                values.append(0)
                error_bars.append(0)
        
        # Create metadata
        metadata = {
            'agreement_threshold': agreement_threshold,
            'breakdown_type': breakdown_type,
            'n_bins': n_bins,
            'confidence_level': confidence_level,
            'bin_edges': bin_edges,
            'ylabel': 'Agreement Rate (%)',
            'xlabel': f'{breakdown_type.title()} Bins'
        }
        
        return categories, values, error_bars, metadata
    
    def stats_script(self, categories: List[str], values: List[float], 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> dict:
        """
        Statistical analysis for agreement breakdown
        
        Args:
            categories: Bin categories from plot_script
            values: Agreement rates from plot_script
            ref_data: Reference data
            test_data: Test data
            params: Method parameters
            
        Returns:
            Dictionary containing statistical results
        """
        
        # Calculate overall statistics
        agreement_threshold = params.get("agreement_threshold", 0.95)
        agreement_rates = np.abs(ref_data - test_data) / np.abs(ref_data)
        agreement_binary = agreement_rates <= (1 - agreement_threshold)
        
        overall_agreement = np.mean(agreement_binary) * 100
        n_samples = len(ref_data)
        
        # Calculate agreement statistics
        stats_results = {
            'overall_agreement_rate': overall_agreement,
            'n_samples': n_samples,
            'agreement_threshold': agreement_threshold * 100,
            'best_bin': categories[np.argmax(values)] if values else 'None',
            'worst_bin': categories[np.argmin(values)] if values else 'None',
            'best_agreement_rate': np.max(values) if values else 0,
            'worst_agreement_rate': np.min(values) if values else 0,
            'agreement_std': np.std(values) if values else 0,
            'agreement_range': np.max(values) - np.min(values) if values else 0
        }
        
        # Add bin-specific statistics
        stats_results['bin_statistics'] = {
            'categories': categories,
            'agreement_rates': values,
            'n_bins': len(categories)
        }
        
        return stats_results
    
    def _remove_outliers(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """Remove outliers from the data"""
        outlier_method = params.get("outlier_method", "iqr")
        
        if outlier_method == "iqr":
            # IQR method
            iqr_factor = params.get("iqr_factor", 1.5)
            
            # Calculate IQR for both datasets
            ref_q25, ref_q75 = np.percentile(ref_data, [25, 75])
            ref_iqr = ref_q75 - ref_q25
            ref_lower = ref_q25 - iqr_factor * ref_iqr
            ref_upper = ref_q75 + iqr_factor * ref_iqr
            
            test_q25, test_q75 = np.percentile(test_data, [25, 75])
            test_iqr = test_q75 - test_q25
            test_lower = test_q25 - iqr_factor * test_iqr
            test_upper = test_q75 + iqr_factor * test_iqr
            
            mask = ((ref_data >= ref_lower) & (ref_data <= ref_upper) & 
                   (test_data >= test_lower) & (test_data <= test_upper))
            
        else:  # zscore
            # Z-score method
            z_threshold = params.get("z_threshold", 3.0)
            
            ref_z = np.abs(stats.zscore(ref_data))
            test_z = np.abs(stats.zscore(test_data))
            
            mask = (ref_z < z_threshold) & (test_z < z_threshold)
        
        return ref_data[mask], test_data[mask] 