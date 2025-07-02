"""
Correlation Comparison Method

This module implements correlation analysis between two data channels,
including Pearson, Spearman, and Kendall correlation coefficients.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class CorrelationComparison(BaseComparison):
    """
    Correlation analysis comparison method.
    
    Computes various correlation coefficients and related statistics
    to assess the linear and monotonic relationships between datasets.
    """
    
    name = "Correlation Analysis"
    description = "Compute Pearson, Spearman, and Kendall correlation coefficients with significance tests"
    category = "Statistical"
    version = "1.0.0"
    
    parameters = {
        'correlation_type': {
            'type': str,
            'default': 'all',
            'choices': ['pearson', 'spearman', 'kendall', 'all'],
            'description': 'Type of correlation to compute'
        },
        'significance_level': {
            'type': float,
            'default': 0.05,
            'min': 0.001,
            'max': 0.5,
            'description': 'Significance level for hypothesis testing'
        },
        'bootstrap_samples': {
            'type': int,
            'default': 1000,
            'min': 100,
            'max': 10000,
            'description': 'Number of bootstrap samples for confidence intervals'
        },
        'compute_confidence_intervals': {
            'type': bool,
            'default': True,
            'description': 'Whether to compute bootstrap confidence intervals'
        }
    }
    
    output_types = ["correlation_statistics", "significance_tests", "confidence_intervals", "plot_data"]
    
    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform correlation analysis between reference and test data.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            ref_time: Optional time array for reference data (unused)
            test_time: Optional time array for test data (unused)
            
        Returns:
            Dictionary containing correlation analysis results
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Initialize results
        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'correlations': {},
            'significance_tests': {},
            'confidence_intervals': {},
            'plot_data': {
                'ref_data': ref_clean,
                'test_data': test_clean
            }
        }
        
        correlation_type = self.params['correlation_type']
        significance_level = self.params['significance_level']
        
        # Compute correlations based on type
        if correlation_type in ['pearson', 'all']:
            results['correlations']['pearson'] = self._compute_pearson(ref_clean, test_clean)
            results['significance_tests']['pearson'] = self._test_significance(
                results['correlations']['pearson'], len(ref_clean), significance_level
            )
        
        if correlation_type in ['spearman', 'all']:
            results['correlations']['spearman'] = self._compute_spearman(ref_clean, test_clean)
            results['significance_tests']['spearman'] = self._test_significance(
                results['correlations']['spearman'], len(ref_clean), significance_level
            )
        
        if correlation_type in ['kendall', 'all']:
            results['correlations']['kendall'] = self._compute_kendall(ref_clean, test_clean)
            results['significance_tests']['kendall'] = self._test_significance(
                results['correlations']['kendall'], len(ref_clean), significance_level
            )
        
        # Compute confidence intervals if requested
        if self.params['compute_confidence_intervals']:
            results['confidence_intervals'] = self._compute_confidence_intervals(
                ref_clean, test_clean, correlation_type
            )
        
        # Add interpretation
        results['interpretation'] = self._interpret_correlations(results['correlations'])
        
        # Store results
        self.results = results
        return results
    
    def _compute_pearson(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute Pearson correlation coefficient."""
        try:
            r, p_value = stats.pearsonr(ref_data, test_data)
            r_squared = r ** 2
            
            return {
                'coefficient': r,
                'p_value': p_value,
                'r_squared': r_squared
            }
        except Exception as e:
            return {
                'coefficient': np.nan,
                'p_value': np.nan,
                'r_squared': np.nan,
                'error': str(e)
            }
    
    def _compute_spearman(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute Spearman rank correlation coefficient."""
        try:
            rho, p_value = stats.spearmanr(ref_data, test_data)
            
            return {
                'coefficient': rho,
                'p_value': p_value
            }
        except Exception as e:
            return {
                'coefficient': np.nan,
                'p_value': np.nan,
                'error': str(e)
            }
    
    def _compute_kendall(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute Kendall's tau correlation coefficient."""
        try:
            tau, p_value = stats.kendalltau(ref_data, test_data)
            
            return {
                'coefficient': tau,
                'p_value': p_value
            }
        except Exception as e:
            return {
                'coefficient': np.nan,
                'p_value': np.nan,
                'error': str(e)
            }
    
    def _test_significance(self, correlation_result: Dict[str, float], n_samples: int, alpha: float) -> Dict[str, Any]:
        """Test statistical significance of correlation."""
        if 'error' in correlation_result:
            return {'error': correlation_result['error']}
        
        p_value = correlation_result.get('p_value', np.nan)
        coefficient = correlation_result.get('coefficient', np.nan)
        
        if np.isnan(p_value) or np.isnan(coefficient):
            return {
                'significant': False,
                'p_value': p_value,
                'alpha': alpha,
                'error': 'Invalid correlation results'
            }
        
        return {
            'significant': p_value < alpha,
            'p_value': p_value,
            'alpha': alpha,
            'effect_size': self._classify_effect_size(abs(coefficient)),
            'degrees_of_freedom': n_samples - 2
        }
    
    def _classify_effect_size(self, abs_correlation: float) -> str:
        """Classify correlation effect size according to Cohen's conventions."""
        if abs_correlation >= 0.5:
            return 'large'
        elif abs_correlation >= 0.3:
            return 'medium'
        elif abs_correlation >= 0.1:
            return 'small'
        else:
            return 'negligible'
    
    def _compute_confidence_intervals(self, ref_data: np.ndarray, test_data: np.ndarray, 
                                    correlation_type: str) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for correlations."""
        n_bootstrap = self.params['bootstrap_samples']
        confidence_level = 1 - self.params['significance_level']
        
        intervals = {}
        
        try:
            if correlation_type in ['pearson', 'all']:
                intervals['pearson'] = self._bootstrap_correlation(
                    ref_data, test_data, 'pearson', n_bootstrap, confidence_level
                )
            
            if correlation_type in ['spearman', 'all']:
                intervals['spearman'] = self._bootstrap_correlation(
                    ref_data, test_data, 'spearman', n_bootstrap, confidence_level
                )
            
            if correlation_type in ['kendall', 'all']:
                intervals['kendall'] = self._bootstrap_correlation(
                    ref_data, test_data, 'kendall', n_bootstrap, confidence_level
                )
        
        except Exception as e:
            intervals['error'] = str(e)
        
        return intervals
    
    def _bootstrap_correlation(self, ref_data: np.ndarray, test_data: np.ndarray, 
                             method: str, n_bootstrap: int, confidence_level: float) -> Dict[str, float]:
        """Compute bootstrap confidence interval for a specific correlation method."""
        correlations = []
        n_samples = len(ref_data)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            ref_boot = ref_data[indices]
            test_boot = test_data[indices]
            
            # Compute correlation
            try:
                if method == 'pearson':
                    r, _ = stats.pearsonr(ref_boot, test_boot)
                elif method == 'spearman':
                    r, _ = stats.spearmanr(ref_boot, test_boot)
                elif method == 'kendall':
                    r, _ = stats.kendalltau(ref_boot, test_boot)
                
                if not np.isnan(r):
                    correlations.append(r)
            except:
                continue
        
        if len(correlations) == 0:
            return {
                'lower': np.nan,
                'upper': np.nan,
                'confidence_level': confidence_level,
                'error': 'No valid bootstrap samples'
            }
        
        # Compute percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(correlations, lower_percentile)
        upper = np.percentile(correlations, upper_percentile)
        
        return {
            'lower': lower,
            'upper': upper,
            'confidence_level': confidence_level,
            'n_bootstrap_samples': len(correlations)
        }
    
    def _interpret_correlations(self, correlations: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Provide interpretation of correlation results."""
        interpretations = {}
        
        for method, result in correlations.items():
            if 'error' in result:
                interpretations[method] = f"Error: {result['error']}"
                continue
            
            coefficient = result.get('coefficient', np.nan)
            if np.isnan(coefficient):
                interpretations[method] = "Invalid correlation coefficient"
                continue
            
            # Magnitude interpretation
            abs_coeff = abs(coefficient)
            if abs_coeff >= 0.9:
                strength = "very strong"
            elif abs_coeff >= 0.7:
                strength = "strong"
            elif abs_coeff >= 0.5:
                strength = "moderate"
            elif abs_coeff >= 0.3:
                strength = "weak"
            else:
                strength = "very weak"
            
            # Direction
            direction = "positive" if coefficient > 0 else "negative"
            
            interpretations[method] = f"{strength} {direction} correlation (r = {coefficient:.3f})"
        
        return interpretations 