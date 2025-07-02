"""
Bland-Altman Comparison Method

This module implements Bland-Altman analysis for assessing agreement
between two measurement methods or instruments.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class BlandAltmanComparison(BaseComparison):
    """
    Bland-Altman analysis for method comparison.
    
    Assesses agreement between two measurement methods by analyzing
    the differences vs. averages of paired measurements.
    """
    
    name = "Bland-Altman Analysis"
    description = "Assess agreement between two measurement methods using Bland-Altman plots and statistics"
    category = "Agreement"
    version = "1.0.0"
    
    parameters = {
        'confidence_level': {
            'type': float,
            'default': 0.95,
            'min': 0.8,
            'max': 0.99,
            'description': 'Confidence level for limits of agreement'
        },
        'percentage_difference': {
            'type': bool,
            'default': False,
            'description': 'Calculate percentage differences instead of absolute differences'
        },
        'log_transform': {
            'type': bool,
            'default': False,
            'description': 'Apply log transformation before analysis (for proportional bias)'
        },
        'outlier_detection': {
            'type': bool,
            'default': True,
            'description': 'Detect and report outliers beyond limits of agreement'
        }
    }
    
    output_types = ["agreement_statistics", "limits_of_agreement", "bias_analysis", "plot_data"]
    
    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform Bland-Altman analysis between reference and test data.
        
        Args:
            ref_data: Reference method data
            test_data: Test method data
            ref_time: Optional time array (unused)
            test_time: Optional time array (unused)
            
        Returns:
            Dictionary containing Bland-Altman analysis results
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Apply transformations if requested
        if self.params['log_transform']:
            if np.any(ref_clean <= 0) or np.any(test_clean <= 0):
                raise ValueError("Log transformation requires all positive values")
            ref_clean = np.log(ref_clean)
            test_clean = np.log(test_clean)
        
        # Calculate means and differences
        means = (ref_clean + test_clean) / 2
        
        if self.params['percentage_difference']:
            differences = 200 * (test_clean - ref_clean) / (ref_clean + test_clean)
        else:
            differences = test_clean - ref_clean
        
        # Initialize results
        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'transformations': {
                'log_transform': self.params['log_transform'],
                'percentage_difference': self.params['percentage_difference']
            },
            'bias_analysis': self._analyze_bias(differences),
            'limits_of_agreement': self._calculate_limits_of_agreement(differences),
            'proportional_bias': self._test_proportional_bias(means, differences),
            'plot_data': {
                'means': means,
                'differences': differences,
                'ref_data': ref_clean,
                'test_data': test_clean
            }
        }
        
        # Outlier detection
        if self.params['outlier_detection']:
            results['outliers'] = self._detect_outliers(means, differences, results['limits_of_agreement'])
        
        # Agreement interpretation
        results['interpretation'] = self._interpret_agreement(results)
        
        # Store results
        self.results = results
        return results
    
    def _analyze_bias(self, differences: np.ndarray) -> Dict[str, float]:
        """Analyze systematic bias in the differences."""
        bias = np.mean(differences)
        bias_std = np.std(differences, ddof=1)
        bias_se = bias_std / np.sqrt(len(differences))
        
        # Test if bias is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(differences, 0)
        
        # Confidence interval for bias
        confidence_level = self.params['confidence_level']
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, len(differences) - 1)
        bias_ci_lower = bias - t_critical * bias_se
        bias_ci_upper = bias + t_critical * bias_se
        
        return {
            'bias': bias,
            'bias_std': bias_std,
            'bias_se': bias_se,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < (1 - confidence_level),
            'confidence_interval': {
                'lower': bias_ci_lower,
                'upper': bias_ci_upper,
                'level': confidence_level
            }
        }
    
    def _calculate_limits_of_agreement(self, differences: np.ndarray) -> Dict[str, float]:
        """Calculate limits of agreement."""
        bias = np.mean(differences)
        diff_std = np.std(differences, ddof=1)
        
        # Calculate limits
        confidence_level = self.params['confidence_level']
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_limit = bias - z_score * diff_std
        upper_limit = bias + z_score * diff_std
        
        # Confidence intervals for limits
        n = len(differences)
        se_limits = diff_std * np.sqrt(3/n)
        t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        
        lower_limit_ci = {
            'lower': lower_limit - t_critical * se_limits,
            'upper': lower_limit + t_critical * se_limits
        }
        
        upper_limit_ci = {
            'lower': upper_limit - t_critical * se_limits,
            'upper': upper_limit + t_critical * se_limits
        }
        
        return {
            'lower_limit': lower_limit,
            'upper_limit': upper_limit,
            'width': upper_limit - lower_limit,
            'confidence_level': confidence_level,
            'lower_limit_ci': lower_limit_ci,
            'upper_limit_ci': upper_limit_ci
        }
    
    def _test_proportional_bias(self, means: np.ndarray, differences: np.ndarray) -> Dict[str, Any]:
        """Test for proportional bias using regression analysis."""
        try:
            # Linear regression of differences vs means
            slope, intercept, r_value, p_value, std_err = stats.linregress(means, differences)
            
            # Test if slope is significantly different from zero
            significant = p_value < (1 - self.params['confidence_level'])
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err,
                'significant': significant,
                'interpretation': 'Proportional bias detected' if significant else 'No proportional bias'
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'interpretation': 'Could not test for proportional bias'
            }
    
    def _detect_outliers(self, means: np.ndarray, differences: np.ndarray, 
                        limits: Dict[str, float]) -> Dict[str, Any]:
        """Detect outliers beyond limits of agreement."""
        lower_limit = limits['lower_limit']
        upper_limit = limits['upper_limit']
        
        # Find outliers
        outlier_mask = (differences < lower_limit) | (differences > upper_limit)
        outlier_indices = np.where(outlier_mask)[0]
        
        outlier_info = []
        for idx in outlier_indices:
            outlier_info.append({
                'index': int(idx),
                'mean': means[idx],
                'difference': differences[idx],
                'type': 'below_lower' if differences[idx] < lower_limit else 'above_upper'
            })
        
        return {
            'n_outliers': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(differences)) * 100,
            'outlier_indices': outlier_indices.tolist(),
            'outlier_details': outlier_info
        }
    
    def _interpret_agreement(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Provide interpretation of agreement results."""
        interpretations = {}
        
        # Bias interpretation
        bias_analysis = results['bias_analysis']
        if bias_analysis['significant']:
            interpretations['bias'] = f"Significant systematic bias detected (p = {bias_analysis['p_value']:.4f})"
        else:
            interpretations['bias'] = "No significant systematic bias"
        
        # Proportional bias interpretation
        prop_bias = results['proportional_bias']
        if 'error' not in prop_bias:
            interpretations['proportional_bias'] = prop_bias['interpretation']
        else:
            interpretations['proportional_bias'] = "Proportional bias analysis failed"
        
        # Limits of agreement interpretation
        limits = results['limits_of_agreement']
        width = limits['width']
        interpretations['limits'] = f"95% limits of agreement: {width:.3f} units wide"
        
        # Outlier interpretation
        if 'outliers' in results:
            outliers = results['outliers']
            outlier_pct = outliers['outlier_percentage']
            if outlier_pct > 5:
                interpretations['outliers'] = f"High outlier rate: {outlier_pct:.1f}% beyond limits"
            elif outlier_pct > 0:
                interpretations['outliers'] = f"Some outliers detected: {outlier_pct:.1f}% beyond limits"
            else:
                interpretations['outliers'] = "No outliers detected"
        
        # Overall agreement assessment
        if bias_analysis['significant'] or (prop_bias.get('significant', False)):
            interpretations['overall'] = "Poor agreement - systematic bias detected"
        elif 'outliers' in results and results['outliers']['outlier_percentage'] > 10:
            interpretations['overall'] = "Moderate agreement - many outliers present"
        else:
            interpretations['overall'] = "Good agreement between methods"
        
        return interpretations 