"""
Correlation Comparison Method

This module implements correlation analysis between two data channels,
including Pearson, Spearman, and Kendall correlation coefficients.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple
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
    
    # Helpful information for the console
    helpful_info = """Correlation Analysis - Measures relationships between datasets
• Pearson: Linear correlation (r), assumes normal distribution
  - Strong: |r| > 0.7, Moderate: 0.3 < |r| < 0.7, Weak: |r| < 0.3
• Spearman: Rank-based correlation (ρ), non-parametric
  - Good for monotonic relationships, robust to outliers
• Kendall: Tau correlation (τ), robust to outliers
  - Better for small samples, less sensitive to extreme values
• R² (coefficient of determination): Proportion of variance explained
• Use for: Assessing how well two variables move together"""
    
    parameters = {
        'correlation_type': {
            'type': str,
            'default': 'pearson',
            'choices': ['pearson', 'spearman', 'all'],
            'description': 'Correlation Type',
            'tooltip': 'Pearson: Linear relationships\nSpearman: Monotonic relationships\nAll: Compute both types'
        },
        'confidence_level': {
            'type': float,
            'default': 0.95,
            'min': 0.8,
            'max': 0.999,
            'description': 'Confidence Level',
            'tooltip': 'Confidence level for statistical tests and confidence intervals'
        },
        'include_rmse': {
            'type': bool,
            'default': True,
            'description': 'Include RMSE',
            'tooltip': 'Calculate Root Mean Square Error along with correlation'
        },
        'remove_outliers': {
            'type': bool,
            'default': False,
            'description': 'Remove Outliers',
            'tooltip': 'Automatically detect and remove outliers before analysis'
        },
        'outlier_method': {
            'type': str,
            'default': 'iqr',
            'choices': ['iqr', 'zscore', 'modified_zscore'],
            'description': 'Outlier Method',
            'tooltip': 'IQR: Interquartile range method (1.5×IQR)\nZ-score: Standard deviation method (>3σ)\nModified Z-score: Median-based robust method'
        },
        'partial_correlation': {
            'type': bool,
            'default': False,
            'description': 'Partial Correlation',
            'tooltip': 'Calculate partial correlation controlling for linear trends (removes time effects)'
        },
        'detrend_method': {
            'type': str,
            'default': 'none',
            'choices': ['none', 'linear', 'polynomial'],
            'description': 'Detrend Method',
            'tooltip': 'None: No detrending\nLinear: Remove linear trends\nPolynomial: Remove polynomial trends (degree 2)'
        }
    }
    
    output_types = ["correlation_statistics", "rmse_metrics", "plot_data"]
    plot_type = "pearson"
    
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
        
        # Remove outliers if requested
        if self.params['remove_outliers']:
            ref_clean, test_clean = self._remove_outliers(ref_clean, test_clean)
        
        # Initialize results
        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'correlations': {},
            'plot_data': {
                'ref_data': ref_clean,
                'test_data': test_clean
            }
        }
        
        correlation_type = self.params['correlation_type']
        
        # Compute correlations based on type
        if correlation_type in ['pearson', 'all']:
            results['correlations']['pearson'] = self._compute_pearson(ref_clean, test_clean)
        
        if correlation_type in ['spearman', 'all']:
            results['correlations']['spearman'] = self._compute_spearman(ref_clean, test_clean)
        
        # Calculate RMSE if requested
        if self.params['include_rmse']:
            results['rmse_metrics'] = self._compute_rmse_metrics(ref_clean, test_clean)
        
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
    
    def _compute_rmse_metrics(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute Root Mean Square Error (RMSE) metrics."""
        try:
            rmse = np.sqrt(np.mean((ref_data - test_data) ** 2))
            return {
                'rmse': rmse
            }
        except Exception as e:
            return {
                'rmse': np.nan,
                'error': str(e)
            }
    
    def _interpret_correlations(self, correlations: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Interpret correlation results with simplified categories."""
        interpretations = {}
        
        for method, result in correlations.items():
            if 'error' in result:
                interpretations[method] = f"Error: {result['error']}"
                continue
            
            coefficient = result.get('coefficient', np.nan)
            if np.isnan(coefficient):
                interpretations[method] = "Invalid correlation"
                continue
            
            abs_coeff = abs(coefficient)
            direction = "positive" if coefficient > 0 else "negative"
            
            if abs_coeff >= 0.8:
                strength = "very strong"
            elif abs_coeff >= 0.6:
                strength = "strong"
            elif abs_coeff >= 0.4:
                strength = "moderate"
            elif abs_coeff >= 0.2:
                strength = "weak"
            else:
                strength = "very weak"
            
            interpretations[method] = f"{strength.title()} {direction} correlation (r = {coefficient:.3f})"
        
        return interpretations
    
    def _remove_outliers(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using IQR method."""
        # Calculate combined z-scores
        ref_z = np.abs(stats.zscore(ref_data))
        test_z = np.abs(stats.zscore(test_data))
        
        # Remove points where either variable is an outlier (z > 3)
        outlier_mask = (ref_z < 3) & (test_z < 3)
        
        return ref_data[outlier_mask], test_data[outlier_mask]
    
    def generate_plot_content(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any] = None, 
                             checked_pairs: list = None) -> None:
        """Generate correlation plot content - scatter plot with correlation statistics"""
        try:
            ref_data = np.array(ref_data)
            test_data = np.array(test_data)
            
            if len(ref_data) == 0:
                ax.text(0.5, 0.5, 'No valid data for correlation analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Get correlation type from config
            correlation_type = plot_config.get('correlation_type', 'pearson') if plot_config else 'pearson'
            confidence_level = plot_config.get('confidence_level', 0.95) if plot_config else 0.95
            
            # Add 1:1 line
            min_val = min(np.min(ref_data), np.min(test_data))
            max_val = max(np.max(ref_data), np.max(test_data))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, 
                   linewidth=2, label='1:1 line')
            
            # Calculate correlation statistics
            stats_text = ""
            try:
                if correlation_type == 'pearson' or correlation_type == 'all':
                    r_value, p_value = stats.pearsonr(ref_data, test_data)
                    r2_value = r_value ** 2
                    stats_text += f'Pearson r = {r_value:.4f}\n'
                    stats_text += f'R² = {r2_value:.4f}\n'
                    if not np.isnan(p_value):
                        stats_text += f'p = {p_value:.2e}\n'
                    
                    # Add confidence interval if available
                    if not np.isnan(p_value) and len(ref_data) > 3:
                        n = len(ref_data)
                        # Fisher z-transformation for confidence interval
                        z_r = 0.5 * np.log((1 + r_value) / (1 - r_value))
                        se_z = 1 / np.sqrt(n - 3)
                        alpha = 1 - confidence_level
                        z_critical = stats.norm.ppf(1 - alpha/2)
                        
                        z_lower = z_r - z_critical * se_z
                        z_upper = z_r + z_critical * se_z
                        
                        # Transform back to correlation scale
                        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                        
                        stats_text += f'{confidence_level*100:.0f}% CI: [{r_lower:.3f}, {r_upper:.3f}]\n'
                    
                    # Add best fit line
                    try:
                        slope, intercept, _, _, _ = stats.linregress(ref_data, test_data)
                        if not np.isnan(slope):
                            x_line = np.array([min_val, max_val])
                            y_line = slope * x_line + intercept
                            ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.7,
                                   label=f'Best fit: y = {slope:.3f}x + {intercept:.3f}')
                    except:
                        pass
                
                # Add other correlation types if requested
                if correlation_type == 'all':
                    try:
                        spearman_r, spearman_p = stats.spearmanr(ref_data, test_data)
                        kendall_tau, kendall_p = stats.kendalltau(ref_data, test_data)
                        stats_text += f'Spearman ρ = {spearman_r:.4f}\n'
                        stats_text += f'Kendall τ = {kendall_tau:.4f}\n'
                    except:
                        pass
                elif correlation_type == 'spearman':
                    r_value, p_value = stats.spearmanr(ref_data, test_data)
                    stats_text += f'Spearman ρ = {r_value:.4f}\n'
                    if not np.isnan(p_value):
                        stats_text += f'p = {p_value:.2e}\n'
                elif correlation_type == 'kendall':
                    r_value, p_value = stats.kendalltau(ref_data, test_data)
                    stats_text += f'Kendall τ = {r_value:.4f}\n'
                    if not np.isnan(p_value):
                        stats_text += f'p = {p_value:.2e}\n'
                
                stats_text += f'n = {len(ref_data):,} points'
                
            except Exception as e:
                stats_text = f'Correlation calculation error: {str(e)}'
            
            # Add statistics text box
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)
            
            ax.set_xlabel('Reference')
            ax.set_ylabel('Test')
            ax.set_title(f'{correlation_type.title()} Correlation Analysis')
            ax.legend(loc='lower right')
            
        except Exception as e:
            print(f"[Correlation] Error in plot generation: {e}")
            ax.text(0.5, 0.5, f'Error generating correlation plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes) 