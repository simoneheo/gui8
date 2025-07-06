"""
Residual Comparison Method

This module implements residual analysis for comparing two datasets,
focusing on the patterns and statistics of residuals (differences).
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class ResidualComparison(BaseComparison):
    """
    Residual analysis comparison method.
    
    Analyzes residuals (differences) between two datasets,
    focusing on the patterns and statistics of residuals.
    """
    
    name = "Residual Analysis"
    description = "Analyze residuals between reference and test data with statistical tests"
    category = "Statistical"
    version = "1.0.0"
    
    # Helpful information for the console
    helpful_info = """Residual Analysis - Analyzes prediction errors between datasets
• Residuals = Reference - Test (or Test - Reference)
• Checks for patterns: heteroscedasticity, non-linearity
• Tests normality of residuals (Shapiro-Wilk, Kolmogorov-Smirnov)
• Identifies outliers and influential points
• Good model: Residuals randomly distributed around zero
• Use for: Validating model assumptions, quality control"""
    
    parameters = {
        'fit_method': {
            'type': str,
            'default': 'linear',
            'choices': ['linear', 'polynomial', 'lowess'],
            'description': 'Fitting Method',
            'tooltip': 'Method for fitting the relationship between variables'
        },
        'polynomial_degree': {
            'type': int,
            'default': 2,
            'min': 1,
            'max': 5,
            'description': 'Polynomial Degree',
            'tooltip': 'Degree of polynomial fit (only used when fit_method is polynomial)'
        },
        'show_residual_stats': {
            'type': bool,
            'default': True,
            'description': 'Show Residual Statistics',
            'tooltip': 'Display statistics about the residuals'
        },
        'detect_outliers': {
            'type': bool,
            'default': True,
            'description': 'Detect Outliers',
            'tooltip': 'Identify outliers in the residuals'
        }
    }
    
    output_types = ["residual_statistics", "trend_analysis", "normality_tests", "outlier_analysis", "plot_data"]
    plot_type = "residual"
    
    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform residual analysis between reference and test data.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing residual analysis results
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Calculate residuals based on type
        residuals = self._calculate_residuals(ref_clean, test_clean)
        
        # Create time array if not provided
        if ref_time is not None and len(ref_time) == len(ref_clean):
            time_array = ref_time
        else:
            time_array = np.arange(len(ref_clean))
        
        # Initialize results
        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'residual_type': self.params['fit_method'],
            'residual_statistics': self._calculate_residual_statistics(residuals),
            'plot_data': {
                'residuals': residuals,
                'ref_data': ref_clean,
                'test_data': test_clean,
                'time': time_array,
                'fitted_values': (ref_clean + test_clean) / 2  # Average as fitted values
            }
        }
        
        # Trend analysis
        if self.params['fit_method'] != 'linear':
            results['trend_analysis'] = self._analyze_trends(residuals, time_array)
        
        # Normality tests
        results['normality_tests'] = self._test_normality(residuals)
        
        # Outlier analysis
        results['outlier_analysis'] = self._analyze_outliers(residuals)
        
        # Interpretation
        results['interpretation'] = self._interpret_residuals(results)
        
        # Store results
        self.results = results
        return results
    
    def _calculate_residuals(self, ref_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
        """Calculate residuals based on the specified type."""
        fit_method = self.params['fit_method']
        
        if fit_method == 'linear':
            return test_data - ref_data
        
        elif fit_method == 'polynomial':
            # Fit a polynomial to the data
            coefficients = np.polyfit(ref_data, test_data, self.params['polynomial_degree'])
            polynomial = np.poly1d(coefficients)
            return test_data - polynomial(ref_data)
        
        elif fit_method == 'lowess':
            # Fit a LOWESS (Locally Weighted Scatterplot Smoothing) regression
            lowess = stats.lowess(test_data, ref_data, frac=0.3)
            return test_data - lowess[:, 1]
        
        else:
            return test_data - ref_data
    
    def _calculate_residual_statistics(self, residuals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics for residuals."""
        return {
            'mean': np.mean(residuals),
            'median': np.median(residuals),
            'std': np.std(residuals, ddof=1),
            'var': np.var(residuals, ddof=1),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'range': np.ptp(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75),
            'iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'mad': np.median(np.abs(residuals - np.median(residuals))),  # Median absolute deviation
            'rmse': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals))
        }
    
    def _analyze_trends(self, residuals: np.ndarray, time_array: np.ndarray) -> Dict[str, Any]:
        """Analyze trends in residuals over time."""
        try:
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_array, residuals)
            
            # Runs test for randomness
            runs_test = self._runs_test(residuals)
            
            # Moving average trend
            window_size = max(5, len(residuals) // 10)
            moving_avg = self._moving_average(residuals, window_size)
            
            return {
                'linear_trend': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'significant': p_value < 0.05
                },
                'runs_test': runs_test,
                'moving_average': {
                    'values': moving_avg,
                    'window_size': window_size
                }
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residuals for normality using multiple tests."""
        tests = {}
        
        try:
            # Shapiro-Wilk test (for n < 5000)
            if len(residuals) < 5000:
                stat, p_value = stats.shapiro(residuals)
                tests['shapiro_wilk'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'normal': p_value > 0.05
                }
            
            # Kolmogorov-Smirnov test
            stat, p_value = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
            tests['kolmogorov_smirnov'] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
            
            # Anderson-Darling test
            result = stats.anderson(residuals, dist='norm')
            tests['anderson_darling'] = {
                'statistic': result.statistic,
                'critical_values': result.critical_values.tolist(),
                'significance_levels': result.significance_level.tolist(),
                'normal': result.statistic < result.critical_values[2]  # 5% level
            }
            
            # Jarque-Bera test
            stat, p_value = stats.jarque_bera(residuals)
            tests['jarque_bera'] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
            
        except Exception as e:
            tests['error'] = str(e)
        
        return tests
    
    def _analyze_outliers(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze outliers in residuals."""
        if not self.params['detect_outliers']:
            return {'outliers': 'Outliers detection disabled'}
        
        # Z-score method
        z_scores = np.abs(stats.zscore(residuals))
        z_outliers = np.where(z_scores > 3.0)[0]
        
        # IQR method
        q25, q75 = np.percentile(residuals, [25, 75])
        iqr = q75 - q25
        iqr_lower = q25 - 1.5 * iqr
        iqr_upper = q75 + 1.5 * iqr
        iqr_outliers = np.where((residuals < iqr_lower) | (residuals > iqr_upper))[0]
        
        # Modified Z-score method (using MAD)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        modified_z_scores = 0.6745 * (residuals - np.median(residuals)) / mad if mad > 0 else np.zeros_like(residuals)
        mad_outliers = np.where(np.abs(modified_z_scores) > 3.0)[0]
        
        return {
            'z_score_outliers': {
                'indices': z_outliers.tolist(),
                'count': len(z_outliers),
                'percentage': (len(z_outliers) / len(residuals)) * 100,
                'threshold': 3.0
            },
            'iqr_outliers': {
                'indices': iqr_outliers.tolist(),
                'count': len(iqr_outliers),
                'percentage': (len(iqr_outliers) / len(residuals)) * 100,
                'lower_bound': iqr_lower,
                'upper_bound': iqr_upper
            },
            'mad_outliers': {
                'indices': mad_outliers.tolist(),
                'count': len(mad_outliers),
                'percentage': (len(mad_outliers) / len(residuals)) * 100,
                'threshold': 3.0
            }
        }
    
    def _runs_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform runs test for randomness."""
        try:
            median_val = np.median(residuals)
            runs, n1, n2 = 0, 0, 0
            
            # Convert to binary sequence
            binary_seq = residuals > median_val
            
            # Count runs
            for i in range(len(binary_seq)):
                if binary_seq[i]:
                    n1 += 1
                else:
                    n2 += 1
                
                if i == 0 or binary_seq[i] != binary_seq[i-1]:
                    runs += 1
            
            # Expected runs and standard deviation
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            runs_std = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                              ((n1 + n2)**2 * (n1 + n2 - 1)))
            
            # Z-score
            z_score = (runs - expected_runs) / runs_std if runs_std > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            return {
                'runs': runs,
                'expected_runs': expected_runs,
                'z_score': z_score,
                'p_value': p_value,
                'random': p_value > 0.05
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def _interpret_residuals(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Provide interpretation of residual analysis results."""
        interpretations = {}
        
        # Statistics interpretation
        stats_data = results['residual_statistics']
        if abs(stats_data['mean']) > 2 * stats_data['std'] / np.sqrt(results['n_samples']):
            interpretations['bias'] = "Significant systematic bias detected"
        else:
            interpretations['bias'] = "No significant systematic bias"
        
        # Normality interpretation
        if 'normality_tests' in results:
            normal_tests = results['normality_tests']
            if 'error' not in normal_tests:
                normal_count = sum([1 for test in normal_tests.values() 
                                  if isinstance(test, dict) and test.get('normal', False)])
                total_tests = len([test for test in normal_tests.values() 
                                 if isinstance(test, dict)])
                if normal_count >= total_tests // 2:
                    interpretations['normality'] = "Residuals appear normally distributed"
                else:
                    interpretations['normality'] = "Residuals deviate from normality"
        
        # Trend interpretation
        if 'trend_analysis' in results:
            trend = results['trend_analysis']
            if 'error' not in trend and trend['linear_trend']['significant']:
                interpretations['trend'] = "Significant trend detected in residuals"
            else:
                interpretations['trend'] = "No significant trend in residuals"
        
        # Outlier interpretation
        outliers = results['outlier_analysis']
        z_outlier_pct = outliers['z_score_outliers']['percentage']
        if z_outlier_pct > 5:
            interpretations['outliers'] = f"High outlier rate: {z_outlier_pct:.1f}%"
        elif z_outlier_pct > 0:
            interpretations['outliers'] = f"Some outliers detected: {z_outlier_pct:.1f}%"
        else:
            interpretations['outliers'] = "No significant outliers"
        
        return interpretations
    
    def generate_plot_content(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any] = None, 
                             checked_pairs: list = None) -> None:
        """Generate residual plot content - residuals vs reference values"""
        try:
            ref_data = np.array(ref_data)
            test_data = np.array(test_data)
            
            if len(ref_data) == 0:
                ax.text(0.5, 0.5, 'No valid data for residual analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Calculate residuals
            residuals = test_data - ref_data
            
            # Get analysis parameters from config
            fit_method = plot_config.get('fit_method', 'linear') if plot_config else 'linear'
            
            # Add zero line for reference
            ax.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1, label='Zero residual')
            
            # Calculate statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals, ddof=1)
            
            # Add mean line
            ax.axhline(mean_residual, color='blue', linestyle='--', linewidth=2, 
                      label=f'Mean residual ({mean_residual:.3f})')
            
            # Add ±1σ and ±2σ lines
            ax.axhline(mean_residual + std_residual, color='orange', linestyle=':', alpha=0.7,
                      label=f'±1σ ({std_residual:.3f})')
            ax.axhline(mean_residual - std_residual, color='orange', linestyle=':', alpha=0.7)
            ax.axhline(mean_residual + 2*std_residual, color='red', linestyle=':', alpha=0.7,
                      label=f'±2σ ({2*std_residual:.3f})')
            ax.axhline(mean_residual - 2*std_residual, color='red', linestyle=':', alpha=0.7)
            
            # Detect and highlight outliers
            if fit_method == 'linear':
                outlier_mask = np.zeros_like(residuals, dtype=bool)
            else:
                outlier_mask = np.zeros_like(residuals, dtype=bool)
            
            # Plot normal points
            normal_mask = ~outlier_mask
            if np.any(normal_mask):
                ax.scatter(ref_data[normal_mask], residuals[normal_mask], alpha=0.6, s=20, 
                          color='blue', label='Normal points')
            
            # Plot outliers
            if np.any(outlier_mask):
                ax.scatter(ref_data[outlier_mask], residuals[outlier_mask], alpha=0.8, s=30, 
                          color='red', marker='x', label=f'Outliers ({np.sum(outlier_mask)})')
            
            # Add trend line if there's a significant trend
            try:
                if fit_method != 'linear':
                    slope, intercept, r_value, p_value, _ = stats.linregress(ref_data, residuals)
                    if p_value < 0.05:  # Significant trend
                        x_line = np.array([np.min(ref_data), np.max(ref_data)])
                        y_line = slope * x_line + intercept
                        ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.7,
                               label=f'Trend (p={p_value:.3f})')
            except:
                pass
            
            # Test normality and add to statistics
            stats_text = f'Mean = {mean_residual:.4f}\n'
            stats_text += f'SD = {std_residual:.4f}\n'
            
            try:
                if fit_method == 'linear':
                    stat, p_val = stats.shapiro(residuals)
                    stats_text += f'Shapiro p = {p_val:.3f}\n'
                    normal_status = "Normal" if p_val > 0.05 else "Non-normal"
                    stats_text += f'Distribution: {normal_status}\n'
                elif fit_method == 'polynomial':
                    # Implement polynomial normality test
                    pass
                elif fit_method == 'lowess':
                    # Implement LOWESS normality test
                    pass
            except:
                pass
            
            stats_text += f'n = {len(ref_data):,} points\n'
            stats_text += f'Outliers = {np.sum(outlier_mask)} ({np.sum(outlier_mask)/len(residuals)*100:.1f}%)'
            
            # Add statistics text box
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)
            
            ax.set_xlabel('Reference Values')
            ax.set_ylabel('Residuals (Test - Reference)')
            ax.set_title('Residual Analysis')
            ax.legend(loc='upper right')
            
        except Exception as e:
            print(f"[Residual] Error in plot generation: {e}")
            ax.text(0.5, 0.5, f'Error generating residual plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes) 