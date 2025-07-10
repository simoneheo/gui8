"""
Residual Comparison Method

This module implements residual analysis for comparing two datasets,
including residual plots, pattern detection, and statistical analysis of residuals.
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class ResidualComparison(BaseComparison):
    """
    Residual analysis comparison method.
    
    Performs residual analysis by fitting a model to the data and analyzing
    the residuals for patterns, outliers, and statistical properties.
    """
    
    name = "residual"
    description = "Residual analysis with pattern detection and statistical testing"
    category = "Error Analysis"
    version = "1.0.0"
    tags = ["scatter", "residual", "error", "pattern", "regression", "outlier"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "fit_method", "type": "str", "default": "linear", "help": "Fitting method: linear, polynomial, robust"},
        {"name": "polynomial_degree", "type": "int", "default": 2, "help": "Polynomial degree for polynomial fitting"},
        {"name": "detect_outliers", "type": "bool", "default": True, "help": "Detect outliers in residuals"},
        {"name": "outlier_threshold", "type": "float", "default": 2.5, "help": "Z-score threshold for outlier detection"},
        {"name": "test_autocorrelation", "type": "bool", "default": True, "help": "Test for autocorrelation in residuals"},
        {"name": "test_normality", "type": "bool", "default": True, "help": "Test normality of residuals"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "help": "Confidence level for statistical tests"},
        {"name": "bootstrap_samples", "type": "int", "default": 1000, "help": "Number of bootstrap samples for confidence intervals"}
    ]
    
    # Plot configuration  
    plot_type = "residual"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'highlight_outliers': {'default': True, 'label': 'Highlight Outliers', 'tooltip': 'Highlight outlier points in residuals'},
        'show_trend_line': {'default': False, 'label': 'Show Trend Line', 'tooltip': 'Show trend line in residuals'},
        'show_confidence_bands': {'default': False, 'label': 'Show Confidence Bands', 'tooltip': 'Show ±2σ confidence bands for residuals'},
        'show_statistical_results': {'default': True, 'label': 'Show Statistical Results', 'tooltip': 'Display residual statistics on the plot'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the residual analysis.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing residual analysis results with statistics and plot data
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Calculate statistics
        stats_results = self.calculate_stats(ref_clean, test_clean, ref_time, test_time)
        
        # Calculate residuals
        residuals = self._calculate_residuals(ref_clean, test_clean)
        
        # Prepare plot data
        plot_data = {
            'ref_data': ref_clean,
            'test_data': test_clean,
            'residuals': residuals,
            'valid_ratio': valid_ratio
        }
        
        # Combine results
        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'statistics': stats_results,
            'plot_data': plot_data
        }
        
        # Store results
        self.results = results
        return results
    
    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate residual analysis statistics.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing residual analysis statistics
        """
        # Fit model and calculate residuals
        model_results = self._fit_model(ref_data, test_data)
        residuals = model_results['residuals']
        
        # Initialize results
        stats_results = {
            'model_fit': model_results,
            'residual_stats': {},
            'pattern_tests': {},
            'outlier_analysis': {}
        }
        
        # Basic residual statistics
        stats_results['residual_stats'] = self._calculate_residual_stats(residuals)
        
        # Test for patterns in residuals
        stats_results['pattern_tests'] = self._test_residual_patterns(residuals, ref_data)
        
        # Outlier analysis
        if self.kwargs.get('detect_outliers', True):
            stats_results['outlier_analysis'] = self._analyze_outliers(residuals, ref_data, test_data)
        
        return stats_results
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate residual plot with performance and overlay options.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        # Calculate residuals
        residuals = self._calculate_residuals(ref_data, test_data)
        
        # Apply performance optimizations
        ref_plot, residuals_plot = self._apply_performance_optimizations(ref_data, residuals, plot_config)
        
        # Create density plot based on configuration
        self._create_density_plot(ax, ref_plot, residuals_plot, plot_config)
        
        # Add residual-specific overlay elements
        self._add_residual_overlays(ax, ref_data, residuals, plot_config, stats_results)
        
        # Add general overlay elements
        self._add_overlay_elements(ax, ref_data, test_data, plot_config, stats_results)
        
        # Set labels and title
        ax.set_xlabel('Reference Data')
        ax.set_ylabel('Residuals (Test - Predicted)')
        ax.set_title('Residual Analysis')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if requested
        if plot_config.get('show_legend', False):
            ax.legend()
    
    def _calculate_residuals(self, ref_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
        """Calculate residuals based on fitted model."""
        model_results = self._fit_model(ref_data, test_data)
        return model_results['residuals']
    
    def _fit_model(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Fit model to data and calculate residuals."""
        fit_method = self.kwargs.get('fit_method', 'linear')
        
        try:
            print(f"[ResidualAnalysis] Attempting to fit {fit_method} model with {len(ref_data)} data points")
            
            if fit_method == 'linear':
                result = self._fit_linear_model(ref_data, test_data)
            elif fit_method == 'polynomial':
                result = self._fit_polynomial_model(ref_data, test_data)
            elif fit_method == 'robust':
                result = self._fit_robust_model(ref_data, test_data)
            else:
                # Default to linear
                result = self._fit_linear_model(ref_data, test_data)
            
            print(f"[ResidualAnalysis] Successfully fitted {result.get('model_type', 'unknown')} model with R²={result.get('r_squared', 'N/A'):.3f}")
            return result
            
        except Exception as e:
            print(f"[ResidualAnalysis] Model fitting failed: {e}")
            print(f"[ResidualAnalysis] Falling back to simple linear regression without sklearn")
            
            # Try a simple numpy-based linear regression as fallback
            try:
                return self._fit_simple_linear_model(ref_data, test_data)
            except Exception as e2:
                print(f"[ResidualAnalysis] Simple linear regression also failed: {e2}")
                print(f"[ResidualAnalysis] Using difference calculation as last resort")
                
                # Ultimate fallback to simple difference
                residuals = test_data - ref_data
                return {
                    'residuals': residuals,
                    'predicted': ref_data,
                    'r_squared': np.nan,
                    'model_type': 'difference',
                    'error': f"Model fitting failed: {str(e)}, Simple regression failed: {str(e2)}"
                }
    
    def _fit_linear_model(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Fit linear model using sklearn."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
        except ImportError as e:
            raise ImportError(f"sklearn not available for linear regression: {e}")
        
        # Validate input data
        if len(ref_data) < 2:
            raise ValueError("Need at least 2 data points for linear regression")
        
        if np.var(ref_data) == 0:
            raise ValueError("Reference data has zero variance - cannot fit linear model")
        
        X = ref_data.reshape(-1, 1)
        y = test_data
        
        model = LinearRegression()
        model.fit(X, y)
        
        predicted = model.predict(X)
        residuals = y - predicted
        r_squared = r2_score(y, predicted)
        
        return {
            'residuals': residuals,
            'predicted': predicted,
            'r_squared': r_squared,
            'model_type': 'linear',
            'slope': model.coef_[0],
            'intercept': model.intercept_
        }
    
    def _fit_simple_linear_model(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Fit linear model using numpy without sklearn."""
        try:
            # Use numpy polyfit for simple linear regression
            coeffs = np.polyfit(ref_data, test_data, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Calculate predicted values
            predicted = slope * ref_data + intercept
            
            # Calculate residuals
            residuals = test_data - predicted
            
            # Calculate R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
            
            if ss_tot == 0:
                r_squared = 1.0 if ss_res == 0 else 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            print(f"[ResidualAnalysis] Simple linear model fitted: slope={slope:.3f}, intercept={intercept:.3f}, R²={r_squared:.3f}")
            
            return {
                'residuals': residuals,
                'predicted': predicted,
                'r_squared': r_squared,
                'model_type': 'simple_linear',
                'slope': slope,
                'intercept': intercept
            }
        except Exception as e:
            raise RuntimeError(f"Simple linear regression failed: {e}")
    
    def _fit_polynomial_model(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Fit polynomial model."""
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
        except ImportError as e:
            raise ImportError(f"sklearn not available for polynomial regression: {e}")
        
        degree = self.kwargs.get('polynomial_degree', 2)
        
        # Validate input data
        if len(ref_data) < degree + 1:
            raise ValueError(f"Need at least {degree + 1} data points for polynomial degree {degree}")
        
        if np.var(ref_data) == 0:
            raise ValueError("Reference data has zero variance - cannot fit polynomial model")
        
        X = ref_data.reshape(-1, 1)
        y = test_data
        
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        predicted = model.predict(X_poly)
        residuals = y - predicted
        r_squared = r2_score(y, predicted)
        
        return {
            'residuals': residuals,
            'predicted': predicted,
            'r_squared': r_squared,
            'model_type': f'polynomial_degree_{degree}',
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
    
    def _fit_robust_model(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Fit robust model using RANSAC or Huber regression."""
        try:
            from sklearn.linear_model import HuberRegressor
            
            X = ref_data.reshape(-1, 1)
            y = test_data
            
            model = HuberRegressor()
            model.fit(X, y)
            
            predicted = model.predict(X)
            residuals = y - predicted
            r_squared = r2_score(y, predicted)
            
            return {
                'residuals': residuals,
                'predicted': predicted,
                'r_squared': r_squared,
                'model_type': 'huber_robust',
                'coef': model.coef_[0],
                'intercept': model.intercept_
            }
        except ImportError:
            # Fallback to linear if sklearn not available
            return self._fit_linear_model(ref_data, test_data)
    
    def _calculate_residual_stats(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Calculate basic residual statistics."""
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals, ddof=1),
            'median': np.median(residuals),
            'mad': np.median(np.abs(residuals - np.median(residuals))),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'rmse': np.sqrt(np.mean(residuals**2))
        }
    
    def _test_residual_patterns(self, residuals: np.ndarray, ref_data: np.ndarray) -> Dict[str, Any]:
        """Test for patterns in residuals."""
        pattern_tests = {}
        
        # Test for autocorrelation
        if self.kwargs.get('test_autocorrelation', True):
            pattern_tests['autocorrelation'] = self._test_autocorrelation(residuals)
        
        # Test for normality
        if self.kwargs.get('test_normality', True):
            pattern_tests['normality'] = self._test_residual_normality(residuals)
        
        # Test for heteroscedasticity
        pattern_tests['heteroscedasticity'] = self._test_heteroscedasticity(residuals, ref_data)
        
        # Test for trend
        pattern_tests['trend'] = self._test_residual_trend(residuals, ref_data)
        
        return pattern_tests
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for autocorrelation in residuals using Durbin-Watson test."""
        try:
            # Simple autocorrelation test
            n = len(residuals)
            if n < 3:
                return {'error': 'Insufficient data for autocorrelation test'}
            
            # Calculate Durbin-Watson statistic
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
            
            # Rough interpretation (exact critical values depend on sample size)
            if dw_stat < 1.5:
                interpretation = "Positive autocorrelation detected"
            elif dw_stat > 2.5:
                interpretation = "Negative autocorrelation detected"
            else:
                interpretation = "No significant autocorrelation"
            
            return {
                'durbin_watson': dw_stat,
                'interpretation': interpretation
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _test_residual_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test normality of residuals."""
        try:
            stat, p_value = stats.shapiro(residuals)
            alpha = 0.05
            normal = p_value > alpha
            
            return {
                'shapiro_stat': stat,
                'p_value': p_value,
                'normal': normal,
                'interpretation': f"Residuals are {'normally' if normal else 'not normally'} distributed (p={p_value:.4f})"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _test_heteroscedasticity(self, residuals: np.ndarray, ref_data: np.ndarray) -> Dict[str, Any]:
        """Test for heteroscedasticity (non-constant variance)."""
        try:
            # Breusch-Pagan test approximation
            abs_residuals = np.abs(residuals)
            correlation, p_value = stats.pearsonr(ref_data, abs_residuals)
            
            alpha = 0.05
            heteroscedastic = p_value < alpha
            
            return {
                'correlation': correlation,
                'p_value': p_value,
                'heteroscedastic': heteroscedastic,
                'interpretation': f"{'Heteroscedasticity' if heteroscedastic else 'Homoscedasticity'} detected (p={p_value:.4f})"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _test_residual_trend(self, residuals: np.ndarray, ref_data: np.ndarray) -> Dict[str, Any]:
        """Test for trend in residuals."""
        try:
            # Test correlation between residuals and reference data
            correlation, p_value = stats.pearsonr(ref_data, residuals)
            
            alpha = 0.05
            trend_present = p_value < alpha
            
            return {
                'correlation': correlation,
                'p_value': p_value,
                'trend_present': trend_present,
                'interpretation': f"{'Trend' if trend_present else 'No trend'} in residuals (p={p_value:.4f})"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_outliers(self, residuals: np.ndarray, ref_data: np.ndarray, 
                         test_data: np.ndarray) -> Dict[str, Any]:
        """Analyze outliers in residuals."""
        try:
            threshold = self.kwargs.get('outlier_threshold', 2.5)
            
            # Z-score based outlier detection
            z_scores = np.abs(stats.zscore(residuals))
            outlier_mask = z_scores > threshold
            
            outlier_indices = np.where(outlier_mask)[0]
            n_outliers = len(outlier_indices)
            outlier_percentage = (n_outliers / len(residuals)) * 100
            
            return {
                'n_outliers': n_outliers,
                'outlier_percentage': outlier_percentage,
                'outlier_indices': outlier_indices.tolist(),
                'outlier_threshold': threshold,
                'outlier_residuals': residuals[outlier_mask].tolist(),
                'outlier_ref_values': ref_data[outlier_mask].tolist(),
                'outlier_test_values': test_data[outlier_mask].tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _add_residual_overlays(self, ax, ref_data: np.ndarray, residuals: np.ndarray, 
                              plot_config: Dict[str, Any] = None, 
                              stats_results: Dict[str, Any] = None) -> None:
        """Add residual-specific overlay elements."""
        if plot_config is None:
            plot_config = {}
        
        # Add debug logging
        print(f"[ResidualOverlay] Plot config keys: {list(plot_config.keys())}")
        
        # Check for parameter compatibility (wizard may send different names)
        show_outliers = plot_config.get('highlight_outliers', plot_config.get('outlier_detection', True))
        show_conf_bands = plot_config.get('show_confidence_bands', plot_config.get('confidence_bands', False))
        show_trend = plot_config.get('show_trend_line', plot_config.get('trend_line', False))
        
        print(f"[ResidualOverlay] Overlays - outliers: {show_outliers}, conf_bands: {show_conf_bands}, trend: {show_trend}")
        
        # Add zero line (always shown for residuals)
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Zero Line')
        
        # Add confidence bands for residuals
        if show_conf_bands and stats_results:
            residual_stats = stats_results.get('residual_stats', {})
            std_res = residual_stats.get('std', 0)
            if std_res > 0:
                x_min, x_max = np.min(ref_data), np.max(ref_data)
                print(f"[ResidualOverlay] Drawing confidence bands: ±{2*std_res:.3f}")
                ax.fill_between([x_min, x_max], -2*std_res, 2*std_res, 
                               alpha=0.3, color='yellow', label='±2σ Residual Bands')
        
        # Highlight outliers
        if show_outliers and stats_results:
            outlier_analysis = stats_results.get('outlier_analysis', {})
            outlier_indices = outlier_analysis.get('outlier_indices', [])
            if outlier_indices:
                print(f"[ResidualOverlay] Highlighting {len(outlier_indices)} outliers")
                ax.scatter(ref_data[outlier_indices], residuals[outlier_indices], 
                          color='red', s=50, alpha=0.8, label='Outliers')
        
        # Add trend line if pattern detected
        if show_trend and stats_results:
            pattern_tests = stats_results.get('pattern_tests', {})
            trend_test = pattern_tests.get('trend', {})
            if trend_test.get('trend_present', False):
                print(f"[ResidualOverlay] Drawing trend line")
                # Fit line to residuals
                slope, intercept, _, _, _ = stats.linregress(ref_data, residuals)
                x_line = np.array([np.min(ref_data), np.max(ref_data)])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'g--', alpha=0.7, label=f'Trend Line (slope={slope:.3f})')
    
    def _add_overlay_elements(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                            plot_config: Dict[str, Any] = None, 
                            stats_results: Dict[str, Any] = None) -> None:
        """
        Override base method to prevent inappropriate overlays for residual plots.
        
        For residual analysis, we only want specific overlays that make sense:
        - NO identity line (residuals vs reference data)
        - NO regression line (residuals should be random around zero)
        - Custom statistical text formatting
        """
        if plot_config is None:
            plot_config = {}
        
        # Only add statistical results text - other overlays don't make sense for residuals
        if plot_config.get('show_statistical_results', True) and stats_results:
            self._add_statistical_text(ax, stats_results, plot_config)
    
    def _format_statistical_text(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format statistical results for display on plot."""
        text_lines = []
        
        # Add model fit information
        if 'model_fit' in stats_results:
            model_fit = stats_results['model_fit']
            r_squared = model_fit.get('r_squared', np.nan)
            if not np.isnan(r_squared):
                text_lines.append(f"R²: {r_squared:.3f}")
            
            model_type = model_fit.get('model_type', 'unknown')
            text_lines.append(f"Model: {model_type}")
        
        # Add residual statistics
        if 'residual_stats' in stats_results:
            res_stats = stats_results['residual_stats']
            rmse = res_stats.get('rmse', np.nan)
            if not np.isnan(rmse):
                text_lines.append(f"RMSE: {rmse:.3f}")
            
            mean_res = res_stats.get('mean', np.nan)
            if not np.isnan(mean_res):
                text_lines.append(f"Mean residual: {mean_res:.3f}")
        
        # Add pattern test results
        if 'pattern_tests' in stats_results:
            pattern_tests = stats_results['pattern_tests']
            
            # Normality test
            if 'normality' in pattern_tests:
                normal_test = pattern_tests['normality']
                if 'normal' in normal_test:
                    text_lines.append(f"Normality: {'Yes' if normal_test['normal'] else 'No'}")
            
            # Trend test
            if 'trend' in pattern_tests:
                trend_test = pattern_tests['trend']
                if 'trend_present' in trend_test:
                    text_lines.append(f"Trend: {'Yes' if trend_test['trend_present'] else 'No'}")
        
        # Add outlier information
        if 'outlier_analysis' in stats_results:
            outlier_analysis = stats_results['outlier_analysis']
            n_outliers = outlier_analysis.get('n_outliers', 0)
            outlier_pct = outlier_analysis.get('outlier_percentage', 0)
            text_lines.append(f"Outliers: {n_outliers} ({outlier_pct:.1f}%)")
        
        return text_lines
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Residual Analysis",
            "description": "Analyzes residuals to detect patterns, outliers, and model adequacy",
            "interpretation": {
                "residuals": "Differences between observed and predicted values",
                "zero_line": "Residuals should be randomly scattered around zero",
                "patterns": "Systematic patterns indicate model inadequacy",
                "outliers": "Points with unusually large residuals",
                "heteroscedasticity": "Non-constant variance across the range"
            },
            "use_cases": [
                "Model validation and adequacy assessment",
                "Detecting systematic errors in measurements",
                "Identifying outliers and influential points",
                "Checking assumptions for statistical models"
            ],
            "tips": [
                "Look for random scatter around zero line",
                "Check for funnel shapes (heteroscedasticity)",
                "Investigate outliers - they may indicate real problems",
                "Consider different model types if patterns are present",
                "Use robust fitting methods for data with outliers"
            ]
        } 