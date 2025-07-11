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
        {"name": "fit_method", "type": "str", "default": "linear", "options": ["linear", "polynomial", "robust"], "help": "Linear (straight line), Polynomial (curved line), or Robust (outlier-resistant)"},
        {"name": "detect_outliers", "type": "bool", "default": True, "help": "Detect and highlight outliers in residuals"},
        {"name": "outlier_threshold", "type": "float", "default": 2.5, "min_value": 0.5, "max_value": 10.0, "help": "Z-score threshold for outlier detection (higher = less sensitive)"}
    ]
    
    # Plot configuration  
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_zero_line': {'default': True, 'label': 'Show y=0 Line', 'tooltip': 'Show horizontal line at y=0 for residual reference', 'type': 'line'},
        'show_statistical_results': {'default': True, 'label': 'Show Statistical Results', 'tooltip': 'Display residual statistics on the plot', 'type': 'text'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the residual analysis.
        
        Streamlined 3-step workflow:
        1. Validate input data (basic validation + remove NaN/infinite values)
        2. plot_script (core transformation + residual computation)
        3. stats_script (statistical calculations)
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing residual analysis results
        """
        try:
            # === STEP 1: VALIDATE INPUT DATA ===
            # Basic validation (shape, type, length compatibility)
            ref_data, test_data = self._validate_input_data(ref_data, test_data)
            # Remove NaN and infinite values
            ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
            
            # === STEP 2: PLOT SCRIPT (core transformation + residual computation) ===
            x_data, y_data, plot_metadata = self.plot_script(ref_clean, test_clean, self.kwargs)
            
            # === STEP 3: STATS SCRIPT (statistical calculations) ===
            stats_results = self.stats_script(x_data, y_data, ref_clean, test_clean, self.kwargs)
            
            # Prepare plot data
            plot_data = {
                'fitted_values': x_data,
                'residuals': y_data,
                'ref_data': ref_clean,
                'test_data': test_clean,
                'valid_ratio': valid_ratio,
                'metadata': plot_metadata
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
            
        except Exception as e:
            raise RuntimeError(f"Residual analysis failed: {str(e)}")

    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for residual analysis
        
        This defines what gets plotted on X and Y axes for residual plots.
        
        Args:
            ref_data: Reference measurements (already cleaned of NaN/infinite values)
            test_data: Test measurements (already cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (x_data, y_data, metadata)
                x_data: Fitted values for X-axis
                y_data: Residuals for Y-axis
                metadata: Plot configuration dictionary
        """
        # Fit regression model
        fitted_values, residuals, model_info = self._fit_regression_model(ref_data, test_data, params)
        
        # Apply residual transformation if requested
        if params.get("standardize_residuals", False):
            residuals = self._standardize_residuals(residuals, model_info)
        
        # Prepare metadata for plotting
        metadata = {
            'x_label': 'Fitted Values',
            'y_label': 'Standardized Residuals' if params.get("standardize_residuals", False) else 'Residuals',
            'title': 'Residual Analysis',
            'plot_type': 'scatter',
            'model_type': params.get("model_type", "linear"),
            'standardized': params.get("standardize_residuals", False),
            'model_info': model_info
        }
        
        return fitted_values, residuals, metadata

    def _fit_regression_model(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Fit regression model and compute residuals."""
        model_type = params.get("model_type", "linear")
        
        if model_type == "linear":
            # Linear regression: test = a * ref + b
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(ref_data, test_data)
            
            fitted_values = slope * ref_data + intercept
            residuals = test_data - fitted_values
            
            model_info = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err,
                'model_type': 'linear'
            }
            
        elif model_type == "quadratic":
            # Quadratic regression: test = a * ref^2 + b * ref + c
            coeffs = np.polyfit(ref_data, test_data, 2)
            fitted_values = np.polyval(coeffs, ref_data)
            residuals = test_data - fitted_values
            
            model_info = {
                'coefficients': coeffs.tolist(),
                'model_type': 'quadratic'
            }
            
        elif model_type == "polynomial":
            # Polynomial regression with standard degree 2 (quadratic)
            degree = 2
            coeffs = np.polyfit(ref_data, test_data, degree)
            fitted_values = np.polyval(coeffs, ref_data)
            residuals = test_data - fitted_values
            
            model_info = {
                'coefficients': coeffs.tolist(),
                'degree': degree,
                'model_type': 'polynomial'
            }
            
        else:  # Default to linear
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(ref_data, test_data)
            
            fitted_values = slope * ref_data + intercept
            residuals = test_data - fitted_values
            
            model_info = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err,
                'model_type': 'linear'
            }
        
        return fitted_values, residuals, model_info

    def _standardize_residuals(self, residuals: np.ndarray, model_info: Dict) -> np.ndarray:
        """Standardize residuals by dividing by standard error."""
        residual_std = np.std(residuals)
        if residual_std > 0:
            return residuals / residual_std
        else:
            return residuals

    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        BACKWARD COMPATIBILITY + SAFETY WRAPPER: Calculate residual analysis statistics.
        
        This method maintains compatibility with existing code and provides comprehensive
        validation and error handling around the core statistical calculations.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing residual analysis statistics
        """
        # Get plot data using the script-based approach
        x_data, y_data, plot_metadata = self.plot_script(ref_data, test_data, self.kwargs)
        
        # === INPUT VALIDATION ===
        if len(x_data) != len(y_data):
            raise ValueError("X and Y data arrays must have the same length")
        
        if len(y_data) < 3:
            raise ValueError("Insufficient data for statistical analysis (minimum 3 samples required)")
        
        # === PURE CALCULATIONS (delegated to stats_script) ===
        stats_results = self.stats_script(x_data, y_data, ref_data, test_data, self.kwargs)
        
        return stats_results

    def stats_script(self, x_data: np.ndarray, y_data: np.ndarray, 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Statistical calculations for residual analysis
        
        Args:
            x_data: Fitted values
            y_data: Residuals
            ref_data: Original reference data
            test_data: Original test data
            params: Method parameters dictionary
            
        Returns:
            Dictionary containing statistical results
        """
        fitted_values = x_data
        residuals = y_data
        
        # Get model info from plot_script
        _, _, plot_metadata = self.plot_script(ref_data, test_data, params)
        model_info = plot_metadata['model_info']
        
        # Basic residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75),
            'iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25),
            'skewness': self._calculate_skewness(residuals),
            'kurtosis': self._calculate_kurtosis(residuals)
        }
        
        # Normality tests
        normality_tests = self._perform_normality_tests(residuals)
        
        # Heteroscedasticity tests
        heteroscedasticity_tests = self._perform_heteroscedasticity_tests(fitted_values, residuals)
        
        # Autocorrelation tests
        autocorrelation_tests = self._perform_autocorrelation_tests(residuals)
        
        # Outlier detection
        outlier_analysis = self._detect_outliers(residuals, params)
        
        # Model fit statistics
        model_fit_stats = self._calculate_model_fit_statistics(ref_data, test_data, fitted_values, residuals, model_info)
        
        stats_results = {
            'residual_stats': residual_stats,
            'normality_tests': normality_tests,
            'heteroscedasticity_tests': heteroscedasticity_tests,
            'autocorrelation_tests': autocorrelation_tests,
            'outlier_analysis': outlier_analysis,
            'model_fit_stats': model_fit_stats,
            'model_info': model_info
        }
        
        return stats_results

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            n = len(data)
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                return np.sum(((data - mean) / std) ** 3) / n
            else:
                return 0.0

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            n = len(data)
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                return np.sum(((data - mean) / std) ** 4) / n - 3
            else:
                return 0.0

    def _perform_normality_tests(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform normality tests on residuals."""
        try:
            from scipy.stats import shapiro, jarque_bera, normaltest
            
            # Shapiro-Wilk test (better for small samples)
            shapiro_stat, shapiro_p = shapiro(residuals)
            
            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(residuals)
            
            # D'Agostino's normality test
            da_stat, da_p = normaltest(residuals)
            
            return {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p, 'is_normal': shapiro_p > 0.05},
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p, 'is_normal': jb_p > 0.05},
                'dagostino': {'statistic': da_stat, 'p_value': da_p, 'is_normal': da_p > 0.05}
            }
        except Exception as e:
            return {'error': str(e)}

    def _perform_heteroscedasticity_tests(self, fitted_values: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform heteroscedasticity tests."""
        try:
            from scipy.stats import pearsonr, spearmanr
            
            # Breusch-Pagan test (correlation between fitted values and squared residuals)
            squared_residuals = residuals ** 2
            bp_corr, bp_p = pearsonr(fitted_values, squared_residuals)
            
            # White test (simplified version using Spearman correlation)
            white_corr, white_p = spearmanr(fitted_values, np.abs(residuals))
            
            return {
                'breusch_pagan': {'correlation': bp_corr, 'p_value': bp_p, 'is_homoscedastic': bp_p > 0.05},
                'white_test': {'correlation': white_corr, 'p_value': white_p, 'is_homoscedastic': white_p > 0.05}
            }
        except Exception as e:
            return {'error': str(e)}

    def _perform_autocorrelation_tests(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform autocorrelation tests on residuals."""
        try:
            # Durbin-Watson test (simplified calculation)
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals ** 2) / np.sum(residuals ** 2)
            
            # Ljung-Box test (simplified version using first-order autocorrelation)
            n = len(residuals)
            if n > 1:
                autocorr_1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                lb_stat = n * (n + 2) * (autocorr_1 ** 2) / (n - 1)
                # Approximate p-value using chi-square distribution
                from scipy.stats import chi2
                lb_p = 1 - chi2.cdf(lb_stat, df=1)
            else:
                autocorr_1 = 0
                lb_stat = 0
                lb_p = 1
            
            return {
                'durbin_watson': {'statistic': dw_stat, 'interpretation': 'no_autocorr' if 1.5 < dw_stat < 2.5 else 'autocorr'},
                'ljung_box': {'statistic': lb_stat, 'p_value': lb_p, 'no_autocorr': lb_p > 0.05},
                'first_order_autocorr': autocorr_1
            }
        except Exception as e:
            return {'error': str(e)}

    def _detect_outliers(self, residuals: np.ndarray, params: dict) -> Dict[str, Any]:
        """Detect outliers in residuals."""
        try:
            # Z-score method
            z_threshold = params.get('outlier_z_threshold', 3.0)
            z_scores = np.abs(residuals - np.mean(residuals)) / np.std(residuals)
            z_outliers = np.where(z_scores > z_threshold)[0]
            
            # IQR method
            q25, q75 = np.percentile(residuals, [25, 75])
            iqr = q75 - q25
            iqr_threshold = params.get('outlier_iqr_factor', 1.5)
            iqr_lower = q25 - iqr_threshold * iqr
            iqr_upper = q75 + iqr_threshold * iqr
            iqr_outliers = np.where((residuals < iqr_lower) | (residuals > iqr_upper))[0]
            
            return {
                'z_score_outliers': {
                    'indices': z_outliers.tolist(),
                    'count': len(z_outliers),
                    'percentage': len(z_outliers) / len(residuals) * 100,
                    'threshold': z_threshold
                },
                'iqr_outliers': {
                    'indices': iqr_outliers.tolist(),
                    'count': len(iqr_outliers),
                    'percentage': len(iqr_outliers) / len(residuals) * 100,
                    'lower_bound': iqr_lower,
                    'upper_bound': iqr_upper
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_model_fit_statistics(self, ref_data: np.ndarray, test_data: np.ndarray, 
                                      fitted_values: np.ndarray, residuals: np.ndarray, model_info: Dict) -> Dict[str, Any]:
        """Calculate model fit statistics."""
        try:
            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Adjusted R-squared
            n = len(test_data)
            p = 1 if model_info['model_type'] == 'linear' else len(model_info.get('coefficients', [1]))
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            # Mean Absolute Error
            mae = np.mean(np.abs(residuals))
            
            # Akaike Information Criterion (AIC)
            aic = n * np.log(ss_res / n) + 2 * p
            
            # Bayesian Information Criterion (BIC)
            bic = n * np.log(ss_res / n) + p * np.log(n)
            
            return {
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'rmse': rmse,
                'mae': mae,
                'aic': aic,
                'bic': bic,
                'n_parameters': p,
                'n_observations': n
            }
        except Exception as e:
            return {'error': str(e)}
    
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
        self._add_overlay_elements(ax, ref_data, residuals, plot_config, stats_results)
        
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
        
        degree = 2  # Standard quadratic polynomial
        
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
        
        # Test for autocorrelation (always run for diagnostic purposes)
        pattern_tests['autocorrelation'] = self._test_autocorrelation(residuals)
        
        # Test for normality (always run for diagnostic purposes)
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
    
    def _add_overlay_elements(self, ax, ref_data: np.ndarray, residuals: np.ndarray, 
                            plot_config: Dict[str, Any] = None, 
                            stats_results: Dict[str, Any] = None) -> None:
        """Add residual-specific overlay elements."""
        if plot_config is None:
            plot_config = {}
        
        # Add debug logging
        print(f"[ResidualOverlay] Plot config keys: {list(plot_config.keys())}")
        
        # Check for parameter compatibility (wizard may send different names)
        show_zero_line = plot_config.get('show_zero_line', True)
        
        print(f"[ResidualOverlay] Overlays - zero_line: {show_zero_line}")
        
        # Add zero line (conditional on overlay option)
        if show_zero_line:
            if plot_config.get('show_legend', False):
                ax.axhline(y=0, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Zero Line')
                print(f"[ResidualOverlay] Added y=0 line with label")
            else:
                ax.axhline(y=0, color='red', linestyle='-', alpha=0.8, linewidth=2)
                print(f"[ResidualOverlay] Added y=0 line without label")
        
        # Add statistical results text
        if plot_config.get('show_statistical_results', True) and stats_results:
            self._add_statistical_text(ax, stats_results, plot_config)
            print(f"[ResidualOverlay] Added statistical text")
 
    def _add_statistical_text(self, ax, stats_results: Dict[str, Any], plot_config: Dict[str, Any] = None) -> None:
        """Add statistical results as text on residual plot."""
        try:
            text_lines = []
            
            # Add model fit information
            if 'model_fit_stats' in stats_results:
                model_fit = stats_results['model_fit_stats']
                r_squared = model_fit.get('r_squared', np.nan)
                if not np.isnan(r_squared):
                    text_lines.append(f"R²: {r_squared:.3f}")
                
                rmse = model_fit.get('rmse', np.nan)
                if not np.isnan(rmse):
                    text_lines.append(f"RMSE: {rmse:.3f}")
            
            # Add residual statistics
            if 'residual_stats' in stats_results:
                res_stats = stats_results['residual_stats']
                mean_res = res_stats.get('mean', np.nan)
                if not np.isnan(mean_res):
                    text_lines.append(f"Mean Residual: {mean_res:.3f}")
                
                std_res = res_stats.get('std', np.nan)
                if not np.isnan(std_res):
                    text_lines.append(f"Std Residual: {std_res:.3f}")
            
            # Add normality test results
            if 'normality_tests' in stats_results:
                norm_tests = stats_results['normality_tests']
                if 'shapiro_wilk' in norm_tests:
                    shapiro_test = norm_tests['shapiro_wilk']
                    is_normal = shapiro_test.get('is_normal', False)
                    p_value = shapiro_test.get('p_value', np.nan)
                    if not np.isnan(p_value):
                        text_lines.append(f"Normality: {'Yes' if is_normal else 'No'} (p={p_value:.3f})")
            
            # Add outlier information
            if 'outlier_analysis' in stats_results:
                outlier_analysis = stats_results['outlier_analysis']
                if 'z_score_outliers' in outlier_analysis:
                    z_outliers = outlier_analysis['z_score_outliers']
                    n_outliers = z_outliers.get('count', 0)
                    outlier_pct = z_outliers.get('percentage', 0)
                    text_lines.append(f"Outliers: {n_outliers} ({outlier_pct:.1f}%)")
            
            # Add autocorrelation test results
            if 'autocorrelation_tests' in stats_results:
                autocorr_tests = stats_results['autocorrelation_tests']
                if 'durbin_watson' in autocorr_tests:
                    dw_test = autocorr_tests['durbin_watson']
                    interpretation = dw_test.get('interpretation', 'Unknown')
                    text_lines.append(f"Autocorrelation: {interpretation}")
            
            if text_lines:
                text = '\n'.join(text_lines)
                ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"[ResidualOverlay] Error adding statistical text: {e}")
    
    def _format_statistical_text(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format statistical results for text overlay."""
        lines = []
        
        mean_residual = stats_results.get('mean_residual', np.nan)
        if not np.isnan(mean_residual):
            lines.append(f"Mean Residual: {mean_residual:.3f}")
        
        std_residual = stats_results.get('std_residual', np.nan)
        if not np.isnan(std_residual):
            lines.append(f"Std Residual: {std_residual:.3f}")
        
        n_samples = stats_results.get('n_samples', 0)
        if n_samples > 0:
            lines.append(f"N: {n_samples}")
        
        return lines
    
    def _get_overlay_functional_properties(self, overlay_id: str, overlay_type: str, 
                                         stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get functional properties for residual overlays (no arbitrary styling)."""
        properties = {}
        
        if overlay_id == 'show_zero_line' and overlay_type == 'line':
            properties.update({
                'y_value': 0,
                'label': 'Zero Line'
            })
        elif overlay_id == 'show_statistical_results' and overlay_type == 'text':
            properties.update({
                'position': (0.02, 0.98),
                'text_lines': self._format_statistical_text(stats_results)
            })
        elif overlay_id == 'show_legend' and overlay_type == 'legend':
            properties.update({
                'label': 'Legend'
            })
        
        return properties
    
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