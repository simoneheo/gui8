"""
Correlation Comparison Method

This module implements correlation analysis between two data channels,
including Pearson, Spearman, and Kendall correlation coefficients.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class CorrelationComparison(BaseComparison):
    """
    Correlation analysis comparison method.
    
    Computes various correlation coefficients and related statistics
    to assess the linear and monotonic relationships between datasets.
    """
    
    name = "correlation"
    description = "Compute Pearson, Spearman, and Kendall correlation coefficients with significance tests"
    category = "Statistical"
    version = "1.0.0"
    tags = ["scatter", "correlation", "statistical", "relationship", "linear", "monotonic"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "correlation_type", "type": "str", "default": "pearson", "options": ["pearson", "spearman", "all"], "help": "Type of correlation to compute"},
        {"name": "include_rmse", "type": "bool", "default": True, "help": "Include RMSE in statistical results"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before calculating correlations"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "options": ["iqr", "zscore"], "help": "Method for detecting outliers"}
    ]
    
    # Plot configuration
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_identity_line': {'default': True, 'label': 'Show y = x Line', 'tooltip': 'Show identity line for perfect correlation reference', 'type': 'line'},
        'show_regression_line': {'default': True, 'label': 'Show Regression Line', 'tooltip': 'Show best-fit regression line', 'type': 'line'},
        'show_regression_equation': {'default': False, 'label': 'Show Regression Equation', 'tooltip': 'Display regression line formula (y = mx + b)', 'type': 'text'},
        'show_statistical_results': {'default': True, 'label': 'Show Statistical Results', 'tooltip': 'Display correlation statistics on the plot', 'type': 'text'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the correlation analysis.
        
        Streamlined 3-step workflow:
        1. Validate input data (basic validation + remove NaN/infinite values)
        2. plot_script (core transformation + correlation computation)
        3. stats_script (statistical calculations)
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            # === STEP 1: VALIDATE INPUT DATA ===
            # Basic validation (shape, type, length compatibility)
            ref_data, test_data = self._validate_input_data(ref_data, test_data)
            # Remove NaN and infinite values
            ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
            
            # === STEP 2: PLOT SCRIPT (core transformation + correlation computation) ===
            x_data, y_data, plot_metadata = self.plot_script(ref_clean, test_clean, self.kwargs)
            
            # === STEP 3: STATS SCRIPT (statistical calculations) ===
            stats_results = self.stats_script(x_data, y_data, ref_clean, test_clean, self.kwargs)
            
            # Prepare plot data
            plot_data = {
                'ref_data': x_data,
                'test_data': y_data,
                'ref_data_orig': ref_clean,
                'test_data_orig': test_clean,
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
            raise RuntimeError(f"Correlation analysis failed: {str(e)}")

    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for correlation analysis
        
        This defines what gets plotted on X and Y axes for correlation visualization.
        
        Args:
            ref_data: Reference measurements (already cleaned of NaN/infinite values)
            test_data: Test measurements (already cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (x_data, y_data, metadata)
                x_data: Reference data for X-axis
                y_data: Test data for Y-axis
                metadata: Plot configuration dictionary
        """
        # Apply transformations if requested
        ref_transformed = self._apply_transformations(ref_data, params)
        test_transformed = self._apply_transformations(test_data, params)
        
        # Handle outlier removal
        if params.get("remove_outliers", False):
            ref_transformed, test_transformed = self._remove_outliers(ref_transformed, test_transformed, params)
        
        # Prepare metadata for plotting
        metadata = {
            'x_label': 'Reference Data',
            'y_label': 'Test Data',
            'title': 'Correlation Analysis',
            'plot_type': 'scatter',
            'transformation': params.get("transformation", "none"),
            'outliers_removed': params.get("remove_outliers", False),
            'show_identity_line': params.get("show_identity_line", True),
            'show_regression_line': params.get("show_regression_line", True)
        }
        
        return ref_transformed.tolist(), test_transformed.tolist(), metadata

    def _apply_transformations(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Apply data transformations."""
        transformation = params.get("transformation", "none")
        
        if transformation == "log":
            # Log transformation (handle negative/zero values)
            data_pos = np.where(data > 0, data, np.finfo(float).eps)
            return np.log(data_pos)
        elif transformation == "sqrt":
            # Square root transformation (handle negative values)
            data_pos = np.where(data >= 0, data, 0)
            return np.sqrt(data_pos)
        elif transformation == "square":
            # Square transformation
            return data ** 2
        elif transformation == "reciprocal":
            # Reciprocal transformation (handle zero values)
            data_nonzero = np.where(data != 0, data, np.finfo(float).eps)
            return 1 / data_nonzero
        elif transformation == "standardize":
            # Z-score standardization
            return (data - np.mean(data)) / np.std(data)
        elif transformation == "normalize":
            # Min-max normalization
            data_min, data_max = np.min(data), np.max(data)
            if data_max != data_min:
                return (data - data_min) / (data_max - data_min)
            else:
                return np.zeros_like(data)
        else:
            # No transformation
            return data

    def _remove_outliers(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers from both datasets."""
        outlier_method = params.get("outlier_method", "iqr")
        
        if outlier_method == "iqr":
            # IQR method on both datasets
            iqr_factor = params.get("outlier_iqr_factor", 1.5)
            
            # Reference data IQR
            ref_q25, ref_q75 = np.percentile(ref_data, [25, 75])
            ref_iqr = ref_q75 - ref_q25
            ref_lower = ref_q25 - iqr_factor * ref_iqr
            ref_upper = ref_q75 + iqr_factor * ref_iqr
            
            # Test data IQR
            test_q25, test_q75 = np.percentile(test_data, [25, 75])
            test_iqr = test_q75 - test_q25
            test_lower = test_q25 - iqr_factor * test_iqr
            test_upper = test_q75 + iqr_factor * test_iqr
            
            mask = ((ref_data >= ref_lower) & (ref_data <= ref_upper) & 
                   (test_data >= test_lower) & (test_data <= test_upper))
            
        elif outlier_method == "zscore":
            # Z-score method on both datasets
            z_threshold = params.get("outlier_z_threshold", 3.0)
            
            ref_z = np.abs((ref_data - np.mean(ref_data)) / np.std(ref_data))
            test_z = np.abs((test_data - np.mean(test_data)) / np.std(test_data))
            
            mask = (ref_z <= z_threshold) & (test_z <= z_threshold)
            
        else:
            # No outlier removal
            mask = np.ones(len(ref_data), dtype=bool)
            
        return ref_data[mask], test_data[mask]

    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        BACKWARD COMPATIBILITY + SAFETY WRAPPER: Calculate correlation statistics.
        
        This method maintains compatibility with existing code and provides comprehensive
        validation and error handling around the core statistical calculations.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing correlation statistics
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

    def stats_script(self, x_data: List[float], y_data: List[float], 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Statistical calculations for correlation analysis
        
        Args:
            x_data: Reference data (potentially transformed)
            y_data: Test data (potentially transformed)
            ref_data: Original reference data
            test_data: Original test data
            params: Method parameters dictionary
            
        Returns:
            Dictionary containing statistical results
        """
        ref_transformed = np.array(x_data)
        test_transformed = np.array(y_data)
        
        # Correlation analysis
        correlation_stats = self._calculate_correlation_statistics(ref_transformed, test_transformed)
        
        # Regression analysis
        regression_stats = self._calculate_regression_statistics(ref_transformed, test_transformed)
        
        # Agreement analysis
        agreement_stats = self._calculate_agreement_statistics(ref_transformed, test_transformed)
        
        # Outlier impact analysis
        outlier_impact = self._analyze_outlier_impact(ref_data, test_data, params)
        
        # Transformation impact analysis
        transformation_impact = self._analyze_transformation_impact(ref_data, test_data, params)
        
        # Robustness analysis
        robustness_stats = self._calculate_robustness_statistics(ref_transformed, test_transformed)
        
        stats_results = {
            'correlation_stats': correlation_stats,
            'regression_stats': regression_stats,
            'agreement_stats': agreement_stats,
            'outlier_impact': outlier_impact,
            'transformation_impact': transformation_impact,
            'robustness_stats': robustness_stats
        }
        
        return stats_results

    def _calculate_correlation_statistics(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate various correlation statistics."""
        try:
            from scipy.stats import pearsonr, spearmanr, kendalltau
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(ref_data, test_data)
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(ref_data, test_data)
            
            # Kendall's tau
            kendall_tau, kendall_p = kendalltau(ref_data, test_data)
            
            # Coefficient of determination
            r_squared = pearson_r ** 2
            
            return {
                'pearson_r': pearson_r,
                'pearson_p_value': pearson_p,
                'pearson_significant': pearson_p < 0.05,
                'spearman_r': spearman_r,
                'spearman_p_value': spearman_p,
                'spearman_significant': spearman_p < 0.05,
                'kendall_tau': kendall_tau,
                'kendall_p_value': kendall_p,
                'kendall_significant': kendall_p < 0.05,
                'r_squared': r_squared,
                'correlation_strength': self._classify_correlation_strength(abs(pearson_r))
            }
        except Exception as e:
            return {'error': str(e)}

    def _classify_correlation_strength(self, abs_r: float) -> str:
        """Classify correlation strength based on absolute correlation coefficient."""
        if abs_r >= 0.9:
            return "very_strong"
        elif abs_r >= 0.7:
            return "strong"
        elif abs_r >= 0.5:
            return "moderate"
        elif abs_r >= 0.3:
            return "weak"
        else:
            return "very_weak"

    def _calculate_regression_statistics(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate regression statistics."""
        try:
            from scipy.stats import linregress
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(ref_data, test_data)
            
            # Predicted values
            predicted = slope * ref_data + intercept
            residuals = test_data - predicted
            
            # Regression statistics
            mse = np.mean(residuals ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            # Confidence intervals (approximate)
            n = len(ref_data)
            t_value = 1.96  # Approximate for large samples
            slope_ci = slope + np.array([-1, 1]) * t_value * std_err
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err,
                'slope_ci_lower': slope_ci[0],
                'slope_ci_upper': slope_ci[1],
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'is_significant': p_value < 0.05,
                'perfect_correlation': abs(r_value) > 0.99
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_agreement_statistics(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate agreement statistics."""
        try:
            # Mean difference (bias)
            mean_diff = np.mean(test_data - ref_data)
            
            # Standard deviation of differences
            std_diff = np.std(test_data - ref_data)
            
            # Limits of agreement (95%)
            loa_lower = mean_diff - 1.96 * std_diff
            loa_upper = mean_diff + 1.96 * std_diff
            
            # Concordance correlation coefficient (CCC)
            mean_ref = np.mean(ref_data)
            mean_test = np.mean(test_data)
            var_ref = np.var(ref_data)
            var_test = np.var(test_data)
            covariance = np.mean((ref_data - mean_ref) * (test_data - mean_test))
            
            ccc = (2 * covariance) / (var_ref + var_test + (mean_ref - mean_test)**2)
            
            # Intraclass correlation coefficient (ICC) approximation
            between_var = np.var((ref_data + test_data) / 2)
            within_var = np.var(ref_data - test_data) / 2
            icc = (between_var - within_var) / (between_var + within_var)
            
            return {
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'loa_lower': loa_lower,
                'loa_upper': loa_upper,
                'concordance_correlation': ccc,
                'intraclass_correlation': icc,
                'agreement_strength': self._classify_agreement_strength(ccc),
                'bias_direction': 'positive' if mean_diff > 0 else 'negative' if mean_diff < 0 else 'neutral'
            }
        except Exception as e:
            return {'error': str(e)}

    def _classify_agreement_strength(self, ccc: float) -> str:
        """Classify agreement strength based on concordance correlation coefficient."""
        if ccc >= 0.99:
            return "almost_perfect"
        elif ccc >= 0.95:
            return "substantial"
        elif ccc >= 0.90:
            return "moderate"
        elif ccc >= 0.80:
            return "poor"
        else:
            return "very_poor"

    def _analyze_outlier_impact(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """Analyze impact of outliers on correlation."""
        try:
            from scipy.stats import pearsonr
            
            # Original correlation
            orig_r, orig_p = pearsonr(ref_data, test_data)
            
            # Correlation without outliers
            ref_clean, test_clean = self._remove_outliers(ref_data, test_data, params)
            if len(ref_clean) > 2:
                clean_r, clean_p = pearsonr(ref_clean, test_clean)
                
                outlier_impact = {
                    'original_correlation': orig_r,
                    'clean_correlation': clean_r,
                    'correlation_change': clean_r - orig_r,
                    'outliers_removed': len(ref_data) - len(ref_clean),
                    'outlier_percentage': (len(ref_data) - len(ref_clean)) / len(ref_data) * 100,
                    'impact_magnitude': abs(clean_r - orig_r),
                    'impact_direction': 'strengthened' if clean_r > orig_r else 'weakened' if clean_r < orig_r else 'unchanged'
                }
            else:
                outlier_impact = {'error': 'Too few points after outlier removal'}
                
            return outlier_impact
        except Exception as e:
            return {'error': str(e)}

    def _analyze_transformation_impact(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """Analyze impact of data transformation on correlation."""
        try:
            from scipy.stats import pearsonr
            
            # Original correlation
            orig_r, orig_p = pearsonr(ref_data, test_data)
            
            # Transformed correlation
            ref_transformed = self._apply_transformations(ref_data, params)
            test_transformed = self._apply_transformations(test_data, params)
            trans_r, trans_p = pearsonr(ref_transformed, test_transformed)
            
            return {
                'original_correlation': orig_r,
                'transformed_correlation': trans_r,
                'correlation_change': trans_r - orig_r,
                'transformation_type': params.get("transformation", "none"),
                'improvement': trans_r > orig_r,
                'change_magnitude': abs(trans_r - orig_r),
                'original_significant': orig_p < 0.05,
                'transformed_significant': trans_p < 0.05
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_robustness_statistics(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate robustness statistics."""
        try:
            from scipy.stats import pearsonr, spearmanr
            
            # Bootstrap correlation estimates
            n_bootstrap = 1000
            bootstrap_correlations = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(ref_data), size=len(ref_data), replace=True)
                ref_boot = ref_data[indices]
                test_boot = test_data[indices]
                
                # Calculate correlation
                r_boot, _ = pearsonr(ref_boot, test_boot)
                bootstrap_correlations.append(r_boot)
            
            bootstrap_correlations = np.array(bootstrap_correlations)
            
            # Confidence intervals
            ci_lower = np.percentile(bootstrap_correlations, 2.5)
            ci_upper = np.percentile(bootstrap_correlations, 97.5)
            
            # Stability metrics
            bootstrap_std = np.std(bootstrap_correlations)
            bootstrap_mean = np.mean(bootstrap_correlations)
            
            return {
                'bootstrap_mean': bootstrap_mean,
                'bootstrap_std': bootstrap_std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'is_stable': bootstrap_std < 0.1,
                'coefficient_of_variation': bootstrap_std / abs(bootstrap_mean) if bootstrap_mean != 0 else np.inf
            }
        except Exception as e:
            return {'error': str(e)}
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate correlation plot with performance and overlay options.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        # Apply performance optimizations
        ref_plot, test_plot = self._apply_performance_optimizations(ref_data, test_data, plot_config)
        
        # Create density plot based on configuration
        self._create_density_plot(ax, ref_plot, test_plot, plot_config)
        
        # Add overlay elements
        self._add_overlay_elements(ax, ref_plot, test_plot, plot_config, stats_results)
        
        # Set labels and title
        ax.set_xlabel('Reference Data')
        ax.set_ylabel('Test Data')
        ax.set_title(f'{self.name.title()} Analysis')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if requested (use overlay option)
        if plot_config.get('show_legend', False):
            ax.legend()
    
    def _add_overlay_elements(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                            plot_config: Dict[str, Any] = None, 
                            stats_results: Dict[str, Any] = None) -> None:
        """
        Add correlation-specific overlay elements to the plot.
        
        Args:
            ax: Matplotlib axes object
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        print(f"[CorrelationOverlay] Plot config keys: {list(plot_config.keys())}")
        
        # Add identity line (y = x)
        if plot_config.get('show_identity_line', False):
            min_val = min(np.min(ref_data), np.min(test_data))
            max_val = max(np.max(ref_data), np.max(test_data))
            if plot_config.get('show_legend', False):
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, 
                       linewidth=2, label='y = x')
            else:
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, 
                       linewidth=2)
            print(f"[CorrelationOverlay] Added identity line")
        
        # Add regression line
        if plot_config.get('show_regression_line', False):
            self._add_regression_line(ax, ref_data, test_data, stats_results, plot_config)
            print(f"[CorrelationOverlay] Added regression line")
        
        # Add regression equation
        if plot_config.get('show_regression_equation', False):
            self._add_regression_equation(ax, ref_data, test_data, stats_results)
            print(f"[CorrelationOverlay] Added regression equation")
        
        # Add confidence bands around regression line
        if plot_config.get('show_confidence_bands', False):
            confidence_level = 0.95  # Standard 95% confidence level
            self._add_confidence_bands(ax, ref_data, test_data, confidence_level, stats_results, plot_config)
            print(f"[CorrelationOverlay] Added confidence bands")
        
        # Add statistical results text (includes R² if requested)
        if plot_config.get('show_statistical_results', False) and stats_results:
            self._add_statistical_text(ax, stats_results, plot_config)
            print(f"[CorrelationOverlay] Added statistical text")
    
    def _add_regression_line(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                           stats_results: Dict[str, Any] = None, plot_config: Dict[str, Any] = None) -> None:
        """Add regression line to correlation plot."""
        try:
            # Get regression stats from results if available
            if stats_results and 'regression_stats' in stats_results:
                regression_stats = stats_results['regression_stats']
                slope = regression_stats.get('slope', None)
                intercept = regression_stats.get('intercept', None)
                r_value = regression_stats.get('r_value', None)
            else:
                # Fallback to computing regression
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(ref_data, test_data)
            
            if slope is not None and intercept is not None:
                x_line = np.array([np.min(ref_data), np.max(ref_data)])
                y_line = slope * x_line + intercept
                
                # Use plot_config parameter for legend visibility
                show_legend = plot_config.get('show_legend', False) if plot_config else False
                if show_legend:
                    ax.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2, label='Regression Line')
                else:
                    ax.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2)
        except Exception as e:
            print(f"[CorrelationOverlay] Error adding regression line: {e}")
    
    def _add_regression_equation(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                               stats_results: Dict[str, Any] = None) -> None:
        """Add regression equation text to correlation plot."""
        try:
            # Get regression stats from results if available
            if stats_results and 'regression_stats' in stats_results:
                regression_stats = stats_results['regression_stats']
                slope = regression_stats.get('slope', None)
                intercept = regression_stats.get('intercept', None)
                r_value = regression_stats.get('r_value', None)
            else:
                # Fallback to computing regression
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(ref_data, test_data)
            
            if slope is not None and intercept is not None:
                # Format the equation
                if intercept >= 0:
                    equation = f'y = {slope:.3f}x + {intercept:.3f}'
                else:
                    equation = f'y = {slope:.3f}x - {abs(intercept):.3f}'
                
                # Add equation text without box (just plain text)
                ax.text(0.05, 0.95, equation, 
                       transform=ax.transAxes, fontsize=10, 
                       color='red', weight='bold')
        except Exception as e:
            print(f"[CorrelationOverlay] Error adding regression equation: {e}")
    
    def _add_confidence_bands(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                            confidence_level: float, stats_results: Dict[str, Any] = None, 
                            plot_config: Dict[str, Any] = None) -> None:
        """Add confidence bands around regression line."""
        try:
            # Get regression stats from results if available
            if stats_results and 'regression_stats' in stats_results:
                regression_stats = stats_results['regression_stats']
                slope = regression_stats.get('slope', None)
                intercept = regression_stats.get('intercept', None)
                std_err = regression_stats.get('std_err', None)
            else:
                # Fallback to computing regression
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(ref_data, test_data)
            
            if slope is not None and intercept is not None and std_err is not None:
                x_line = np.linspace(np.min(ref_data), np.max(ref_data), 100)
                y_line = slope * x_line + intercept
                
                n = len(ref_data)
                x_mean = np.mean(ref_data)
                x_std = np.std(ref_data)
                
                if x_std > 0:
                    se_pred = std_err * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / (n * x_std**2))
                    
                    from scipy.stats import t
                    t_value = t.ppf((1 + confidence_level) / 2, n - 2)
                    ci_lower = y_line - t_value * se_pred
                    ci_upper = y_line + t_value * se_pred
                    
                    # Use plot_config parameter for legend visibility
                    show_legend = plot_config.get('show_legend', False) if plot_config else False
                    if show_legend:
                        ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color='red', label='95% Confidence Bands')
                    else:
                        ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color='red')
        except Exception as e:
            print(f"[CorrelationOverlay] Error adding confidence bands: {e}")
    

    
    def _add_statistical_text(self, ax, stats_results: Dict[str, Any], plot_config: Dict[str, Any] = None) -> None:
        """Add statistical results as text on correlation plot."""
        try:
            text_lines = []
            
            # Add correlation results
            if 'correlation_stats' in stats_results:
                corr_stats = stats_results['correlation_stats']
                pearson_r = corr_stats.get('pearson_r', np.nan)
                pearson_p = corr_stats.get('pearson_p_value', np.nan)
                r_squared = corr_stats.get('r_squared', np.nan)
                
                if not np.isnan(pearson_r):
                    if not np.isnan(pearson_p):
                        text_lines.append(f"Pearson r: {pearson_r:.3f} (p={pearson_p:.3f})")
                    else:
                        text_lines.append(f"Pearson r: {pearson_r:.3f}")
                
                # Add R² value
                if not np.isnan(r_squared):
                    text_lines.append(f"R²: {r_squared:.3f}")
                
                # Add Spearman if available
                spearman_r = corr_stats.get('spearman_r', np.nan)
                if not np.isnan(spearman_r):
                    text_lines.append(f"Spearman ρ: {spearman_r:.3f}")
            
            # Add regression results
            if 'regression_stats' in stats_results:
                reg_stats = stats_results['regression_stats']
                rmse = reg_stats.get('rmse', np.nan)
                if not np.isnan(rmse):
                    text_lines.append(f"RMSE: {rmse:.3f}")
            
            # Add agreement results
            if 'agreement_stats' in stats_results:
                agree_stats = stats_results['agreement_stats']
                mean_diff = agree_stats.get('mean_difference', np.nan)
                if not np.isnan(mean_diff):
                    text_lines.append(f"Mean Bias: {mean_diff:.3f}")
            
            if text_lines:
                text = '\n'.join(text_lines)
                ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"[CorrelationOverlay] Error adding statistical text: {e}")
    
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
        """Compute Spearman correlation coefficient."""
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
        """Compute Kendall correlation coefficient."""
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
    
    def _compute_rmse_metrics(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute RMSE and related metrics."""
        try:
            differences = test_data - ref_data
            rmse = np.sqrt(np.mean(differences ** 2))
            mae = np.mean(np.abs(differences))
            mse = np.mean(differences ** 2)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mse': mse
            }
        except Exception as e:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mse': np.nan,
                'error': str(e)
            }
    
    def _interpret_correlations(self, correlations: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Interpret correlation results."""
        interpretation = {}
        
        for corr_type, corr_data in correlations.items():
            if 'coefficient' not in corr_data or np.isnan(corr_data['coefficient']):
                interpretation[corr_type] = "Could not compute correlation"
                continue
            
            r = corr_data['coefficient']
            p_value = corr_data.get('p_value', np.nan)
            
            # Strength interpretation
            if abs(r) >= 0.9:
                strength = "Very Strong"
            elif abs(r) >= 0.7:
                strength = "Strong"
            elif abs(r) >= 0.5:
                strength = "Moderate"
            elif abs(r) >= 0.3:
                strength = "Weak"
            else:
                strength = "Very Weak"
            
            # Direction
            direction = "Positive" if r > 0 else "Negative"
            
            # Significance
            if not np.isnan(p_value):
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                interpretation[corr_type] = f"{strength} {direction} ({r:.3f}, p={p_value:.3f}, {significance})"
            else:
                interpretation[corr_type] = f"{strength} {direction} ({r:.3f})"
        
        return interpretation
    
    def _format_statistical_text(self, stats_results: Dict[str, Any], plot_config: Dict[str, Any] = None) -> List[str]:
        """Format statistical results for display on plot."""
        text_lines = []
        
        if plot_config is None:
            plot_config = {}
        
        # Add correlation results
        if 'correlations' in stats_results:
            for corr_type, corr_data in stats_results['correlations'].items():
                if 'coefficient' in corr_data and not np.isnan(corr_data['coefficient']):
                    r = corr_data['coefficient']
                    p = corr_data.get('p_value', np.nan)
                    if not np.isnan(p):
                        text_lines.append(f"{corr_type.title()}: r={r:.3f}, p={p:.3f}")
                    else:
                        text_lines.append(f"{corr_type.title()}: r={r:.3f}")
        
        # Add R² if available
        if ('correlations' in stats_results and 'pearson' in stats_results['correlations']):
            pearson_data = stats_results['correlations']['pearson']
            r_squared = pearson_data.get('r_squared', np.nan)
            if not np.isnan(r_squared):
                text_lines.append(f"R²: {r_squared:.3f}")
        
        # Add RMSE if available
        if 'rmse_metrics' in stats_results:
            rmse = stats_results['rmse_metrics'].get('rmse', np.nan)
            if not np.isnan(rmse):
                text_lines.append(f"RMSE: {rmse:.3f}")
        
        # Add interpretation
        if 'interpretation' in stats_results:
            for corr_type, interpretation in stats_results['interpretation'].items():
                if len(interpretation) < 50:  # Only show short interpretations
                    text_lines.append(f"{corr_type.title()}: {interpretation}")
        
        return text_lines
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Correlation Analysis",
            "description": "Measures relationships between datasets using correlation coefficients",
            "interpretation": {
                "pearson": "Linear correlation (r), assumes normal distribution",
                "spearman": "Rank-based correlation (ρ), non-parametric, robust to outliers",
                "kendall": "Rank-based correlation (τ), more robust but computationally intensive",
                "strength": "Strong: |r| > 0.7, Moderate: 0.3 < |r| < 0.7, Weak: |r| < 0.3"
            },
            "use_cases": [
                "Assessing how well two variables move together",
                "Method validation and agreement assessment",
                "Feature correlation analysis",
                "Signal similarity assessment"
            ],
            "tips": [
                "Use Pearson for linear relationships with normal data",
                "Use Spearman for monotonic relationships or non-normal data",
                "Use Kendall for small samples or when robustness is important",
                "Check p-values for statistical significance",
                "Consider RMSE for absolute agreement assessment"
            ]
        } 
    
    def _get_overlay_functional_properties(self, overlay_id: str, overlay_type: str, 
                                         stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get functional properties for correlation overlays (no arbitrary styling)."""
        properties = {}
        
        if overlay_id == 'show_identity_line' and overlay_type == 'line':
            properties.update({
                'label': 'y = x'
            })
        elif overlay_id == 'show_regression_line' and overlay_type == 'line':
            # Get regression statistics to compute line data
            if 'regression_stats' in stats_results:
                reg_stats = stats_results['regression_stats']
                slope = reg_stats.get('slope')
                intercept = reg_stats.get('intercept')
                
                if slope is not None and intercept is not None:
                    # Get data range from scatter data (approximate from stats)
                    # We'll use a reasonable range that will be adjusted by RenderPlotOp
                    x_min = -10  # Will be replaced by actual data range
                    x_max = 10   # Will be replaced by actual data range
                    
                    # Compute regression line points
                    x_data = [x_min, x_max]
                    y_data = [slope * x_min + intercept, slope * x_max + intercept]
                    
                    properties.update({
                        'label': 'Regression Line',
                        'x_data': x_data,
                        'y_data': y_data,
                        'slope': slope,
                        'intercept': intercept
                    })
                else:
                    properties.update({
                        'label': 'Regression Line'
                    })
            else:
                properties.update({
                    'label': 'Regression Line'
                })
        elif overlay_id == 'show_regression_equation' and overlay_type == 'text':
            properties.update({
                'position': (0.05, 0.95),
                'text_lines': self._format_regression_equation(stats_results)
            })
        elif overlay_id == 'show_confidence_bands' and overlay_type == 'fill':
            properties.update({
                'fill_between': True,
                'confidence_level': 0.95,
                'label': '95% Confidence Bands'
            })
        elif overlay_id == 'show_statistical_results' and overlay_type == 'text':
            properties.update({
                'position': (0.02, 0.98),
                'text_lines': self._format_correlation_stats(stats_results)
            })
        elif overlay_id == 'show_legend' and overlay_type == 'legend':
            properties.update({
                'label': 'Legend'
            })
        
        return properties
    
    def _format_regression_equation(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format regression equation for text overlay."""
        lines = []
        
        if 'regression_stats' in stats_results:
            reg_stats = stats_results['regression_stats']
            slope = reg_stats.get('slope')
            intercept = reg_stats.get('intercept')
            
            if slope is not None and intercept is not None:
                if intercept >= 0:
                    equation = f'y = {slope:.3f}x + {intercept:.3f}'
                else:
                    equation = f'y = {slope:.3f}x - {abs(intercept):.3f}'
                lines.append(equation)
        
        return lines
    
    def _format_correlation_stats(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format correlation statistics for text overlay."""
        lines = []
        
        if 'correlation_stats' in stats_results:
            corr_stats = stats_results['correlation_stats']
            pearson_r = corr_stats.get('pearson_r', np.nan)
            if not np.isnan(pearson_r):
                lines.append(f"Pearson r: {pearson_r:.3f}")
            
            r_squared = corr_stats.get('r_squared', np.nan)
            if not np.isnan(r_squared):
                lines.append(f"R²: {r_squared:.3f}")
        
        if 'regression_stats' in stats_results:
            reg_stats = stats_results['regression_stats']
            rmse = reg_stats.get('rmse', np.nan)
            if not np.isnan(rmse):
                lines.append(f"RMSE: {rmse:.3f}")
        
        return lines 