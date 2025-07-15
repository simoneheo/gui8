"""
Residual Comparison Method

This module implements residual analysis for comparing two datasets,
including residual plots, pattern detection, and statistical analysis of residuals.
"""

import numpy as np
from scipy import stats
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
    tags = ["scatter", "residual", "error", "pattern", "regression", "outlier"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "fit_method", "type": "str", "default": "linear", "options": ["linear", "polynomial", "robust"], "help": "Linear (straight line), Polynomial (curved line), or Robust (outlier-resistant)"},
        {"name": "log_transform", "type": "bool", "default": False, "help": "Log transform the data"},
        {"name": "sqrt_transform", "type": "bool", "default": False, "help": "Square root transform the data"},
        {"name": "standardize_residuals", "type": "bool", "default": False, "help": "Standardize the residuals"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before calculating residual statistics"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "options": ["iqr", "zscore"], "help": "Method for detecting outliers: IQR (robust) or Z-score (assumes normal distribution)"}
    ]
    
    # Plot configuration  
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'statistical_results': {'default': True, 'label': 'Statistical Results', 'tooltip': 'Display residual statistics on the plot', 'type': 'text'}
    }

    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for residual analysis
        
        This defines what gets plotted on X and Y axes for residual plots.
        Includes outlier removal if requested in parameters.
        
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

        # ---------- initial sanitation -------------------------------------- #
        valid = np.isfinite(ref_data) & np.isfinite(test_data)
        ref, test = ref_data[valid], test_data[valid]

        # ---------- optional transforms ------------------------------------- #
        if params.get("log_transform", False):
            if (ref <= 0).any() or (test <= 0).any():
                raise ValueError("Log transform requested but data contain non‑positive values")
            ref, test = np.log(ref), np.log(test)

        if params.get("sqrt_transform", False):
            if (ref < 0).any() or (test < 0).any():
                raise ValueError("Sqrt transform requested but data contain negative values")
            ref, test = np.sqrt(ref), np.sqrt(test)

        # ---------- regression + residuals ---------------------------------- #
        # Fit regression model (previously _fit_regression function)
        fit_method = params.get("fit_method", "linear")

        if fit_method == "polynomial":
            degree = max(int(params.get("polynomial_degree", 2)), 1)
            coeffs = np.polyfit(ref, test, degree)
            fitted = np.polyval(coeffs, ref)
            model_info = {"type": "polynomial", "degree": degree, "coeffs": coeffs}
        else:  # default linear
            coeffs = np.polyfit(ref, test, 1)
            fitted = np.polyval(coeffs, ref)
            model_info = {
                "type": "linear",
                "slope": coeffs[0],
                "intercept": coeffs[1],
                "coeffs": coeffs,
            }

        residuals = test - fitted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((test - test.mean()) ** 2)
        model_info["r_squared"] = 1 - ss_res / ss_tot if ss_tot else 0.0
        model_info["residual_std"] = residuals.std(ddof=1) if residuals.size > 1 else 0.0

        # ---------- standardize residuals if requested ------------------- #
        if params.get("standardize_residuals", False):
            # Standardize residuals (previously _standardise function)
            sigma = model_info.get("residual_std", 1.0)
            residuals = residuals / sigma if sigma else residuals

        # ---------- outlier handling ---------------------------------------- #
        # Create outlier mask (previously _outlier_mask function)
        if params.get("remove_outliers", False):
            method = params.get("outlier_method", "iqr")
            
            if method == "zscore":
                z_threshold = 3
                z_scores = stats.zscore(residuals)
                # Handle case where zscore returns tuple (when there are NaN values)
                if isinstance(z_scores, tuple):
                    z_scores = z_scores[0]  # Take the first element if it's a tuple
                z = np.abs(z_scores)
                mask = z < z_threshold
            else:  # IQR default
                iqr_factor = 1.5
                q1, q3 = np.percentile(residuals, [25, 75])
                iqr = q3 - q1
                lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
                mask = (residuals >= lo) & (residuals <= hi)

            n_out = (~mask).sum()
            out_stats = None
            if n_out:
                out_stats = {
                    "method": method,
                    "n_outliers": int(n_out),
                    "outlier_pct": float(n_out) / residuals.size * 100,
                    "iqr_factor": 1.5 if method == "iqr" else None,
                    "z_threshold": 3 if method == "zscore" else None,
                }
            
            # Apply outlier mask
            ref, test = ref[mask], test[mask]
            fitted, residuals = fitted[mask], residuals[mask]
        else:
            # No outlier removal
            mask = np.ones_like(residuals, dtype=bool)
            out_stats = None

        # ---------- metadata ------------------------------------------------ #
        meta = {
            "x_label": "Fitted Values",
            "y_label": "Standardised Residuals" if params.get("standardize_residuals", False) else "Residuals",
            "title": "Residual Analysis",
            "model": model_info,
            "outlier_stats": out_stats,
        }

        x_data = fitted    
        y_data = residuals

        return x_data, y_data, meta
    
    
    def stats_script(self, x_data: list, y_data: list,
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Statistical calculations for residual analysis.

        Args:
            x_data: Fitted values
            y_data: Residuals
            ref_data: Original reference data
            test_data: Original test data
            params: Method parameters dictionary

        Returns:
            Dictionary containing statistical results
        """
        import numpy as np
        from scipy.stats import shapiro, pearsonr

        fitted_values = np.array(x_data)
        residuals = np.array(y_data)

        # Initial data cleaning
        valid = np.isfinite(ref_data) & np.isfinite(test_data)
        ref, test = ref_data[valid], test_data[valid]

        # Apply transforms if needed
        if params.get("log_transform", False):
            if (ref <= 0).any() or (test <= 0).any():
                ref, test = ref, test  # Skip transform if invalid
            else:
                ref, test = np.log(ref), np.log(test)

        if params.get("sqrt_transform", False):
            if (ref < 0).any() or (test < 0).any():
                ref, test = ref, test  # Skip transform if invalid
            else:
                ref, test = np.sqrt(ref), np.sqrt(test)

        # Fit regression model to get model info
        fit_method = params.get("fit_method", "linear")

        if fit_method == "polynomial":
            degree = max(int(params.get("polynomial_degree", 2)), 1)
            try:
                coeffs = np.polyfit(ref, test, degree)
                model_info = {"type": "polynomial", "degree": degree, "coeffs": coeffs}
            except:
                # Fallback to linear if polynomial fails
                coeffs = np.polyfit(ref, test, 1)
                model_info = {
                    "type": "linear",
                    "slope": coeffs[0],
                    "intercept": coeffs[1],
                    "coeffs": coeffs,
                }
        else:  # default linear
            try:
                coeffs = np.polyfit(ref, test, 1)
                model_info = {
                    "type": "linear",
                    "slope": coeffs[0],
                    "intercept": coeffs[1],
                    "coeffs": coeffs,
                }
            except:
                # Fallback values if fitting fails
                model_info = {
                    "type": "linear",
                    "slope": np.nan,
                    "intercept": np.nan,
                    "coeffs": [np.nan, np.nan],
                }

        # === Residual Descriptives ===
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals, ddof=1)
        residual_median = np.median(residuals)
        residual_min = np.min(residuals)
        residual_max = np.max(residuals)
        q25, q75 = np.percentile(residuals, [25, 75])
        iqr = q75 - q25

        # === Shapiro-Wilk normality test ===
        try:
            shapiro_result = shapiro(residuals)
            # Handle case where shapiro returns tuple
            if isinstance(shapiro_result, tuple):
                shapiro_stat, shapiro_p = shapiro_result
            else:
                shapiro_stat, shapiro_p = shapiro_result, np.nan
            is_normal = float(shapiro_p) > 0.05
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan
            is_normal = False

        # === Heteroscedasticity test (Breusch-Pagan proxy) ===
        try:
            pearson_result = pearsonr(fitted_values, np.abs(residuals))
            # Handle case where pearsonr returns tuple
            if isinstance(pearson_result, tuple):
                bp_corr, bp_p = pearson_result
            else:
                bp_corr, bp_p = pearson_result, np.nan
            # Ensure bp_p is a number for comparison
            bp_p_value = bp_p if isinstance(bp_p, (int, float)) else np.nan
            is_homoscedastic = bp_p_value > 0.05
        except Exception:
            bp_corr, bp_p = np.nan, np.nan
            is_homoscedastic = True

        # === Durbin-Watson autocorrelation test ===
        try:
            dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
            no_autocorr = 1.5 < dw_stat < 3
        except Exception:
            dw_stat, no_autocorr = np.nan, True

        # === Outlier analysis via z-score ===
        try:
            z_threshold = 3
            z_scores = np.abs(residuals - residual_mean) / residual_std if residual_std > 0 else np.zeros_like(residuals)
            outlier_mask = z_scores > z_threshold
            n_outliers = np.sum(outlier_mask)
            outlier_percentage = (n_outliers / len(residuals)) * 100
        except Exception:
            n_outliers, outlier_percentage, z_threshold = 0, 0.0, 2.5

        # === Model Fit Quality ===
        try:
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse = np.sqrt(np.mean(residuals ** 2))
            mae = np.mean(np.abs(residuals))
        except Exception:
            r_squared, rmse, mae = np.nan, np.nan, np.nan

        stats_results = {
            # Residual stats
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_median': residual_median,
            'residual_min': residual_min,
            'residual_max': residual_max,
            'residual_q25': q25,
            'residual_q75': q75,
            'residual_iqr': iqr,

            # Normality test
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'is_normal': is_normal,

            # Heteroscedasticity test
            'bp_correlation': bp_corr,
            'bp_p_value': bp_p,
            'is_homoscedastic': is_homoscedastic,

            # Autocorrelation test
            'durbin_watson': dw_stat,
            'no_autocorr': no_autocorr,

            # Outlier detection
            'n_outliers': n_outliers,
            'outlier_percentage': outlier_percentage,
            'outlier_threshold': z_threshold,

            # Model fit
            'r_squared': r_squared,
            'rmse': rmse,
            'mae': mae,
            'sample_size': len(residuals),

            # Model info
            'model_type': model_info.get('type', 'unknown'),
            'slope': model_info.get('slope', np.nan),
            'intercept': model_info.get('intercept', np.nan)
        }

        return stats_results

    def _create_overlays(self, ref_data: np.ndarray, test_data: np.ndarray, 
                        stats_results: Dict[str, Any], params: dict) -> Dict[str, Dict[str, Any]]:
        """
        Create overlay definitions for residual analysis.
        
        Returns a dictionary of overlay definitions that will be rendered by the base class.
        Each overlay definition contains type, main data, and style information.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            stats_results: Statistical results from stats_script
            params: Method parameters
            
        Returns:
            Dictionary of overlay definitions
        """
        
        
        statistical_results = {
            'type': 'text',
            'show': params.get('statistical_results', True),
            'label': 'Statistical Results',
            'main': self._get_statistical_results(stats_results)
        }
                        
        return {
            'statistical_results': statistical_results
        }
    
    
    def _get_statistical_results(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for statistical results text overlay."""
        # Only return the most informative statistics for text overlay
        essential_stats = {
            # Key residual stats
            'residual_mean': stats_results.get('residual_mean'),
            'residual_std': stats_results.get('residual_std'),
            'residual_iqr': stats_results.get('residual_iqr') }
        return essential_stats
    
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