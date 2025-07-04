"""
Root Mean Square Error (RMSE) Comparison Method
"""

import numpy as np
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class RMSEComparison(BaseComparison):
    name = "RMSE"
    description = "Compute Root Mean Square Error between signals"
    category = "Error Metric"
    version = "1.0.0"

    parameters = {
        'normalize_by': {
            'type': str,
            'default': 'none',
            'choices': ['none', 'mean', 'range', 'std'],
            'description': 'Normalize By',
            'tooltip': 'None: Raw RMSE\nMean: RMSE / mean(reference)\nRange: RMSE / range(reference)\nStd: RMSE / std(reference)'
        },
        'percentage_error': {
            'type': bool,
            'default': False,
            'description': 'Percentage Error',
            'tooltip': 'Calculate RMSE as percentage of reference values\nUseful for comparing across different scales'
        },
        'confidence_interval': {
            'type': bool,
            'default': True,
            'description': 'Confidence Interval',
            'tooltip': 'Calculate confidence interval for RMSE using bootstrap'
        },
        'bootstrap_samples': {
            'type': int,
            'default': 1000,
            'min': 100,
            'max': 10000,
            'description': 'Bootstrap Samples',
            'tooltip': 'Number of bootstrap resamples for confidence intervals'
        },
        'decompose_error': {
            'type': bool,
            'default': True,
            'description': 'Decompose Error',
            'tooltip': 'Decompose RMSE into systematic and random components\nHelps identify sources of error'
        },
        'outlier_robust': {
            'type': bool,
            'default': False,
            'description': 'Outlier Robust',
            'tooltip': 'Use median-based robust RMSE calculation\nLess sensitive to outliers'
        }
    }

    output_types = ["rmse_value", "plot_data"]
    plot_type = "rmse"

    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        rmse = np.sqrt(np.mean((test_clean - ref_clean) ** 2))

        return {
            'method': self.name,
            'n_samples': len(ref_clean),
            'rmse_value': rmse,
            'plot_data': {
                'ref_data': ref_clean,
                'test_data': test_clean,
                'residuals': test_clean - ref_clean
            },
            'interpretation': f"RMSE = {rmse:.3f}"
        }
    
    def generate_plot_content(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any] = None, 
                             checked_pairs: list = None) -> None:
        """Generate RMSE plot content - scatter plot with RMSE statistics"""
        try:
            ref_data = np.array(ref_data)
            test_data = np.array(test_data)
            
            if len(ref_data) == 0:
                ax.text(0.5, 0.5, 'No valid data for RMSE analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Calculate RMSE and related metrics
            rmse = np.sqrt(np.mean((test_data - ref_data) ** 2))
            mae = np.mean(np.abs(test_data - ref_data))
            bias = np.mean(test_data - ref_data)
            
            # Add identity line
            min_val = min(np.min(ref_data), np.min(test_data))
            max_val = max(np.max(ref_data), np.max(test_data))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Agreement')
            
            # Add error bands around identity line
            x_range = np.linspace(min_val, max_val, 100)
            ax.fill_between(x_range, x_range - rmse, x_range + rmse, alpha=0.2, color='red', label=f'±RMSE ({rmse:.3f})')
            
            # Add RMSE statistics
            stats_text = f'RMSE = {rmse:.4f}\n'
            stats_text += f'MAE = {mae:.4f}\n'
            stats_text += f'Bias = {bias:.4f}\n'
            stats_text += f'n = {len(ref_data):,} points'
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)
            
            ax.set_xlabel('Reference')
            ax.set_ylabel('Test')
            ax.set_title('Root Mean Square Error Analysis')
            
        except Exception as e:
            print(f"[RMSE] Error in plot generation: {e}")
            ax.text(0.5, 0.5, f'Error generating RMSE plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)