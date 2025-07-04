"""
Lin's Concordance Correlation Coefficient (CCC) Comparison Method
"""

import numpy as np
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class CCCComparison(BaseComparison):
    name = "Lin's CCC"
    description = "Compute Lin's Concordance Correlation Coefficient for agreement analysis"
    category = "Agreement"
    version = "1.0.0"

    parameters = {
        'confidence_level': {
            'type': float,
            'default': 0.95,
            'min': 0.8,
            'max': 0.999,
            'description': 'Confidence Level',
            'tooltip': 'Confidence level for CCC confidence intervals (0.95 = 95%)'
        },
        'bias_correction': {
            'type': bool,
            'default': True,
            'description': 'Bias Correction',
            'tooltip': 'Apply bias correction for small samples\nRecommended for n < 50'
        },
        'bootstrap_ci': {
            'type': bool,
            'default': False,
            'description': 'Bootstrap CI',
            'tooltip': 'Use bootstrap method for confidence intervals\nMore robust than analytical method'
        },
        'bootstrap_samples': {
            'type': int,
            'default': 1000,
            'min': 100,
            'max': 10000,
            'description': 'Bootstrap Samples',
            'tooltip': 'Number of bootstrap resamples for confidence intervals'
        },
        'interpretation_scale': {
            'type': str,
            'default': 'mcbride',
            'choices': ['mcbride', 'landis_koch', 'custom'],
            'description': 'Interpretation Scale',
            'tooltip': 'McBride: Poor <0.90, Moderate 0.90-0.95, Substantial 0.95-0.99, Excellent >0.99\nLandis-Koch: Similar to kappa interpretation\nCustom: User-defined thresholds'
        }
    }

    output_types = ["ccc_statistic", "plot_data"]
    plot_type = "ccc"

    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)

        mean_ref = np.mean(ref_clean)
        mean_test = np.mean(test_clean)
        var_ref = np.var(ref_clean)
        var_test = np.var(test_clean)
        cov = np.mean((ref_clean - mean_ref) * (test_clean - mean_test))

        ccc = (2 * cov) / (var_ref + var_test + (mean_ref - mean_test)**2)

        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'ccc_statistic': ccc,
            'plot_data': {
                'ref_data': ref_clean,
                'test_data': test_clean
            },
            'interpretation': f"CCC = {ccc:.3f} (perfect agreement = 1.0)"
        }
        self.results = results
        return results
    
    def generate_plot_content(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any] = None, 
                             checked_pairs: list = None) -> None:
        """Generate Lin's CCC plot content - scatter plot with CCC statistics"""
        try:
            ref_data = np.array(ref_data)
            test_data = np.array(test_data)
            
            if len(ref_data) == 0:
                ax.text(0.5, 0.5, 'No valid data for CCC analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Calculate CCC
            mean_ref = np.mean(ref_data)
            mean_test = np.mean(test_data)
            var_ref = np.var(ref_data)
            var_test = np.var(test_data)
            cov = np.mean((ref_data - mean_ref) * (test_data - mean_test))
            
            ccc = (2 * cov) / (var_ref + var_test + (mean_ref - mean_test)**2)
            
            # Add identity line
            min_val = min(np.min(ref_data), np.min(test_data))
            max_val = max(np.max(ref_data), np.max(test_data))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Agreement')
            
            # Add CCC statistics
            stats_text = f'Lin\'s CCC = {ccc:.4f}\n'
            stats_text += f'n = {len(ref_data):,} points\n'
            
            # Interpretation
            if ccc >= 0.99:
                interpretation = 'Excellent'
            elif ccc >= 0.95:
                interpretation = 'Substantial'
            elif ccc >= 0.90:
                interpretation = 'Moderate'
            else:
                interpretation = 'Poor'
            stats_text += f'Agreement: {interpretation}'
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)
            
            ax.set_xlabel('Reference')
            ax.set_ylabel('Test')
            ax.set_title('Lin\'s Concordance Correlation Coefficient')
            
        except Exception as e:
            print(f"[CCC] Error in plot generation: {e}")
            ax.text(0.5, 0.5, f'Error generating CCC plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)