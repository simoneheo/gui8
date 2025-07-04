"""
Intraclass Correlation Coefficient (ICC) Comparison Method
"""

import numpy as np
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class ICCComparison(BaseComparison):
    name = "Intraclass Correlation Coefficient"
    description = "Compute ICC for agreement and reliability"
    category = "Reliability"
    version = "1.0.0"

    parameters = {
        'icc_type': {
            'type': str,
            'default': 'ICC(2,1)',
            'choices': ['ICC(1,1)', 'ICC(2,1)', 'ICC(3,1)', 'ICC(1,k)', 'ICC(2,k)', 'ICC(3,k)'],
            'description': 'ICC Type',
            'tooltip': 'ICC(1,1): Each target measured by different raters\nICC(2,1): Random sample of raters\nICC(3,1): Fixed set of raters\nk versions: Average of multiple measurements'
        },
        'confidence_level': {
            'type': float,
            'default': 0.95,
            'min': 0.8,
            'max': 0.999,
            'description': 'Confidence Level',
            'tooltip': 'Confidence level for ICC confidence intervals (0.95 = 95%)'
        },
        'interpretation_scale': {
            'type': str,
            'default': 'cicchetti',
            'choices': ['cicchetti', 'fleiss', 'koo_li'],
            'description': 'Interpretation Scale',
            'tooltip': 'Cicchetti: Poor <0.40, Fair 0.40-0.59, Good 0.60-0.74, Excellent ≥0.75\nFleiss: Similar scale\nKoo & Li: More recent guidelines'
        },
        'bootstrap_ci': {
            'type': bool,
            'default': False,
            'description': 'Bootstrap CI',
            'tooltip': 'Use bootstrap method for confidence intervals\nMore robust than F-distribution method'
        },
        'bootstrap_samples': {
            'type': int,
            'default': 1000,
            'min': 100,
            'max': 10000,
            'description': 'Bootstrap Samples',
            'tooltip': 'Number of bootstrap resamples for confidence intervals'
        }
    }

    output_types = ["icc_statistic", "plot_data"]
    plot_type = "icc"

    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        data = np.vstack([ref_data, test_data]).T
        n, k = data.shape
        mean_per_target = np.mean(data, axis=1)
        ms_between = np.var(mean_per_target, ddof=1) * k
        ms_within = np.sum((data - mean_per_target[:, None])**2) / (n*(k-1))
        icc = (ms_between - ms_within) / (ms_between + (k-1)*ms_within)

        return {
            'method': self.name,
            'n_samples': n,
            'icc_statistic': icc,
            'plot_data': {
                'ref_data': ref_data,
                'test_data': test_data
            },
            'interpretation': f"ICC = {icc:.3f} (higher = better reliability)"
        }
    
    def generate_plot_content(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any] = None, 
                             checked_pairs: list = None) -> None:
        """Generate ICC plot content - scatter plot with ICC statistics"""
        try:
            ref_data = np.array(ref_data)
            test_data = np.array(test_data)
            
            if len(ref_data) == 0:
                ax.text(0.5, 0.5, 'No valid data for ICC analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Simple ICC calculation (ICC(2,1) - two-way random effects, single measures)
            n = len(ref_data)
            
            # Calculate means
            grand_mean = (np.mean(ref_data) + np.mean(test_data)) / 2
            subject_means = (ref_data + test_data) / 2
            
            # Calculate variance components (simplified)
            between_subject_var = np.var(subject_means, ddof=1)
            within_subject_var = np.mean([np.var([ref_data[i], test_data[i]], ddof=1) for i in range(n)])
            
            # ICC calculation
            icc = (between_subject_var - within_subject_var) / (between_subject_var + within_subject_var)
            icc = max(0, icc)  # ICC cannot be negative in this context
            
            # Add identity line
            min_val = min(np.min(ref_data), np.min(test_data))
            max_val = max(np.max(ref_data), np.max(test_data))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Agreement')
            
            # Add ICC statistics
            stats_text = f'ICC = {icc:.4f}\n'
            stats_text += f'n = {n:,} points\n'
            
            # Interpretation (Cicchetti guidelines)
            if icc >= 0.75:
                interpretation = 'Excellent'
            elif icc >= 0.60:
                interpretation = 'Good'
            elif icc >= 0.40:
                interpretation = 'Fair'
            else:
                interpretation = 'Poor'
            stats_text += f'Reliability: {interpretation}'
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)
            
            ax.set_xlabel('Reference')
            ax.set_ylabel('Test')
            ax.set_title('Intraclass Correlation Coefficient')
            
        except Exception as e:
            print(f"[ICC] Error in plot generation: {e}")
            ax.text(0.5, 0.5, f'Error generating ICC plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)