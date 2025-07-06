"""
Cross-Correlation Comparison Method
"""

import numpy as np
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class CrossCorrelationComparison(BaseComparison):
    name = "Cross-Correlation"
    description = "Compute cross-correlation and lag between two signals"
    category = "Time Shift"
    version = "1.0.0"

    parameters = {
        'max_lag': {
            'type': int,
            'default': 50,
            'min': 1,
            'max': 1000,
            'description': 'Maximum Lag',
            'tooltip': 'Maximum lag to compute cross-correlation'
        },
        'normalize': {
            'type': bool,
            'default': True,
            'description': 'Normalize',
            'tooltip': 'Normalize cross-correlation values'
        },
        'find_peak': {
            'type': bool,
            'default': True,
            'description': 'Find Peak',
            'tooltip': 'Automatically find the peak cross-correlation and its lag'
        }
    }

    output_types = ["cross_correlation", "plot_data"]
    plot_type = "cross_correlation"
    requires_pairs = True  # This method needs individual pair data

    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)

        corr = np.correlate(test_clean - np.mean(test_clean), ref_clean - np.mean(ref_clean), mode='full')
        lags = np.arange(-len(ref_clean)+1, len(ref_clean))
        max_corr_idx = np.argmax(corr)
        best_lag = lags[max_corr_idx]

        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'cross_correlation': {
                'max_correlation': corr[max_corr_idx],
                'lag': best_lag
            },
            'plot_data': {
                'correlation_values': corr,
                'lags': lags
            },
            'interpretation': f"Max correlation at lag = {best_lag}"
        }
        self.results = results
        return results
    
    def generate_plot_content(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any] = None, 
                             checked_pairs: list = None) -> None:
        """Generate cross-correlation plot content"""
        try:
            if not checked_pairs:
                ax.text(0.5, 0.5, 'No pairs selected for cross-correlation', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            max_correlation = 0
            best_lag = 0
            
            for i, pair in enumerate(checked_pairs):
                pair_name = pair['name']
                # This would need to be implemented to get pair-specific data
                # For now, using the combined data as a fallback
                
                if len(ref_data) == 0:
                    continue
                
                # Filter valid data
                valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                ref_clean = ref_data[valid_mask]
                test_clean = test_data[valid_mask]
                
                if len(ref_clean) < 10:
                    continue
                
                # Calculate cross-correlation
                max_lag = min(50, len(ref_clean) // 4)
                correlation = np.correlate(test_clean - np.mean(test_clean), 
                                         ref_clean - np.mean(ref_clean), mode='full')
                correlation = correlation / (np.std(ref_clean) * np.std(test_clean) * len(ref_clean))
                
                # Create lag array
                lags = np.arange(-max_lag, max_lag + 1)
                mid = len(correlation) // 2
                
                # Extract relevant portion
                start_idx = max(0, mid - max_lag)
                end_idx = min(len(correlation), mid + max_lag + 1)
                correlation_subset = correlation[start_idx:end_idx]
                
                if len(correlation_subset) != len(lags):
                    lags = lags[:len(correlation_subset)]
                
                # Plot cross-correlation
                color = colors[i % len(colors)]
                ax.plot(lags, correlation_subset, color=color, linewidth=2, 
                       label=f'{pair_name}', alpha=0.8)
                
                # Find peak
                peak_idx = np.argmax(np.abs(correlation_subset))
                peak_lag = lags[peak_idx]
                peak_corr = correlation_subset[peak_idx]
                
                if abs(peak_corr) > abs(max_correlation):
                    max_correlation = peak_corr
                    best_lag = peak_lag
                
                # Mark peak
                ax.plot(peak_lag, peak_corr, 'o', color=color, markersize=8, 
                       markerfacecolor='white', markeredgewidth=2)
            
            # Add zero line
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Add statistics
            stats_text = f'Best correlation: {max_correlation:.3f}\n'
            stats_text += f'at lag: {best_lag}'
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)
            
            ax.set_xlabel('Lag')
            ax.set_ylabel('Cross-Correlation')
            ax.set_title('Cross-Correlation Analysis')
            ax.legend(loc='upper right')
            
        except Exception as e:
            print(f"[CrossCorrelation] Error in plot generation: {e}")
            ax.text(0.5, 0.5, f'Error generating cross-correlation plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)