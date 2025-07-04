"""
Dynamic Time Warping (DTW) Comparison Method
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class DTWComparison(BaseComparison):
    name = "Dynamic Time Warping"
    description = "Compare time-series alignment using DTW distance"
    category = "Shape"
    version = "1.0.0"

    parameters = {
        'distance_metric': {
            'type': str,
            'default': 'euclidean',
            'choices': ['euclidean', 'manhattan', 'cosine', 'correlation'],
            'description': 'Distance Metric',
            'tooltip': 'Euclidean: Standard geometric distance\nManhattan: Sum of absolute differences\nCosine: Angle-based similarity\nCorrelation: Pearson correlation distance'
        },
        'window_type': {
            'type': str,
            'default': 'sakoe_chiba',
            'choices': ['none', 'sakoe_chiba', 'itakura'],
            'description': 'Window Type',
            'tooltip': 'None: No constraint (slowest, most flexible)\nSakoe-Chiba: Parallelogram constraint\nItakura: Parallelogram constraint (faster)'
        },
        'window_size': {
            'type': int,
            'default': 10,
            'min': 1,
            'max': 100,
            'description': 'Window Size',
            'tooltip': 'Size of the warping window constraint\nLarger values allow more warping but increase computation time'
        },
        'step_pattern': {
            'type': str,
            'default': 'symmetric2',
            'choices': ['symmetric1', 'symmetric2', 'asymmetric'],
            'description': 'Step Pattern',
            'tooltip': 'Symmetric1: Simple diagonal steps\nSymmetric2: Diagonal and adjacent steps\nAsymmetric: Allows different warping for each signal'
        },
        'normalize_distance': {
            'type': bool,
            'default': True,
            'description': 'Normalize Distance',
            'tooltip': 'Normalize DTW distance by path length\nMakes distances comparable across different signal lengths'
        },
        'return_path': {
            'type': bool,
            'default': True,
            'description': 'Return Path',
            'tooltip': 'Return the optimal warping path\nUseful for visualizing how signals are aligned'
        }
    }

    output_types = ["dtw_distance", "plot_data"]
    plot_type = "dtw"
    requires_pairs = True  # This method needs individual pair data

    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)

        distance, path = fastdtw(ref_clean, test_clean, dist=euclidean)

        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'dtw_distance': distance,
            'plot_data': {
                'ref_data': ref_clean,
                'test_data': test_clean,
                'alignment_path': path
            },
            'interpretation': f"DTW distance = {distance:.3f} (lower is better alignment)"
        }
        self.results = results
        return results
    
    def generate_plot_content(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any] = None, 
                             checked_pairs: list = None) -> None:
        """Generate DTW plot content"""
        try:
            if not checked_pairs:
                ax.text(0.5, 0.5, 'No pairs selected for DTW analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            total_distance = 0
            pair_count = 0
            
            for i, pair in enumerate(checked_pairs):
                pair_name = pair['name']
                # For now, using the combined data as a fallback
                # This would need to be implemented to get pair-specific data
                
                if len(ref_data) == 0:
                    continue
                
                # Filter valid data
                valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                ref_clean = ref_data[valid_mask]
                test_clean = test_data[valid_mask]
                
                if len(ref_clean) < 10:
                    continue
                
                # Calculate DTW
                try:
                    distance, path = fastdtw(ref_clean, test_clean, dist=euclidean)
                    total_distance += distance
                    pair_count += 1
                    
                    # Create time indices
                    ref_time_idx = np.arange(len(ref_clean))
                    test_time_idx = np.arange(len(test_clean))
                    
                    # Plot original signals
                    color = colors[i % len(colors)]
                    ax.plot(ref_time_idx, ref_clean, color=color, linewidth=2, 
                           label=f'{pair_name} (Ref)', alpha=0.8)
                    ax.plot(test_time_idx, test_clean, color=color, linewidth=2, 
                           linestyle='--', label=f'{pair_name} (Test)', alpha=0.8)
                    
                    # Plot alignment path (simplified - just show some connections)
                    if len(path) > 0:
                        # Sample some alignment points to avoid overcrowding
                        step = max(1, len(path) // 20)
                        for j in range(0, len(path), step):
                            ref_idx, test_idx = path[j]
                            if ref_idx < len(ref_clean) and test_idx < len(test_clean):
                                ax.plot([ref_idx, test_idx], [ref_clean[ref_idx], test_clean[test_idx]], 
                                       color=color, alpha=0.3, linewidth=1)
                    
                except Exception as dtw_error:
                    print(f"[DTW] Error calculating DTW for {pair_name}: {dtw_error}")
                    continue
            
            # Add statistics
            if pair_count > 0:
                avg_distance = total_distance / pair_count
                stats_text = f'Average DTW distance: {avg_distance:.3f}\n'
                stats_text += f'Pairs analyzed: {pair_count}'
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', fontsize=10)
            
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Signal Value')
            ax.set_title('Dynamic Time Warping Analysis')
            ax.legend(loc='upper right')
            
        except Exception as e:
            print(f"[DTW] Error in plot generation: {e}")
            ax.text(0.5, 0.5, f'Error generating DTW plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)