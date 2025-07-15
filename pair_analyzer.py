"""
Pair Analyzer

This module provides the PairAnalyzer class that processes pairs from a PairManager
using a specified method configuration, with intelligent caching for performance.
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from overlay import Overlay

class MethodConfigOp:
    """Represents a method configuration operation for caching purposes."""
    
    def __init__(self, method_name: str, parameters: Dict[str, Any], 
                 plot_script: Optional[str] = None, stats_script: Optional[str] = None,
                 performance_options: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[str] = None):
        self.method_name = method_name
        self.parameters = parameters
        self.plot_script = plot_script
        self.stats_script = stats_script
        self.performance_options = performance_options or {}
        self.timestamp = timestamp or str(time.time())
    
    def get_cache_key(self) -> str:
        """Generate a unique cache key for this method configuration."""
        config_data = {
            'method': self.method_name,
            'parameters': self.parameters,
            'plot_script': self.plot_script,
            'stats_script': self.stats_script,
            'performance_options': self.performance_options
        }
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def has_modified_plot_script(self) -> bool:
        """Check if plot script has been modified from default."""
        return self.plot_script is not None and self.plot_script.strip() != ""
    
    def has_modified_stats_script(self) -> bool:
        """Check if stats script has been modified from default."""
        return self.stats_script is not None and self.stats_script.strip() != ""
    
    def get_plot_script_content(self) -> str:
        """Get plot script content (modified or default)."""
        if self.has_modified_plot_script():
            return self.plot_script or ""
        return ""
    
    def get_stats_script_content(self) -> str:
        """Get stats script content (modified or default)."""
        if self.has_modified_stats_script():
            return self.stats_script or ""
        return ""


class PairAnalyzer:
    """
    Analyzes pairs using specified comparison methods with intelligent caching.
    
    The analyzer processes visible pairs from a PairManager, runs the appropriate
    comparison method, and generates both scatter data and global overlays.
    """
    
    def __init__(self, comparison_registry=None):
        """
        Initialize the PairAnalyzer.
        
        Args:
            comparison_registry: Registry containing comparison method classes
        """
        self.comparison_registry = comparison_registry
        self._cache = {}  # Internal cache: {cache_key: results}
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        
    def analyze(self, pair_manager, method_config: MethodConfigOp) -> Dict[str, Any]:
        """
        Analyze visible pairs using the specified method configuration.
        
        Args:
            pair_manager: PairManager instance containing pairs
            method_config: MethodConfigOp instance with method and parameters
            
        Returns:
            Dictionary containing:
            - overlays: List of global overlay objects
            - scatter_data: Combined plot data with styling
            - errors: Dictionary of any analysis errors
            - stats: Cache performance statistics
        """
        print(f"[PairAnalyzer] Starting analysis with method: {method_config.method_name}")
        
        # Get visible pairs
        visible_pairs = self._get_visible_pairs(pair_manager)
        if not visible_pairs:
            print("[PairAnalyzer] No visible pairs to analyze")
            return self._empty_results()
        
        print(f"[PairAnalyzer] Processing {len(visible_pairs)} visible pairs")
        
        # Get comparison method class
        comparison_cls = self._get_comparison_class(method_config.method_name)
        if not comparison_cls:
            return self._error_results(f"Comparison method '{method_config.method_name}' not found")
        
        # Get plot_type from comparison class
        plot_type = getattr(comparison_cls, 'plot_type', 'scatter')
        print(f"[PairAnalyzer] Using plot_type: {plot_type}")
        
        # Process each visible pair
        scatter_data = []
        all_stats_results = []
        errors = {}
        
        for pair in visible_pairs:
            try:
                # Check cache first
                cache_key = self._generate_pair_cache_key(pair.pair_id, method_config)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    self._cache_stats['hits'] += 1
                    print(f"[PairAnalyzer] Cache hit for pair: {pair.name}")
                    pair_scatter_data = cached_result['scatter_data']
                    pair_stats = cached_result['stats_results']
                else:
                    self._cache_stats['misses'] += 1
                    print(f"[PairAnalyzer] Cache miss for pair: {pair.name}, computing...")
                    
                    # Run analysis for this pair
                    pair_scatter_data, pair_stats = self._analyze_single_pair(
                        pair, comparison_cls, method_config
                    )
                    
                    # Cache the results
                    self._cache_result(cache_key, {
                        'scatter_data': pair_scatter_data,
                        'stats_results': pair_stats,
                        'computed_at': time.time()
                    })
                
                # Add to combined results
                scatter_data.append(pair_scatter_data)
                all_stats_results.append(pair_stats)
                
            except Exception as e:
                print(f"[PairAnalyzer] Error analyzing pair {pair.name}: {e}")
                errors[pair.pair_id] = str(e)
        
        # Generate global overlays from combined statistics
        overlays = self._generate_overlays(comparison_cls, all_stats_results, method_config, scatter_data)
        
        # Update line overlays to span the correct x_min/x_max based on visible data
        print(f"[PairAnalyzer] DEBUG: Calling _update_line_overlays_for_visible_data with {len(overlays)} overlays and {len(scatter_data)} scatter_data")
        overlays = self._update_line_overlays_for_visible_data(overlays, scatter_data, pair_manager)
        
        # Update cache statistics
        self._cache_stats['size'] = len(self._cache)
        
        print(f"[PairAnalyzer] Analysis complete. Cache stats: {self._cache_stats}")
        
        return {
            'overlays': overlays,
            'scatter_data': scatter_data,
            'errors': errors,
            'cache_stats': self._cache_stats.copy(),
            'method_name': method_config.method_name,
            'plot_type': plot_type,
            'n_pairs_processed': len(scatter_data)
        }
    
    def _get_visible_pairs(self, pair_manager) -> List[Any]:
        """Get list of pairs that should be processed (show=True)."""
        if not hasattr(pair_manager, 'get_visible_pairs'):
            print("[PairAnalyzer] PairManager missing get_visible_pairs method")
            return []
        
        visible_pairs = pair_manager.get_visible_pairs()
        print(f"[PairAnalyzer] Found {len(visible_pairs)} visible pairs")
        
        return visible_pairs
    
    def _get_comparison_class(self, method_name: str):
        """Get comparison class from registry."""
        if not self.comparison_registry:
            print("[PairAnalyzer] No comparison registry available")
            return None
        
        # Try to get the class from registry
        if hasattr(self.comparison_registry, 'get'):
            return self.comparison_registry.get(method_name)
        elif hasattr(self.comparison_registry, 'all'):
            all_methods = self.comparison_registry.all()
            return all_methods.get(method_name)
        else:
            print(f"[PairAnalyzer] Unknown registry interface: {type(self.comparison_registry)}")
            return None
    
    def _generate_pair_cache_key(self, pair_id: str, method_config: MethodConfigOp) -> str:
        """Generate cache key combining pair ID and method config."""
        config_key = method_config.get_cache_key()
        return f"{pair_id}_{config_key}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        return self._cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache analysis result."""
        self._cache[cache_key] = result
        
        # Simple cache size management - remove oldest if too large
        max_cache_size = 100  # Adjust based on memory constraints
        if len(self._cache) > max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = sorted(self._cache.keys(), 
                               key=lambda k: self._cache[k].get('computed_at', 0))
            for key in oldest_keys[:len(self._cache) - max_cache_size + 10]:
                del self._cache[key]
            print(f"[PairAnalyzer] Cache pruned to {len(self._cache)} entries")
    

    
    def _analyze_single_pair(self, pair, comparison_cls, method_config: MethodConfigOp) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Analyze a single pair using the comparison method.
        
        Returns:
            Tuple of (scatter_data, stats_results)
        """
        # Create comparison instance with parameters
        comparison = comparison_cls(**method_config.parameters)
        
        # Get aligned data from pair
        ref_data = pair.aligned_ref_data
        test_data = pair.aligned_test_data
        
        # Run plot_script (custom or default) to get scatter data
        x_data, y_data, plot_metadata = self._execute_plot_script(
            comparison, ref_data, test_data, method_config
        )
        
        # Run stats_script (custom or default) to get statistical results
        stats_results = self._execute_stats_script(
            comparison, x_data, y_data, ref_data, test_data, method_config
        )
        
        # Get plot_type from comparison class
        plot_type = getattr(comparison_cls, 'plot_type', 'scatter')
        
        # Prepare scatter data with pair styling
        scatter_data = {
            'x_data': x_data,
            'y_data': y_data,
            'pair_name': pair.name,
            'pair_id': pair.pair_id,
            'color': getattr(pair, 'color', '#1f77b4'),
            'alpha': getattr(pair, 'alpha', 0.6),
            'marker_size': getattr(pair, 'marker_size', 50),
            'edge_color': getattr(pair, 'edge_color', '#000000'),
            'edge_width': getattr(pair, 'edge_width', 1.0),
            'z_order': getattr(pair, 'z_order', 0),
            'n_points': len(x_data),
            'metadata': plot_metadata
        }
        
        # Handle plot_type specific styling
        if plot_type == "line":
            scatter_data['marker'] = '-'  # Line style instead of marker
            scatter_data['line_style'] = 'solid'
            print(f"[PairAnalyzer] Line plot styling applied for pair '{pair.name}'")
        elif plot_type == "histogram":
            # Histogram-specific styling for overlapping display
            scatter_data['bar_alpha'] = getattr(pair, 'alpha', 0.6)  # For overlapping transparency
            scatter_data['bar_edge_color'] = getattr(pair, 'edge_color', '#000000')
            scatter_data['bar_edge_width'] = getattr(pair, 'edge_width', 1.0)
            scatter_data['bar_fill_color'] = getattr(pair, 'color', '#1f77b4')
            scatter_data['bar_z_order'] = getattr(pair, 'z_order', 0)
            
            # Add histogram-specific metadata fields
            scatter_data['histogram_type'] = plot_metadata.get('histogram_type', 'counts')
            scatter_data['bin_widths'] = plot_metadata.get('bin_widths', [])
            scatter_data['bin_edges'] = plot_metadata.get('bin_edges', [])
            
            # For histograms, x_data represents bin centers/edges, y_data represents counts/frequencies
            print(f"[PairAnalyzer] Histogram styling applied for pair '{pair.name}'")
            print(f"[PairAnalyzer] Histogram data: {len(x_data)} bins, histogram_type: {scatter_data['histogram_type']}")
        elif plot_type == "scatter":
            scatter_data['marker'] = getattr(pair, 'marker_type', 'o')
            print(f"[PairAnalyzer] Standard marker styling applied for pair '{pair.name}'")
        
        # Debug logging for styling
        print(f"[PairAnalyzer] Pair '{pair.name}' scatter data:")
        print(f"  - color: {scatter_data['color']}")
        if 'marker' in scatter_data:
            print(f"  - marker: {scatter_data['marker']}")
        print(f"  - alpha: {scatter_data['alpha']}")
        print(f"  - n_points: {scatter_data['n_points']}")
        print(f"  - plot_type: {plot_type}")
        
        return scatter_data, stats_results
    
    def _generate_overlays(self, comparison_cls, all_stats_results: List[Dict[str, Any]], 
                          method_config: MethodConfigOp, scatter_data: List[Dict[str, Any]]) -> List[Overlay]:
        """
        Generate global overlay objects from combined statistics.
        
        Args:
            comparison_cls: Comparison method class
            all_stats_results: List of stats results from all pairs
            method_config: Method configuration
            scatter_data: List of scatter data from all pairs
            
        Returns:
            List of Overlay objects
        """
        overlays = []
        
        if not all_stats_results:
            return overlays
        
        # Combine statistics from all pairs using recomputation approach
        combined_stats = self._combine_statistics(all_stats_results, scatter_data, comparison_cls, method_config)
        
        print(f"[PairAnalyzer] Combined stats for overlay generation: {list(combined_stats.keys()) if combined_stats else 'None'}")
        
        # Create an instance of the comparison class to generate overlays
        try:
            comparison_instance = comparison_cls(**method_config.parameters)
            overlays = comparison_instance.generate_overlays(combined_stats)
            print(f"[PairAnalyzer] Generated {len(overlays)} overlays using {comparison_cls.name}")
        except Exception as e:
            print(f"[PairAnalyzer] Error generating overlays from {comparison_cls.name}: {e}")
            import traceback
            traceback.print_exc()
        
        return overlays
    
    def _combine_statistics(self, all_stats_results: List[Dict[str, Any]], scatter_data: List[Dict[str, Any]], 
                           comparison_cls, method_config: MethodConfigOp) -> Dict[str, Any]:
        """
        Combine statistics from multiple pairs by recomputing on combined data.
        
        This method concatenates x,y data from all visible pairs and recomputes 
        global statistics using the comparison method's stats_script.
        """
        if not all_stats_results or not scatter_data:
            return {}
        
        try:
            # Concatenate all x_data and y_data from visible pairs
            combined_x_data = []
            combined_y_data = []
            combined_ref_data = []
            combined_test_data = []
            
            for pair_scatter in scatter_data:
                x_data = pair_scatter.get('x_data', [])
                y_data = pair_scatter.get('y_data', [])
                
                if len(x_data) > 0 and len(y_data) > 0:
                    combined_x_data.extend(x_data)
                    combined_y_data.extend(y_data)
                    # For ref/test data, we'll use x_data/y_data as approximation
                    # In a perfect world, we'd have access to original aligned data
                    combined_ref_data.extend(x_data)
                    combined_test_data.extend(y_data)
            
            if not combined_x_data or not combined_y_data:
                print("[PairAnalyzer] No valid combined data for statistics computation")
                return all_stats_results[0] if all_stats_results else {}
            
            # Convert to numpy arrays if they aren't already
            import numpy as np
            combined_x_data = np.array(combined_x_data)
            combined_y_data = np.array(combined_y_data)
            combined_ref_data = np.array(combined_ref_data)
            combined_test_data = np.array(combined_test_data)
            
            # Create comparison instance and run stats_script on combined data
            comparison_instance = comparison_cls(**method_config.parameters)
            combined_stats = comparison_instance.stats_script(
                combined_x_data, combined_y_data, 
                combined_ref_data, combined_test_data, 
                method_config.parameters
            )
            
            print(f"[PairAnalyzer] Recomputed combined statistics from {len(scatter_data)} visible pairs")
            print(f"[PairAnalyzer] Combined data points: {len(combined_x_data)}")
            print(f"[PairAnalyzer] Combined stats keys: {list(combined_stats.keys()) if combined_stats else 'None'}")
            
            return combined_stats
            
        except Exception as e:
            print(f"[PairAnalyzer] Error recomputing combined statistics: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to first result if recomputation fails
            print("[PairAnalyzer] Falling back to first pair's statistics")
            return all_stats_results[0] if all_stats_results else {}
    
    def get_cached_results(self, pair_id: str, method_config: MethodConfigOp) -> Optional[Dict[str, Any]]:
        """Get cached results for a specific pair and method configuration."""
        cache_key = self._generate_pair_cache_key(pair_id, method_config)
        return self._get_cached_result(cache_key)
    
    def get_all_cached_results(self, method_config: MethodConfigOp) -> List[Dict[str, Any]]:
        """Get all cached results for a method configuration."""
        cached_results = []
        for cache_key, result in self._cache.items():
            if method_config.get_cache_key() in cache_key:
                cached_results.append(result)
        return cached_results
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'overlays': [],
            'scatter_data': [],
            'errors': {},
            'cache_stats': self._cache_stats.copy(),
            'method_name': None,
            'n_pairs_processed': 0
        }
    
    def _error_results(self, error_message: str) -> Dict[str, Any]:
        """Return error results structure."""
        print(f"[PairAnalyzer] Error: {error_message}")
        return {
            'overlays': [],
            'scatter_data': [],
            'errors': {'analyzer': error_message},
            'cache_stats': self._cache_stats.copy(),
            'method_name': None,
            'n_pairs_processed': 0
        }
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        print("[PairAnalyzer] Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            'size': len(self._cache),
            'stats': self._cache_stats.copy(),
            'keys': list(self._cache.keys())
        }
    
    def recombine_visible_pairs(self, pair_manager, method_config: MethodConfigOp) -> Dict[str, Any]:
        """
        Recombine cached statistics from only visible pairs to update overlays.
        
        This method is optimized for visibility toggles - it uses cached individual
        pair results but recombines them based on current visibility state.
        
        Args:
            pair_manager: PairManager instance containing pairs
            method_config: MethodConfigOp instance with method and parameters
            
        Returns:
            Dictionary containing updated overlays and scatter data for visible pairs
        """
        print(f"[PairAnalyzer] Recombining visible pairs for method: {method_config.method_name}")
        
        # Get visible pairs
        visible_pairs = self._get_visible_pairs(pair_manager)
        if not visible_pairs:
            print("[PairAnalyzer] No visible pairs to recombine")
            return self._empty_results()
        
        print(f"[PairAnalyzer] Recombining {len(visible_pairs)} visible pairs")
        
        # Get comparison method class
        comparison_cls = self._get_comparison_class(method_config.method_name)
        if not comparison_cls:
            return self._error_results(f"Comparison method '{method_config.method_name}' not found")
        
        # Collect cached results for visible pairs
        visible_scatter_data = []
        visible_stats_results = []
        missing_pairs = []
        
        for pair in visible_pairs:
            cache_key = self._generate_pair_cache_key(pair.pair_id, method_config)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                visible_scatter_data.append(cached_result['scatter_data'])
                visible_stats_results.append(cached_result['stats_results'])
                print(f"[PairAnalyzer] Using cached result for visible pair: {pair.name}")
            else:
                missing_pairs.append(pair)
                print(f"[PairAnalyzer] Missing cached result for pair: {pair.name}")
        
        # If we have missing pairs, we need to compute them
        if missing_pairs:
            print(f"[PairAnalyzer] Computing {len(missing_pairs)} missing pairs...")
            for pair in missing_pairs:
                try:
                    # Compute the missing pair
                    pair_scatter_data, pair_stats = self._analyze_single_pair(
                        pair, comparison_cls, method_config
                    )
                    
                    # Cache the result
                    cache_key = self._generate_pair_cache_key(pair.pair_id, method_config)
                    self._cache_result(cache_key, {
                        'scatter_data': pair_scatter_data,
                        'stats_results': pair_stats,
                        'computed_at': time.time()
                    })
                    
                    # Add to visible results
                    visible_scatter_data.append(pair_scatter_data)
                    visible_stats_results.append(pair_stats)
                    
                except Exception as e:
                    print(f"[PairAnalyzer] Error computing missing pair {pair.name}: {e}")
        
        # Generate overlays from combined statistics of visible pairs only
        overlays = self._generate_overlays(comparison_cls, visible_stats_results, method_config, visible_scatter_data)
        
        # Update line overlays to span the correct x_min/x_max based on visible data
        print(f"[PairAnalyzer] DEBUG: Calling _update_line_overlays_for_visible_data in recombine_visible_pairs with {len(overlays)} overlays and {len(visible_scatter_data)} scatter_data")
        overlays = self._update_line_overlays_for_visible_data(overlays, visible_scatter_data, pair_manager)
        
        print(f"[PairAnalyzer] Recombination complete: {len(visible_scatter_data)} scatter datasets, {len(overlays)} overlays")
        
        return {
            'overlays': overlays,
            'scatter_data': visible_scatter_data,
            'errors': {},
            'cache_stats': self._cache_stats.copy(),
            'method_name': method_config.method_name,
            'n_pairs_processed': len(visible_scatter_data),
            'recombined': True  # Flag to indicate this was a recombination
        }
    
    def _update_line_overlays_for_visible_data(self, overlays: List[Overlay], scatter_data: List[Dict[str, Any]], pair_manager) -> List[Overlay]:
        """
        Update line overlays to span the correct x_min/x_max and y_min/y_max based on currently visible data.
        
        Args:
            overlays: List of overlay objects
            scatter_data: List of scatter data from visible pairs
            
        Returns:
            Updated list of overlays with corrected line coordinates
        """
        print(f"[PairAnalyzer] DEBUG: _update_line_overlays_for_visible_data method called with {len(overlays) if overlays else 0} overlays")
        
        if not overlays or not scatter_data:
            print(f"[PairAnalyzer] DEBUG: No overlays ({len(overlays) if overlays else 0}) or scatter_data ({len(scatter_data) if scatter_data else 0}) for range update")
            return overlays
        
        print(f"[PairAnalyzer] DEBUG: Processing {len(overlays)} overlays for range update")
        for i, overlay in enumerate(overlays):
            print(f"[PairAnalyzer] DEBUG: Overlay {i}: id='{overlay.id}', name='{overlay.name}', type='{overlay.type}', show={overlay.show}")
        
        # Get visible data bounds from pair_manager
        visible_bounds = pair_manager.get_visible_data_bounds()
        
        print(f"[PairAnalyzer] DEBUG: visible_bounds from pair_manager: {visible_bounds}")
        
        if not visible_bounds or visible_bounds['xmin'] is None or visible_bounds['xmax'] is None:
            print("[PairAnalyzer] No visible data bounds available for line overlay updates")
            return overlays
        
        x_min = visible_bounds['xmin']
        x_max = visible_bounds['xmax']
        y_min = visible_bounds['ymin'] if visible_bounds['ymin'] is not None else -1
        y_max = visible_bounds['ymax'] if visible_bounds['ymax'] is not None else 1
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Add small padding to the ranges
        if x_range > 0:
            x_padding = 0.05 * x_range
            x_min -= x_padding
            x_max += x_padding
        else:
            x_min -= 0.1
            x_max += 0.1
            
        if y_range > 0:
            y_padding = 0.05 * y_range
            y_min -= y_padding
            y_max += y_padding
        else:
            y_min -= 0.1
            y_max += 0.1
        
        print(f"[PairAnalyzer] Updating line overlays for x range: [{x_min:.3f}, {x_max:.3f}], y range: [{y_min:.3f}, {y_max:.3f}]")
        
        # Update each line overlay
        for overlay in overlays:
            if overlay.type == 'line':
                # DEBUG: Show overlay data before update
                print(f"[PairAnalyzer] DEBUG: Processing LINE overlay '{overlay.id}' (name: '{overlay.name}', type: {overlay.type}): show={overlay.show}")
                if overlay.id == 'identity_line' or 'identity' in overlay.id.lower() or 'y = x' in overlay.name.lower():
                    print(f"[PairAnalyzer] DEBUG: **IDENTITY LINE CANDIDATE** - overlay.id='{overlay.id}', overlay.name='{overlay.name}'")
                if overlay.type == 'hline' or 'hline' in overlay.id.lower() or 'bias' in overlay.name.lower() or 'bias' in overlay.id.lower():
                    print(f"[PairAnalyzer] DEBUG: **HORIZONTAL LINE CANDIDATE** - overlay.id='{overlay.id}', overlay.name='{overlay.name}', overlay.type='{overlay.type}'")
                print(f"[PairAnalyzer] DEBUG: Before update - Line overlay data: {overlay.data}")
                print(f"[PairAnalyzer] DEBUG: Before update - Line overlay style: {overlay.style}")
                
                self._update_single_line_overlay(overlay, x_min, x_max, y_min, y_max)
                
                # DEBUG: Show overlay data after update
                print(f"[PairAnalyzer] DEBUG: After update - Overlay '{overlay.id}' ({overlay.name}): show={overlay.show}")
                print(f"[PairAnalyzer] DEBUG: After update - Overlay data: {overlay.data}")
                print(f"[PairAnalyzer] DEBUG: After update - Overlay style: {overlay.style}")
                
            elif overlay.type == 'vline':
                # DEBUG: Show vline overlay data before update
                print(f"[PairAnalyzer] DEBUG: Processing VLINE overlay '{overlay.id}' (name: '{overlay.name}', type: {overlay.type}): show={overlay.show}")
                print(f"[PairAnalyzer] DEBUG: Before update - Vline overlay data: {overlay.data}")
                
                self._update_single_vline_overlay(overlay, x_min, x_max, y_min, y_max)
                
                # DEBUG: Show vline overlay data after update
                print(f"[PairAnalyzer] DEBUG: After update - Vline overlay '{overlay.id}' ({overlay.name}): show={overlay.show}")
                print(f"[PairAnalyzer] DEBUG: After update - Vline data: {overlay.data}")
                
            elif overlay.type == 'fill':
                # DEBUG: Show fill overlay data before update
                print(f"[PairAnalyzer] DEBUG: Before update - Fill overlay '{overlay.id}' ({overlay.name}): show={overlay.show}")
                print(f"[PairAnalyzer] DEBUG: Before update - Fill data: {overlay.data}")
                
                self._update_single_fill_overlay(overlay, x_min, x_max, y_min, y_max)
                
                # DEBUG: Show fill overlay data after update
                print(f"[PairAnalyzer] DEBUG: After update - Fill overlay '{overlay.id}' ({overlay.name}): show={overlay.show}")
                print(f"[PairAnalyzer] DEBUG: After update - Fill data: {overlay.data}")
        
        return overlays
    
    def _execute_plot_script(self, comparison, ref_data, test_data, method_config: MethodConfigOp) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Execute plot script with fallback to default method.
        
        Args:
            comparison: Comparison instance
            ref_data: Reference data
            test_data: Test data
            method_config: Method configuration
            
        Returns:
            Tuple of (x_data, y_data, plot_metadata)
        """
        # Check if custom plot script should be used
        print(f"[PairAnalyzer] Checking for modified plot script: {method_config.has_modified_plot_script()}")
        if method_config.has_modified_plot_script():
            script_content = method_config.get_plot_script_content()
            print(f"[PairAnalyzer] Script content length: {len(script_content)}")
            print(f"[PairAnalyzer] Script content preview: {script_content[:100]}...")
            try:
                print(f"[PairAnalyzer] Executing custom plot script for {comparison.__class__.__name__}")
                return self._execute_custom_plot_script(
                    script_content,
                    ref_data, test_data, method_config.parameters
                )
            except Exception as e:
                print(f"[PairAnalyzer] Custom plot script failed: {e}")
                print("[PairAnalyzer] Falling back to default plot script")
        else:
            print("[PairAnalyzer] No modified plot script detected, using default")
        
        # Use default plot script
        try:
            result = comparison.plot_script(ref_data, test_data, method_config.parameters)
            if result is None or len(result) != 3:
                print(f"[PairAnalyzer] Default plot script returned invalid result: {result}")
                return ref_data, test_data, {}
            
            x_data, y_data, plot_metadata = result
            
            # Validate individual components
            if x_data is None:
                print("[PairAnalyzer] Default plot script returned None for x_data, using ref_data")
                x_data = ref_data
            if y_data is None:
                print("[PairAnalyzer] Default plot script returned None for y_data, using test_data")
                y_data = test_data
            if plot_metadata is None:
                plot_metadata = {}
                
            return x_data, y_data, plot_metadata
            
        except Exception as e:
            print(f"[PairAnalyzer] Default plot script failed: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal fallback data
            return ref_data, test_data, {}
    
    def _execute_stats_script(self, comparison, x_data, y_data, ref_data, test_data, method_config: MethodConfigOp) -> Dict[str, Any]:
        """
        Execute stats script with fallback to default method.
        
        Args:
            comparison: Comparison instance
            x_data: X data for plotting
            y_data: Y data for plotting
            ref_data: Reference data
            test_data: Test data
            method_config: Method configuration
            
        Returns:
            Dictionary of statistical results
        """
        # Check if custom stats script should be used
        print(f"[PairAnalyzer] Checking for modified stats script: {method_config.has_modified_stats_script()}")
        if method_config.has_modified_stats_script():
            script_content = method_config.get_stats_script_content()
            print(f"[PairAnalyzer] Stats script content length: {len(script_content)}")
            print(f"[PairAnalyzer] Stats script content preview: {script_content[:100]}...")
            try:
                print(f"[PairAnalyzer] Executing custom stats script for {comparison.__class__.__name__}")
                return self._execute_custom_stats_script(
                    script_content,
                    x_data, y_data, ref_data, test_data, method_config.parameters
                )
            except Exception as e:
                print(f"[PairAnalyzer] Custom stats script failed: {e}")
                print("[PairAnalyzer] Falling back to default stats script")
        else:
            print("[PairAnalyzer] No modified stats script detected, using default")
        
        # Use default stats script
        try:
            result = comparison.stats_script(x_data, y_data, ref_data, test_data, method_config.parameters)
            if result is None:
                print("[PairAnalyzer] Default stats script returned None, using empty dict")
                return {}
            return result
        except Exception as e:
            print(f"[PairAnalyzer] Default stats script failed: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal fallback stats
            return {}
    
    def _execute_custom_plot_script(self, script_content: str, ref_data, test_data, parameters: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Execute custom plot script in a controlled environment.
        
        Args:
            script_content: Custom script code (already processed by wizard)
            ref_data: Reference data
            test_data: Test data
            parameters: Method parameters
            
        Returns:
            Tuple of (x_data, y_data, plot_metadata)
        """
        # Script content is already processed by the wizard window, no need to preprocess again
        processed_script = script_content
        
        # Create safe execution environment with necessary imports
        import scipy.stats
        from typing import Dict, Any, Optional, Tuple, List
        
        safe_globals = {
            '__builtins__': {
                '__import__': __import__,  # Allow imports for necessary modules
                'len': len,
                'min': min,
                'max': max,
                'abs': abs,
                'sum': sum,
                'float': float,
                'int': int,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'reversed': reversed,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'print': print,  # Allow print for debugging
            },
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'stats': scipy.stats,
            'pearsonr': scipy.stats.pearsonr,
            'spearmanr': scipy.stats.spearmanr,
            'kendalltau': scipy.stats.kendalltau,
            # Add typing support for scripts that use type hints
            'List': List,
            'Dict': Dict,
            'Tuple': Tuple,
            'Any': Any,
            'Optional': Optional,
        }
        
        # Create execution context
        local_vars = {
            'ref_data': ref_data,
            'test_data': test_data,
            'parameters': parameters,
            'params': parameters,  # Alias for convenience
            'x_data': None,
            'y_data': None,
            'plot_metadata': {}
        }
        
        # Execute the script
        try:
            exec(processed_script, safe_globals, local_vars)
            print(f"[PairAnalyzer][DEBUG] Custom plot script executed successfully")
        except TypeError as e:
            if "cannot unpack non-iterable NoneType object" in str(e):
                print(f"[PairAnalyzer] Custom plot script failed: A function call returned None when a tuple was expected")
                print(f"[PairAnalyzer] This usually happens when a nested function is not properly defined or imported")
                print(f"[PairAnalyzer] Check that all function calls in your script return the expected values")
                print(f"[PairAnalyzer] Error details: {e}")
                # Return empty results on error
                return ref_data, test_data, {}
            else:
                print(f"[PairAnalyzer] Custom plot script failed with TypeError: {e}")
                import traceback
                traceback.print_exc()
                return ref_data, test_data, {}
        except Exception as e:
            print(f"[PairAnalyzer] Custom plot script failed: {e}")
            import traceback
            traceback.print_exc()
            # Return empty results on error
            return ref_data, test_data, {}
        
        # Extract results with validation
        x_data = local_vars.get('x_data')
        y_data = local_vars.get('y_data')
        plot_metadata = local_vars.get('plot_metadata', {})
        
        print(f"[PairAnalyzer][DEBUG] plot_metadata from custom script: {plot_metadata}")
        
        # Validate results
        if x_data is None:
            print("[PairAnalyzer] Custom plot script returned None for x_data, using ref_data")
            x_data = ref_data
        if y_data is None:
            print("[PairAnalyzer] Custom plot script returned None for y_data, using test_data")
            y_data = test_data
        if plot_metadata is None:
            plot_metadata = {}
        
        return x_data, y_data, plot_metadata
    
    def _execute_custom_stats_script(self, script_content: str, x_data, y_data, ref_data, test_data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute custom stats script in a controlled environment.
        
        Args:
            script_content: Custom script code (already processed by wizard)
            x_data: X data for plotting
            y_data: Y data for plotting
            ref_data: Reference data
            test_data: Test data
            parameters: Method parameters
            
        Returns:
            Dictionary of statistical results
        """
        # Script content is already processed by the wizard window, no need to preprocess again
        processed_script = script_content
        
        # Create safe execution environment with necessary imports
        import scipy.stats
        from typing import Dict, Any, Optional, Tuple, List
        
        # Create safe execution environment
        safe_globals = {
            '__builtins__': {
                '__import__': __import__,  # Allow imports for necessary modules
                'len': len,
                'min': min,
                'max': max,
                'abs': abs,
                'sum': sum,
                'float': float,
                'int': int,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'reversed': reversed,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'print': print,  # Allow print for debugging
            },
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'stats': scipy.stats,
            'pearsonr': scipy.stats.pearsonr,
            'spearmanr': scipy.stats.spearmanr,
            'kendalltau': scipy.stats.kendalltau,
            'linregress': scipy.stats.linregress,
            # Add typing support for scripts that use type hints
            'List': List,
            'Dict': Dict,
            'Tuple': Tuple,
            'Any': Any,
            'Optional': Optional,
        }
        
        # Create execution context
        local_vars = {
            'x_data': x_data,
            'y_data': y_data,
            'ref_data': ref_data,
            'test_data': test_data,
            'parameters': parameters,
            'params': parameters,  # Alias for convenience
            'stats_results': {}
        }
        
        # Execute the script
        exec(processed_script, safe_globals, local_vars)
        
        # Extract results with validation
        stats_results = local_vars.get('stats_results', {})
        
        # Validate results
        if stats_results is None:
            print("[PairAnalyzer] Custom stats script returned None for stats_results, using empty dict")
            stats_results = {}
        
        return stats_results
    
    def _update_single_line_overlay(self, overlay: Overlay, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Update a single line overlay to span the new x and y ranges.
        
        Args:
            overlay: Line overlay object
            x_min: New minimum x value
            x_max: New maximum x value
            y_min: New minimum y value
            y_max: New maximum y value
        """
        try:
            overlay_data = overlay.data or {}
            
            print(f"[PairAnalyzer] DEBUG: _update_single_line_overlay called for '{overlay.id}' with x_range=[{x_min:.3f}, {x_max:.3f}], y_range=[{y_min:.3f}, {y_max:.3f}]")
            
            # For line overlays, update x coordinates while keeping y coordinates
            if 'x' in overlay_data and 'y' in overlay_data:
                old_x = overlay_data['x']
                old_y = overlay_data['y']
                
                print(f"[PairAnalyzer] DEBUG: Found x,y data - old_x={old_x}, old_y={old_y}")
                
                if len(old_x) >= 2 and len(old_y) >= 2:
                    # Continue with existing logic...
                    self._process_line_overlay_with_xy_data(overlay, old_x, old_y, x_min, x_max, y_min, y_max, overlay_data)
                else:
                    print(f"[PairAnalyzer] DEBUG: Insufficient data points - x length: {len(old_x)}, y length: {len(old_y)}")
            elif 'y' in overlay_data and (overlay.type == 'hline' or 'hline' in overlay.id.lower() or 'bias' in overlay.name.lower()):
                # Handle horizontal line overlays that only have 'y' data (constant y value)
                old_y = overlay_data['y']
                old_x = []  # No x data initially
                
                print(f"[PairAnalyzer] DEBUG: Found horizontal line with y data only - old_y={old_y}")
                
                # Process as horizontal line
                new_x = [x_min, x_max]
                new_y = [old_y, old_y]  # Same y value for both endpoints
                
                overlay.data = {
                    'x': new_x,
                    'y': new_y
                }
                
                print(f"[PairAnalyzer] DEBUG: Updated horizontal line overlay.data to: {overlay.data}")
                print(f"[PairAnalyzer] Updated horizontal line '{overlay.id}': x=[] -> [{x_min:.3f}, {x_max:.3f}], y={old_y} -> [{new_y[0]:.3f}, {new_y[1]:.3f}]")
            else:
                print(f"[PairAnalyzer] DEBUG: No suitable data found in overlay data: {list(overlay_data.keys())}")
                
        except Exception as e:
            print(f"[PairAnalyzer] DEBUG: Error updating line overlay {overlay.id}: {e}")
            print(f"[PairAnalyzer] Error updating line overlay {overlay.id}: {e}")
                
    def _process_line_overlay_with_xy_data(self, overlay, old_x, old_y, x_min, x_max, y_min, y_max, overlay_data):
        """Process line overlay that has both x and y data arrays."""
        try:
            overlay_name = overlay.name.lower()
            overlay_id = overlay.id.lower()
                    
            print(f"[PairAnalyzer] DEBUG: Checking overlay type - name='{overlay_name}', id='{overlay_id}'")
                    
            # Vertical lines for histograms (mean, median, etc.) - update y range but keep constant x value
            if 'mean' in overlay_name or 'median' in overlay_name or overlay_id in ['mean_line', 'median_line']:
                # For histogram vertical lines, keep x constant but update y to span histogram height
                if len(old_x) > 0:
                    x_value = old_x[0]  # Use the first x value as the constant
                    new_x = [x_value, x_value]  # Same x value for both endpoints
                    new_y = [0, y_max]  # Span from 0 to max histogram height
                else:
                    new_x = [0, 0]  # Fallback if no x data
                    new_y = [0, y_max]
                print(f"[PairAnalyzer] DEBUG: Histogram vertical line detected - updating to x=[{new_x[0]:.3f}, {new_x[1]:.3f}], y=[{new_y[0]:.3f}, {new_y[1]:.3f}]")
                print(f"[PairAnalyzer] Updated histogram vertical line '{overlay.id}': x=[{old_x[0]:.3f}, {old_x[1]:.3f}] -> [{new_x[0]:.3f}, {new_x[1]:.3f}], y=[{old_y[0]:.3f}, {old_y[1]:.3f}] -> [{new_y[0]:.3f}, {new_y[1]:.3f}]")
                    
            # Identity line (y = x) - update both x and y to span the full range
            elif 'identity' in overlay_name or 'y = x' in overlay_name or 'identity' in overlay_id:
                new_x = [x_min, x_max]
                new_y = [x_min, x_max]  # For identity line: y = x
                print(f"[PairAnalyzer] DEBUG: Identity line detected - updating to x=[{x_min:.3f}, {x_max:.3f}], y=[{x_min:.3f}, {x_max:.3f}]")
                print(f"[PairAnalyzer] DEBUG: Identity line - Detection criteria: 'identity' in overlay_name: {'identity' in overlay_name}, 'y = x' in overlay_name: {'y = x' in overlay_name}, 'identity' in overlay_id: {'identity' in overlay_id}")
                print(f"[PairAnalyzer] Updated identity line '{overlay.id}': x=[{old_x[0]:.3f}, {old_x[1]:.3f}] -> [{x_min:.3f}, {x_max:.3f}], y=[{old_y[0]:.3f}, {old_y[1]:.3f}] -> [{x_min:.3f}, {x_max:.3f}]")
                    
            # Horizontal line (hline) - update x range but keep constant y value
            elif overlay.type == 'hline' or 'hline' in overlay_id or 'bias' in overlay_name or 'bias' in overlay_id:
                new_x = [x_min, x_max]
                # For horizontal lines, keep the same y value across the entire x range
                if len(old_y) > 0:
                    y_value = old_y[0]  # Use the first y value as the constant
                    new_y = [y_value, y_value]  # Same y value for both endpoints
                elif 'y' in overlay_data:
                    # Handle case where overlay has 'y' key with constant value
                    y_value = overlay_data['y']
                    new_y = [y_value, y_value]
                else:
                    new_y = [0, 0]  # Fallback if no y data
                print(f"[PairAnalyzer] DEBUG: Horizontal line detected - updating to x=[{x_min:.3f}, {x_max:.3f}], y=[{new_y[0]:.3f}, {new_y[1]:.3f}]")
                print(f"[PairAnalyzer] DEBUG: Horizontal line - Detection criteria: overlay.type=='hline': {overlay.type == 'hline'}, 'hline' in overlay_id: {'hline' in overlay_id}, 'bias' in overlay_name: {'bias' in overlay_name}, 'bias' in overlay_id: {'bias' in overlay_id}")
                if len(old_y) > 0:
                    print(f"[PairAnalyzer] Updated horizontal line '{overlay.id}': x=[{old_x[0]:.3f}, {old_x[1]:.3f}] -> [{x_min:.3f}, {x_max:.3f}], y=[{old_y[0]:.3f}, {old_y[1]:.3f}] -> [{new_y[0]:.3f}, {new_y[1]:.3f}]")
                else:
                    print(f"[PairAnalyzer] Updated horizontal line '{overlay.id}': x=[] -> [{x_min:.3f}, {x_max:.3f}], y=[] -> [{new_y[0]:.3f}, {new_y[1]:.3f}]")
                    
            # Regression line - update x range but calculate new y values based on regression
            elif 'regression' in overlay_name or 'regression' in overlay_id:
                new_x = [x_min, x_max]
                # Keep the slope calculation but update endpoints
                if len(old_x) >= 2 and len(old_y) >= 2:
                    # Calculate slope from existing points
                    if old_x[1] != old_x[0]:
                        slope = (old_y[1] - old_y[0]) / (old_x[1] - old_x[0])
                        intercept = old_y[0] - slope * old_x[0]
                        new_y = [slope * x_min + intercept, slope * x_max + intercept]
                        print(f"[PairAnalyzer] DEBUG: Regression line - slope={slope:.3f}, intercept={intercept:.3f}, new_y=[{new_y[0]:.3f}, {new_y[1]:.3f}]")
                    else:
                        new_y = [old_y[0], old_y[1]]  # Keep same y if vertical line
                        print(f"[PairAnalyzer] DEBUG: Regression line - vertical line, keeping y={new_y}")
                else:
                    new_y = [old_y[0], old_y[1]] if len(old_y) >= 2 else [0, 0]
                    print(f"[PairAnalyzer] DEBUG: Regression line - insufficient data, using y={new_y}")
                print(f"[PairAnalyzer] Updated regression line '{overlay.id}': x=[{old_x[0]:.3f}, {old_x[1]:.3f}] -> [{x_min:.3f}, {x_max:.3f}], y=[{old_y[0]:.3f}, {old_y[1]:.3f}] -> [{new_y[0]:.3f}, {new_y[1]:.3f}]")
                    
            # Confidence interval lines - update x range but keep y values
            elif 'confidence' in overlay_name or 'confidence' in overlay_id or 'ci' in overlay_id:
                new_x = [x_min, x_max]
                new_y = [old_y[0], old_y[1]]  # Keep the same y values
                print(f"[PairAnalyzer] DEBUG: Confidence interval line - old_y={old_y}, keeping y={new_y}")
                print(f"[PairAnalyzer] Updated confidence interval line '{overlay.id}': x=[{old_x[0]:.3f}, {old_x[1]:.3f}] -> [{x_min:.3f}, {x_max:.3f}]")
                    
            # Other lines (bias, LoA, etc.) - keep the y values but update x to span the new range
            else:
                new_x = [x_min, x_max]
                new_y = [old_y[0], old_y[1]]  # Keep the same y values
                print(f"[PairAnalyzer] DEBUG: Other line type - overlay falls into 'else' case")
                print(f"[PairAnalyzer] DEBUG: Other line type - old_y={old_y}, keeping y={new_y}")
                print(f"[PairAnalyzer] Updated line overlay '{overlay.id}': x=[{old_x[0]:.3f}, {old_x[1]:.3f}] -> [{x_min:.3f}, {x_max:.3f}]")
                    
            # Apply the update
            overlay.data = {
                'x': new_x,
                'y': new_y
            }
            print(f"[PairAnalyzer] DEBUG: Updated overlay.data to: {overlay.data}")
        except Exception as e:
            print(f"[PairAnalyzer] DEBUG: Error updating line overlay {overlay.id}: {e}")
            print(f"[PairAnalyzer] Error updating line overlay {overlay.id}: {e}")
    
    def _update_single_fill_overlay(self, overlay: Overlay, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Update a single fill overlay (confidence intervals) to span the new x and y ranges.
        
        Args:
            overlay: Fill overlay object
            x_min: New minimum x value
            x_max: New maximum x value
            y_min: New minimum y value
            y_max: New maximum y value
        """
        try:
            overlay_data = overlay.data or {}
            
            print(f"[PairAnalyzer] DEBUG: _update_single_fill_overlay called for '{overlay.id}' with x_range=[{x_min:.3f}, {x_max:.3f}]")
            
            # For fill overlays, update x coordinates while interpolating y coordinates
            if 'x' in overlay_data and 'y_lower' in overlay_data and 'y_upper' in overlay_data:
                old_x = overlay_data['x']
                old_y_lower = overlay_data['y_lower']
                old_y_upper = overlay_data['y_upper']
                
                print(f"[PairAnalyzer] DEBUG: Found fill data - old_x length: {len(old_x)}, old_y_lower length: {len(old_y_lower)}, old_y_upper length: {len(old_y_upper)}")
                
                if len(old_x) >= 2 and len(old_y_lower) >= 2 and len(old_y_upper) >= 2:
                    # Check if this is a histogram fill overlay (std dev bands)
                    if 'std' in overlay.id.lower() or 'std' in overlay.name.lower():
                        # For histogram std dev bands, use the x values as-is and span full height
                        new_x = old_x.copy()  # Keep original x bounds
                        new_y_lower = [0] * len(old_x)  # Bottom of histogram
                        new_y_upper = [y_max] * len(old_x)  # Top of histogram
                        
                        print(f"[PairAnalyzer] DEBUG: Histogram std dev band detected - using x bounds as-is, spanning full height")
                        print(f"[PairAnalyzer] DEBUG: Histogram std dev - x={new_x}, y_lower={new_y_lower}, y_upper={new_y_upper}")
                    else:
                        # Regular interpolation for non-histogram fills
                        import numpy as np
                        
                        # Convert to numpy arrays for interpolation
                        old_x_arr = np.array(old_x)
                        old_y_lower_arr = np.array(old_y_lower)
                        old_y_upper_arr = np.array(old_y_upper)
                        
                        # Create new x values spanning the data range
                        new_x = [x_min, x_max]
                        
                        # Interpolate y values for the new x range
                        new_y_lower = np.interp(new_x, old_x_arr, old_y_lower_arr).tolist()
                        new_y_upper = np.interp(new_x, old_x_arr, old_y_upper_arr).tolist()
                        
                        print(f"[PairAnalyzer] DEBUG: Regular interpolation for fill overlay")
                        print(f"[PairAnalyzer] DEBUG: Interpolated y_lower: {old_y_lower[0]:.3f}, {old_y_lower[-1]:.3f} -> {new_y_lower[0]:.3f}, {new_y_lower[-1]:.3f}")
                        print(f"[PairAnalyzer] DEBUG: Interpolated y_upper: {old_y_upper[0]:.3f}, {old_y_upper[-1]:.3f} -> {new_y_upper[0]:.3f}, {new_y_upper[-1]:.3f}")
                    
                    overlay.data = {
                        'x': new_x,
                        'y_lower': new_y_lower,
                        'y_upper': new_y_upper
                    }
                    
                    print(f"[PairAnalyzer] DEBUG: Updated fill overlay.data to: {overlay.data}")
                    print(f"[PairAnalyzer] Updated fill overlay '{overlay.id}': x=[{old_x[0]:.3f}, {old_x[-1]:.3f}] -> [{new_x[0]:.3f}, {new_x[-1]:.3f}]")
                else:
                    print(f"[PairAnalyzer] DEBUG: Insufficient fill data points - x length: {len(old_x)}, y_lower length: {len(old_y_lower)}, y_upper length: {len(old_y_upper)}")
            else:
                print(f"[PairAnalyzer] DEBUG: No fill data found in overlay data: {list(overlay_data.keys())}")
            
        except Exception as e:
            print(f"[PairAnalyzer] DEBUG: Error updating fill overlay {overlay.id}: {e}")
            print(f"[PairAnalyzer] Error updating fill overlay {overlay.id}: {e}")
    
    def _update_single_vline_overlay(self, overlay: Overlay, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Update a single vertical line overlay to span the new y range.
        
        Args:
            overlay: Vline overlay object
            x_min: New minimum x value (not used for vlines)
            x_max: New maximum x value (not used for vlines)
            y_min: New minimum y value
            y_max: New maximum y value
        """
        try:
            overlay_data = overlay.data or {}
            
            print(f"[PairAnalyzer] DEBUG: _update_single_vline_overlay called for '{overlay.id}' with y_range=[{y_min:.3f}, {y_max:.3f}]")
            
            # For vline overlays, keep x coordinates constant but update y to span the full range
            if 'x' in overlay_data:
                x_values = overlay_data['x']
                if not isinstance(x_values, list):
                    x_values = [x_values]
                
                print(f"[PairAnalyzer] DEBUG: Found x values for vline: {x_values}")
                
                # Keep the x values constant, update y to span full range
                new_y = [y_min, y_max]
                
                # Update overlay data to include y coordinates for proper line rendering
                overlay.data = {
                    'x': x_values,
                    'y': new_y  # Will be used for each vertical line
                }
                
                print(f"[PairAnalyzer] DEBUG: Updated vline overlay.data to: {overlay.data}")
                print(f"[PairAnalyzer] Updated vline overlay '{overlay.id}': x={x_values}, y=[{y_min:.3f}, {y_max:.3f}]")
            else:
                print(f"[PairAnalyzer] DEBUG: No x values found in vline overlay data: {list(overlay_data.keys())}")
                
        except Exception as e:
            print(f"[PairAnalyzer] DEBUG: Error updating vline overlay {overlay.id}: {e}")
            print(f"[PairAnalyzer] Error updating vline overlay {overlay.id}: {e}") 