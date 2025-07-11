import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import warnings
import traceback
import sys
from io import StringIO

class AnalysisSession:
    """Simple wrapper for analysis results to maintain compatibility"""
    def __init__(self, analysis_results: Dict[str, Any]):
        self.analysis_results = analysis_results
        self.session_id = f"session_{int(time.time())}"
        self.created_at = datetime.now()

class PairAnalyzer:
    """
    Simplified service class that analyzes individual pairs and returns direct statistics.
    
    Key simplifications:
    - No AnalysisSession wrapper complexity
    - Direct analyze_pair(pair) -> statistics_dict interface  
    - Caching at pair level instead of session level
    - Simple, focused responsibility
    """
    
    def __init__(self, comparison_registry=None):
        self.comparison_registry = comparison_registry
        
        # Simple pair-level cache - keyed by (pair_id, method_name, method_params_hash)
        self.pair_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Configuration
        self.default_method = "Correlation Analysis"
        
        # For compatibility with existing code
        self.active_sessions = {}
        
        print("[PairAnalyzer] Initialized with simplified architecture")
    
    def analyze(self, comparison_config: Dict[str, Any], pair_manager, script_config: Dict[str, Any] = None) -> AnalysisSession:
        """
        Analyze all pairs in the pair_manager using the specified comparison method.
        
        Args:
            comparison_config: Configuration with method_name, method_params, etc.
            pair_manager: PairManager containing Pair objects with aligned data
            script_config: Optional script configuration (for compatibility)
            
        Returns:
            AnalysisSession with analysis_results containing statistics for each pair
        """
        try:
            print(f"[PairAnalyzer] Starting analysis with config: {comparison_config}")
            
            # Extract method info from config
            method_name = comparison_config.get('method_name', self.default_method)
            method_params = comparison_config.get('method_params', {})
            
            # Get all pairs from pair_manager
            pairs = pair_manager.get_pairs_in_order() if hasattr(pair_manager, 'get_pairs_in_order') else []
            
            if not pairs:
                print("[PairAnalyzer] No pairs found in pair_manager")
                return AnalysisSession({'error': 'No pairs available for analysis'})
            
            print(f"[PairAnalyzer] Analyzing {len(pairs)} pairs with method '{method_name}'")
            
            # Analyze each pair
            analysis_results = {}
            for pair in pairs:
                pair_name = getattr(pair, 'name', f'pair_{pair.pair_id}')
                
                # Analyze individual pair
                pair_stats = self.analyze_pair(pair, method_name, method_params)
                
                # Format results for compatibility with existing code
                analysis_results[pair_name] = {
                    'statistics': pair_stats,
                    'aligned_data': self._get_aligned_data_from_pair(pair),
                    'pair_id': pair.pair_id,
                    'method': method_name,
                    'method_params': method_params
                }
                
                r_val = pair_stats.get('r', np.nan)
                r_str = f"{r_val:.3f}" if not np.isnan(r_val) else "N/A"
                print(f"[PairAnalyzer] Analyzed pair '{pair_name}': r={r_str}")
            
            # Create analysis session
            session = AnalysisSession(analysis_results)
            self.active_sessions[session.session_id] = session
            
            print(f"[PairAnalyzer] Analysis complete for {len(analysis_results)} pairs")
            return session
            
        except Exception as e:
            print(f"[PairAnalyzer] Error in analyze: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisSession({'error': str(e)})
    
    def update_pair_visibility(self, session_id: str, pair_name: str, visible: bool):
        """Update pair visibility in analysis session (for compatibility)"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if pair_name in session.analysis_results:
                    session.analysis_results[pair_name]['visible'] = visible
                    print(f"[PairAnalyzer] Updated visibility for pair '{pair_name}': {visible}")
        except Exception as e:
            print(f"[PairAnalyzer] Error updating pair visibility: {e}")
    
    def _get_aligned_data_from_pair(self, pair) -> Dict[str, Any]:
        """Get aligned data from pair for compatibility with existing code"""
        try:
            if hasattr(pair, 'aligned_ref_data') and hasattr(pair, 'aligned_test_data'):
                return {
                    'ref_data': pair.aligned_ref_data,
                    'test_data': pair.aligned_test_data,
                    'n_valid': len(pair.aligned_ref_data) if pair.aligned_ref_data is not None else 0,
                    'alignment_method': getattr(pair, 'alignment_method', 'unknown')
                }
            elif hasattr(pair, 'aligned_data') and pair.aligned_data:
                return pair.aligned_data
            else:
                return {'error': 'No aligned data found'}
        except Exception as e:
            print(f"[PairAnalyzer] Error getting aligned data: {e}")
            return {'error': str(e)}

    def analyze_pair(self, pair, method_name: Optional[str] = None, method_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a single pair and return direct statistics.
        
        Args:
            pair: Pair object with aligned data
            method_name: Comparison method name (defaults to correlation)
            method_params: Method parameters
            
        Returns:
            Dict containing statistics like {'r': 0.85, 'rms': 0.12, 'n': 1000}
        """
        try:
            # Use defaults if not provided
            method_name = method_name or self.default_method
            method_params = method_params or {}
            
            # Check cache first
            cache_key = self._create_pair_cache_key(pair, method_name, method_params)
            if cache_key in self.pair_cache:
                self.cache_stats['hits'] += 1
                return self.pair_cache[cache_key]
            
            self.cache_stats['misses'] += 1
            
            # Extract data from pair
            pair_data = self._extract_pair_data(pair)
            if not pair_data:
                return {'error': 'No valid data in pair', 'r': np.nan, 'rms': np.nan, 'n': 0}
            
            # Get comparison method
            method_instance = self._get_comparison_method(method_name, method_params)
            if not method_instance:
                return {'error': 'Could not create comparison method', 'r': np.nan, 'rms': np.nan, 'n': 0}
            
            # Calculate statistics
            ref_data = np.array(pair_data['ref_data'])
            test_data = np.array(pair_data['test_data'])
            
            start_time = time.time()
            statistics = method_instance.calculate_stats(ref_data, test_data)
            computation_time = time.time() - start_time
            
            # Add metadata
            statistics.update({
                'n': len(ref_data),
                'computation_time': computation_time,
                'method': method_name,
                'pair_id': pair.pair_id,
                'pair_name': pair.name
            })
            
            # Cache results
            self.pair_cache[cache_key] = statistics
            
            return statistics
            
        except Exception as e:
            print(f"[PairAnalyzer] Error analyzing pair {getattr(pair, 'name', 'unknown')}: {e}")
            return {'error': str(e), 'r': np.nan, 'rms': np.nan, 'n': 0}
    
    def analyze_pairs(self, pairs: List, method_name: Optional[str] = None, method_params: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple pairs and return statistics for each.
        
        Args:
            pairs: List of Pair objects
            method_name: Comparison method name
            method_params: Method parameters
            
        Returns:
            Dict mapping pair_id -> statistics_dict
        """
        results = {}
        
        for pair in pairs:
            if hasattr(pair, 'show') and not pair.show:
                continue  # Skip hidden pairs
                
            pair_stats = self.analyze_pair(pair, method_name, method_params)
            results[pair.pair_id] = pair_stats
        
        return results
    
    def get_combined_statistics(self, pairs: List, method_name: Optional[str] = None, method_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get combined statistics across all pairs.
        
        Args:
            pairs: List of Pair objects
            method_name: Comparison method name
            method_params: Method parameters
            
        Returns:
            Dict with combined statistics
        """
        try:
            # Collect all data from pairs
            all_ref_data = []
            all_test_data = []
            valid_pairs = 0
            
            for pair in pairs:
                if hasattr(pair, 'show') and not pair.show:
                    continue
                    
                pair_data = self._extract_pair_data(pair)
                if pair_data:
                    all_ref_data.extend(pair_data['ref_data'])
                    all_test_data.extend(pair_data['test_data'])
                    valid_pairs += 1
            
            if not all_ref_data or not all_test_data:
                return {'error': 'No valid data', 'r': np.nan, 'rms': np.nan, 'n': 0}
            
            # Get comparison method
            method_name = method_name or self.default_method
            method_params = method_params or {}
            method_instance = self._get_comparison_method(method_name, method_params)
            
            if not method_instance:
                return {'error': 'Could not create comparison method', 'r': np.nan, 'rms': np.nan, 'n': 0}
            
            # Calculate combined statistics
            ref_data = np.array(all_ref_data)
            test_data = np.array(all_test_data)
            
            start_time = time.time()
            statistics = method_instance.calculate_stats(ref_data, test_data)
            computation_time = time.time() - start_time
            
            # Add metadata
            statistics.update({
                'n': len(ref_data),
                'n_pairs': valid_pairs,
                'computation_time': computation_time,
                'method': method_name,
                'combined': True
            })
            
            return statistics
            
        except Exception as e:
            print(f"[PairAnalyzer] Error in combined analysis: {e}")
            return {'error': str(e), 'r': np.nan, 'rms': np.nan, 'n': 0}
    
    def invalidate_cache(self, pair_id: str = None):
        """Invalidate cache for specific pair or all pairs"""
        if pair_id:
            # Remove cache entries for specific pair
            keys_to_remove = [k for k in self.pair_cache.keys() if pair_id in str(k)]
            for key in keys_to_remove:
                del self.pair_cache[key]
            print(f"[PairAnalyzer] Invalidated cache for pair {pair_id}")
        else:
            # Clear all cache
            self.pair_cache.clear()
            print("[PairAnalyzer] Cleared all cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'cached_pairs': len(self.pair_cache)
        }
    
    def _extract_pair_data(self, pair) -> Optional[Dict[str, Any]]:
        """Extract aligned data from a Pair object"""
        try:
            # Check if pair has aligned data stored directly
            if hasattr(pair, 'aligned_ref_data') and hasattr(pair, 'aligned_test_data'):
                ref_data = pair.aligned_ref_data
                test_data = pair.aligned_test_data
                
                if ref_data is not None and test_data is not None:
                    # Filter valid data
                    valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                    ref_clean = ref_data[valid_mask]
                    test_clean = test_data[valid_mask]
                    
                    if len(ref_clean) > 0:
                        return {
                            'ref_data': ref_clean,
                            'test_data': test_clean,
                            'n_valid': len(ref_clean),
                            'n_total': len(ref_data),
                            'valid_ratio': len(ref_clean) / len(ref_data)
                        }
            
            # Fallback: check if pair has legacy aligned_data dict
            elif hasattr(pair, 'aligned_data') and pair.aligned_data:
                ref_data = pair.aligned_data.get('ref_data')
                test_data = pair.aligned_data.get('test_data')
                
                if ref_data is not None and test_data is not None:
                    valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                    ref_clean = ref_data[valid_mask]
                    test_clean = test_data[valid_mask]
                    
                    if len(ref_clean) > 0:
                        return {
                            'ref_data': ref_clean,
                            'test_data': test_clean,
                            'n_valid': len(ref_clean),
                            'n_total': len(ref_data),
                            'valid_ratio': len(ref_clean) / len(ref_data)
                        }
            
            print(f"[PairAnalyzer] No valid aligned data found in pair {getattr(pair, 'name', 'unknown')}")
            return None
            
        except Exception as e:
            print(f"[PairAnalyzer] Error extracting data from pair: {e}")
            return None
    
    def _get_comparison_method(self, method_name: str, method_params: Dict[str, Any]):
        """Get comparison method instance"""
        try:
            if not self.comparison_registry:
                # Fallback: try to import and use registry directly
                from comparison.comparison_registry import ComparisonRegistry
                registry = ComparisonRegistry()
                method_class = registry.get(method_name)
                if method_class:
                    return method_class(**method_params)
                else:
                    return None
            else:
                method_class = self.comparison_registry.get(method_name)
                if method_class:
                    return method_class(**method_params)
                else:
                    return None
                
        except Exception as e:
            print(f"[PairAnalyzer] Error creating comparison method '{method_name}': {e}")
            return None
    
    def _create_pair_cache_key(self, pair, method_name: str, method_params: Dict[str, Any]) -> str:
        """Create cache key for a pair analysis"""
        import hashlib
        
        # Create hash of method parameters
        params_str = str(sorted(method_params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        # Include pair modification time if available 
        pair_mod_time = getattr(pair, 'modified_at', datetime.now()).isoformat()
        
        return f"{pair.pair_id}_{method_name}_{params_hash}_{pair_mod_time}" 