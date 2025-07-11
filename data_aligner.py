import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
from pair import AlignmentConfig, AlignmentMethod
from channel import Channel
import warnings


@dataclass
class AlignmentResult:
    """Container for alignment results - just the aligned data"""
    ref_data: np.ndarray
    test_data: np.ndarray

    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.success and not self.error_message:
            self.error_message = "Unknown alignment error"


class DataAligner:
    """
    Service for aligning data from two channels according to alignment configuration.
    
    Supports both index-based and time-based alignment methods with various
    options for handling different data scenarios.
    """
    
    def __init__(self):
        self._alignment_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def align_from_wizard_params(self, ref_channel: Channel, test_channel: Channel, 
                                alignment_params: Dict[str, Any]) -> AlignmentResult:
        """
        Align channels using parameters from the comparison wizard.
        
        Args:
            ref_channel: Reference channel object
            test_channel: Test channel object
            alignment_params: Dictionary with alignment parameters from wizard
                Expected keys:
                - mode: 'Index-Based' or 'Time-Based'
                - start_index, end_index: for index-based alignment
                - start_time, end_time: for time-based alignment
                - offset: time offset (always 0 for index-based)
                - interpolation: interpolation method for time-based
                - resolution: resolution for time-based alignment
                
        Returns:
            AlignmentResult with aligned data or error information
        """
        try:
            # Convert wizard parameters to AlignmentConfig
            alignment_config = self._convert_wizard_params_to_config(alignment_params)
            
            # Use existing alignment method
            return self.align_channels(ref_channel, test_channel, alignment_config)
            
        except Exception as e:
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message=f"Error converting wizard parameters: {e}"
            )
    
    def _convert_wizard_params_to_config(self, alignment_params: Dict[str, Any]) -> 'AlignmentConfig':
        """
        Convert comparison wizard alignment parameters to AlignmentConfig.
        
        Args:
            alignment_params: Dictionary with alignment parameters from wizard
            
        Returns:
            AlignmentConfig object
        """
        from pair import AlignmentConfig, AlignmentMethod
        
        mode = alignment_params.get('mode', 'Index-Based')
        
        if mode == 'Index-Based':
            # Index-based alignment
            start_index = alignment_params.get('start_index', 0)
            end_index = alignment_params.get('end_index', 100)
            offset = alignment_params.get('offset', 0)
            
            return AlignmentConfig(
                method=AlignmentMethod.INDEX,
                mode='custom',  # Always use custom mode for wizard
                offset=offset,
                start_index=start_index,
                end_index=end_index,
                start_time=None,
                end_time=None,
                interpolation=None,
                round_to=None
            )
            
        elif mode == 'Time-Based':
            # Time-based alignment
            start_time = alignment_params.get('start_time', 0.0)
            end_time = alignment_params.get('end_time', 10.0)
            offset = alignment_params.get('offset', 0.0)
            interpolation = alignment_params.get('interpolation', 'nearest')
            resolution = alignment_params.get('resolution', 0.1)
            
            # Store resolution for time grid creation
            # round_to is now used for display precision only
            round_to = None
            if resolution is not None and resolution > 0:
                # Calculate decimal places needed for the resolution
                if resolution < 1:
                    round_to = abs(int(np.log10(resolution)))
                else:
                    round_to = 0
            
            return AlignmentConfig(
                method=AlignmentMethod.TIME,
                mode='custom',  # Always use custom mode as the wizard provides specific ranges
                offset=offset,
                start_index=None,
                end_index=None,
                start_time=start_time,
                end_time=end_time,
                interpolation=interpolation,
                round_to=round_to,
                resolution=resolution  # Add resolution to AlignmentConfig
            )
            
        else:
            raise ValueError(f"Unknown alignment mode: {mode}")
    
    def align_channels(self, ref_channel: Channel, test_channel: Channel, 
                      alignment_config: AlignmentConfig) -> AlignmentResult:
        """
        Align two channels according to the specified configuration.
        
        Args:
            ref_channel: Reference channel object
            test_channel: Test channel object
            alignment_config: Alignment configuration
            
        Returns:
            AlignmentResult with aligned data or error information
        """
        # Validate inputs
        validation_result = self._validate_inputs(ref_channel, test_channel, alignment_config)
        if not validation_result['valid']:
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message=validation_result['error']
            )
        
        # Check cache
        cache_key = self._generate_cache_key(ref_channel, test_channel, alignment_config)
        if cache_key in self._alignment_cache:
            self._cache_hits += 1
            return self._alignment_cache[cache_key]
        
        self._cache_misses += 1
        
        # Perform alignment based on method
        if alignment_config.method == AlignmentMethod.INDEX:
            result = self._align_by_index(ref_channel, test_channel, alignment_config)
        elif alignment_config.method == AlignmentMethod.TIME:
            result = self._align_by_time(ref_channel, test_channel, alignment_config)
        else:
            result = AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message=f"Unknown alignment method: {alignment_config.method}"
            )
        
        # Cache result
        self._alignment_cache[cache_key] = result
        return result
    
    def _validate_inputs(self, ref_channel: Channel, test_channel: Channel, 
                        alignment_config: AlignmentConfig) -> Dict[str, Any]:
        """Validate input channels and alignment configuration"""
        # Check channels exist and have data
        if not ref_channel or not test_channel:
            return {'valid': False, 'error': 'Both channels must be provided'}
        
        if ref_channel.ydata is None or test_channel.ydata is None:
            return {'valid': False, 'error': 'Both channels must have data'}
        
        if len(ref_channel.ydata) == 0 or len(test_channel.ydata) == 0:
            return {'valid': False, 'error': 'Both channels must have non-empty data'}
        
        # Check for self-comparison
        if (ref_channel.channel_id == test_channel.channel_id and 
            ref_channel.file_id == test_channel.file_id):
            return {'valid': False, 'error': 'Cannot align channel to itself'}
        
        # Validate alignment config
        try:
            if alignment_config.method == AlignmentMethod.INDEX:
                if alignment_config.mode == 'custom':
                    if (alignment_config.start_index is None or 
                        alignment_config.end_index is None):
                        return {'valid': False, 'error': 'Custom index mode requires start and end indices'}
                    if alignment_config.start_index >= alignment_config.end_index:
                        return {'valid': False, 'error': 'Start index must be less than end index'}
            elif alignment_config.method == AlignmentMethod.TIME:
                if ref_channel.xdata is None or test_channel.xdata is None:
                    return {'valid': False, 'error': 'Time-based alignment requires x-data for both channels'}
                if alignment_config.mode == 'custom':
                    if (alignment_config.start_time is None or 
                        alignment_config.end_time is None):
                        return {'valid': False, 'error': 'Custom time mode requires start and end times'}
                    if alignment_config.start_time >= alignment_config.end_time:
                        return {'valid': False, 'error': 'Start time must be less than end time'}
        except Exception as e:
            return {'valid': False, 'error': f'Invalid alignment configuration: {e}'}
        
        return {'valid': True}
    
    def _align_by_index(self, ref_channel: Channel, test_channel: Channel, 
                       alignment_config: AlignmentConfig) -> AlignmentResult:
        """Align data using index-based method"""
        ref_data = ref_channel.ydata
        test_data = test_channel.ydata
        
        # Ensure data is not None
        if ref_data is None or test_data is None:
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message="Channel data is None"
            )
        
        # Apply offset
        offset = int(alignment_config.offset)
        
        if alignment_config.mode == 'truncate':
            # Truncate both to the length of the shorter array
            min_length = min(len(ref_data), len(test_data))
            aligned_ref = ref_data[:min_length]
            aligned_test = test_data[:min_length]
            
            # Apply offset by shifting one array
            if offset > 0:
                if offset < len(aligned_ref):
                    aligned_ref = aligned_ref[offset:]
                    aligned_test = aligned_test[:-offset] if offset < len(aligned_test) else np.array([])
                else:
                    aligned_ref = np.array([])
                    aligned_test = np.array([])
            elif offset < 0:
                offset = abs(offset)
                if offset < len(aligned_test):
                    aligned_test = aligned_test[offset:]
                    aligned_ref = aligned_ref[:-offset] if offset < len(aligned_ref) else np.array([])
                else:
                    aligned_ref = np.array([])
                    aligned_test = np.array([])
            
        elif alignment_config.mode == 'custom':
            start_idx = alignment_config.start_index
            end_idx = alignment_config.end_index
            
            # Ensure indices are not None
            if start_idx is None or end_idx is None:
                return AlignmentResult(
                    ref_data=np.array([]),
                    test_data=np.array([]),
                    success=False,
                    error_message="Start or end index is None"
                )
            
            # Extract data within the specified range
            ref_start = max(0, start_idx)
            ref_end = min(len(ref_data), end_idx)
            test_start = max(0, start_idx + offset)
            test_end = min(len(test_data), end_idx + offset)
            
            aligned_ref = ref_data[ref_start:ref_end]
            aligned_test = test_data[test_start:test_end]
            
            # Ensure both arrays have the same length
            min_length = min(len(aligned_ref), len(aligned_test))
            aligned_ref = aligned_ref[:min_length]
            aligned_test = aligned_test[:min_length]
        
        else:
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message=f"Unknown index alignment mode: {alignment_config.mode}"
            )
        
        # Check if we have valid aligned data
        if len(aligned_ref) == 0 or len(aligned_test) == 0:
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message="No overlapping data after alignment"
            )
        
        alignment_info = {
            'method': 'index',
            'mode': alignment_config.mode,
            'offset': offset,
            'original_ref_length': len(ref_data),
            'original_test_length': len(test_data),
            'aligned_length': len(aligned_ref),
            'start_index': alignment_config.start_index if alignment_config.mode == 'custom' else 0,
            'end_index': alignment_config.end_index if alignment_config.mode == 'custom' else min(len(ref_data), len(test_data))
        }
        
        return AlignmentResult(
            ref_data=aligned_ref,
            test_data=aligned_test,
            success=True, # Index alignment is always successful
            error_message=None
        )
    
    def _aggregate_duplicate_times(self, x: np.ndarray, y: np.ndarray, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate y values for x values that round to the same time using mean"""
        if len(x) == 0 or len(y) == 0:
            return x, y
            
        # Round x values to resolution
        rounded_x = np.round(x / resolution) * resolution
        
        # Find unique rounded x values
        unique_x = np.unique(rounded_x)
        aggregated_y = []
        
        for ux in unique_x:
            mask = rounded_x == ux
            if np.sum(mask) > 1:
                # Multiple values - take mean
                aggregated_y.append(np.mean(y[mask]))
            else:
                # Single value
                aggregated_y.append(y[mask][0])
        
        return unique_x, np.array(aggregated_y)
    
    def _align_by_time(self, ref_channel: Channel, test_channel: Channel, 
                      alignment_config: AlignmentConfig) -> AlignmentResult:
        """Align data using time-based method with optional uniform time grid"""
        ref_xdata = ref_channel.xdata
        ref_ydata = ref_channel.ydata
        test_xdata = test_channel.xdata
        test_ydata = test_channel.ydata
        
        # Ensure data is not None
        if (ref_xdata is None or ref_ydata is None or 
            test_xdata is None or test_ydata is None):
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message="Channel xdata or ydata is None"
            )
        
        offset = alignment_config.offset
        
        if alignment_config.mode == 'overlap':
            # Find overlapping time range
            ref_start = ref_xdata[0]
            ref_end = ref_xdata[-1]
            test_start = test_xdata[0] + offset
            test_end = test_xdata[-1] + offset
            
            overlap_start = max(ref_start, test_start)
            overlap_end = min(ref_end, test_end)
            
            if overlap_start >= overlap_end:
                return AlignmentResult(
                    ref_data=np.array([]),
                    test_data=np.array([]),
                    success=False,
                    error_message="No overlapping time range between channels"
                )
            
            # Extract data within overlap region
            ref_mask = (ref_xdata >= overlap_start) & (ref_xdata <= overlap_end)
            test_mask = (test_xdata >= overlap_start - offset) & (test_xdata <= overlap_end - offset)
            
            aligned_ref_x = ref_xdata[ref_mask]
            aligned_ref_y = ref_ydata[ref_mask]
            aligned_test_x = test_xdata[test_mask]
            aligned_test_y = test_ydata[test_mask]
            
        elif alignment_config.mode == 'custom':
            start_time = alignment_config.start_time
            end_time = alignment_config.end_time
            
            # Ensure times are not None
            if start_time is None or end_time is None:
                return AlignmentResult(
                    ref_data=np.array([]),
                    test_data=np.array([]),
                    success=False,
                    error_message="Start or end time is None"
                )
            
            # Extract data within specified time range
            ref_mask = (ref_xdata >= start_time) & (ref_xdata <= end_time)
            test_mask = (test_xdata >= start_time - offset) & (test_xdata <= end_time - offset)
            
            aligned_ref_x = ref_xdata[ref_mask]
            aligned_ref_y = ref_ydata[ref_mask]
            aligned_test_x = test_xdata[test_mask]
            aligned_test_y = test_ydata[test_mask]
            
        else:
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message=f"Unknown time alignment mode: {alignment_config.mode}"
            )
        
        # Check if we have valid data
        if len(aligned_ref_y) == 0 or len(aligned_test_y) == 0:
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message="No data within specified time range"
            )
        
        # Aggregate duplicate time points if resolution is specified
        if alignment_config.resolution is not None and alignment_config.resolution > 0:
            # Aggregate reference data
            aligned_ref_x, aligned_ref_y = self._aggregate_duplicate_times(aligned_ref_x, aligned_ref_y, alignment_config.resolution)
            
            # Aggregate test data
            aligned_test_x, aligned_test_y = self._aggregate_duplicate_times(aligned_test_x, aligned_test_y, alignment_config.resolution)
            
            # Create uniform time grid
            time_start = max(aligned_ref_x[0], aligned_test_x[0])
            time_end = min(aligned_ref_x[-1], aligned_test_x[-1])
            
            # Ensure we have a valid time range
            if time_start >= time_end:
                return AlignmentResult(
                    ref_data=np.array([]),
                    test_data=np.array([]),
                    success=False,
                    error_message="No overlapping time range for uniform grid"
                )
            
            # Create uniform time points
            uniform_time_grid = np.arange(time_start, time_end + alignment_config.resolution, 
                                        alignment_config.resolution)
            
            # Interpolate both channels to uniform grid
            interpolation_method = alignment_config.interpolation or 'linear'
            aligned_ref_y = self._interpolate_to_grid(aligned_ref_x, aligned_ref_y, 
                                                    uniform_time_grid, interpolation_method)
            aligned_test_y = self._interpolate_to_grid(aligned_test_x, aligned_test_y, 
                                                     uniform_time_grid, interpolation_method)
            
            # Use uniform grid as the time axis
            aligned_ref_x = uniform_time_grid
            aligned_test_x = uniform_time_grid
            
        else:
            # Use original interpolation logic for non-uniform grids
            if alignment_config.interpolation and len(aligned_ref_x) != len(aligned_test_x):
                aligned_ref_y, aligned_test_y = self._interpolate_data(
                    aligned_ref_x, aligned_ref_y, 
                    aligned_test_x, aligned_test_y,
                    alignment_config.interpolation
                )
                aligned_ref_x = aligned_test_x  # Use test time points as reference
        
        # Round time values if specified (for display purposes)
        if alignment_config.round_to is not None:
            aligned_ref_x = np.round(aligned_ref_x, alignment_config.round_to)
            aligned_test_x = np.round(aligned_test_x, alignment_config.round_to)
        
        alignment_info = {
            'method': 'time',
            'mode': alignment_config.mode,
            'offset': offset,
            'interpolation': alignment_config.interpolation,
            'round_to': alignment_config.round_to,
            'resolution': alignment_config.resolution,
            'original_ref_length': len(ref_ydata),
            'original_test_length': len(test_ydata),
            'aligned_length': len(aligned_ref_y),
            'ref_time_range': (aligned_ref_x[0], aligned_ref_x[-1]) if len(aligned_ref_x) > 0 else None,
            'test_time_range': (aligned_test_x[0], aligned_test_x[-1]) if len(aligned_test_x) > 0 else None
        }
        
        return AlignmentResult(
            ref_data=aligned_ref_y,
            test_data=aligned_test_y,
            success=True, # Time alignment is always successful
            error_message=None
        )
    
    def _interpolate_data(self, ref_x: np.ndarray, ref_y: np.ndarray,
                         test_x: np.ndarray, test_y: np.ndarray,
                         method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate data to common time points"""
        try:
            if method == 'linear':
                from scipy.interpolate import interp1d
                
                # Interpolate reference data to test time points
                ref_interp = interp1d(ref_x, ref_y, kind='linear', 
                                    bounds_error=False, fill_value=np.nan)
                aligned_ref_y = ref_interp(test_x)
                
                # Handle NaN values by using nearest neighbor for those points
                nan_mask = np.isnan(aligned_ref_y)
                if np.any(nan_mask):
                    nearest_interp = interp1d(ref_x, ref_y, kind='nearest', 
                                            bounds_error=False, fill_value=np.nan)
                    aligned_ref_y[nan_mask] = nearest_interp(test_x[nan_mask])
                
                return aligned_ref_y, test_y
                
            elif method == 'cubic':
                from scipy.interpolate import interp1d
                
                # Interpolate reference data to test time points
                ref_interp = interp1d(ref_x, ref_y, kind='cubic', 
                                    bounds_error=False, fill_value=np.nan)
                aligned_ref_y = ref_interp(test_x)
                
                # Handle NaN values by using nearest neighbor for those points
                nan_mask = np.isnan(aligned_ref_y)
                if np.any(nan_mask):
                    nearest_interp = interp1d(ref_x, ref_y, kind='nearest', 
                                            bounds_error=False, fill_value=ref_y[0])
                    aligned_ref_y[nan_mask] = nearest_interp(test_x[nan_mask])
                
                return aligned_ref_y, test_y
                
            elif method == 'nearest':
                from scipy.interpolate import interp1d
                
                # Interpolate reference data to test time points
                ref_interp = interp1d(ref_x, ref_y, kind='nearest', 
                                    bounds_error=False, fill_value=ref_y[0])
                aligned_ref_y = ref_interp(test_x)
                
                return aligned_ref_y, test_y
                
            else:
                warnings.warn(f"Unknown interpolation method: {method}, using linear")
                return self._interpolate_data(ref_x, ref_y, test_x, test_y, 'linear')
                
        except ImportError:
            warnings.warn("scipy not available, using simple nearest neighbor interpolation")
            # Fallback to simple nearest neighbor
            aligned_ref_y = np.zeros_like(test_y)
            for i, tx in enumerate(test_x):
                # Find nearest reference time point
                nearest_idx = np.argmin(np.abs(ref_x - tx))
                aligned_ref_y[i] = ref_y[nearest_idx]
            
            return aligned_ref_y, test_y
    
    def _interpolate_to_grid(self, source_x: np.ndarray, source_y: np.ndarray,
                            target_x: np.ndarray, method: str) -> np.ndarray:
        """Interpolate data to a uniform time grid"""
        try:
            from scipy.interpolate import interp1d
            
            if method == 'linear':
                interp_func = interp1d(source_x, source_y, kind='linear', 
                                     bounds_error=False, fill_value=np.nan)
            elif method == 'cubic':
                interp_func = interp1d(source_x, source_y, kind='cubic', 
                                     bounds_error=False, fill_value=np.nan)
            elif method == 'nearest':
                interp_func = interp1d(source_x, source_y, kind='nearest', 
                                     bounds_error=False, fill_value=source_y[0])
            else:
                # Default to linear
                interp_func = interp1d(source_x, source_y, kind='linear', 
                                     bounds_error=False, fill_value=np.nan)
            
            interpolated_y = interp_func(target_x)
            
            # Handle NaN values for linear/cubic interpolation
            if method in ['linear', 'cubic']:
                nan_mask = np.isnan(interpolated_y)
                if np.any(nan_mask):
                    nearest_interp = interp1d(source_x, source_y, kind='nearest', 
                                            bounds_error=False, fill_value=source_y[0])
                    interpolated_y[nan_mask] = nearest_interp(target_x[nan_mask])
            
            return interpolated_y
            
        except ImportError:
            warnings.warn("scipy not available, using simple nearest neighbor interpolation")
            # Fallback to simple nearest neighbor
            interpolated_y = np.zeros_like(target_x)
            for i, tx in enumerate(target_x):
                # Find nearest source time point
                nearest_idx = np.argmin(np.abs(source_x - tx))
                interpolated_y[i] = source_y[nearest_idx]
            
            return interpolated_y
    
    def _generate_cache_key(self, ref_channel: Channel, test_channel: Channel, 
                           alignment_config: AlignmentConfig) -> str:
        """Generate cache key for alignment result"""
        import hashlib
        
        # Create a string representation of the alignment request
        key_parts = [
            ref_channel.channel_id,
            test_channel.channel_id,
            ref_channel.data_hash,
            test_channel.data_hash,
            str(alignment_config.method.value),
            str(alignment_config.mode),
            str(alignment_config.offset),
            str(alignment_config.start_index),
            str(alignment_config.end_index),
            str(alignment_config.start_time),
            str(alignment_config.end_time),
            str(alignment_config.interpolation),
            str(alignment_config.round_to),
            str(alignment_config.resolution)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the alignment cache"""
        self._alignment_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._alignment_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        } 