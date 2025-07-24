import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
from pair import AlignmentConfig, AlignmentMethod
from channel import Channel
import warnings


@dataclass
class ValidationResult:
    """Result of validation operations"""
    is_valid: bool
    issues: list
    warnings: list
    quality_metrics: Optional[Dict[str, Any]] = None
    data_quality_score: Optional[float] = None


@dataclass
class AlignmentResult:
    """Container for alignment results - just the aligned data"""
    ref_data: np.ndarray
    test_data: np.ndarray
    success: bool = True
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    warnings: Optional[list] = None
    
    def __post_init__(self):
        if not self.success and not self.error_message:
            self.error_message = "Unknown alignment error"


class DataQualityValidator:
    """Comprehensive data quality assessment for alignment inputs"""
    
    def validate_channel_data(self, channel: Channel) -> ValidationResult:
        """Validate channel data quality and characteristics"""
        issues = []
        warnings = []
        
        # Basic data presence
        if channel.ydata is None or len(channel.ydata) == 0:
            issues.append("Channel has no Y data")
        
        # Data type validation
        if channel.ydata is not None and not isinstance(channel.ydata, np.ndarray):
            try:
                channel.ydata = np.array(channel.ydata)
                warnings.append("Y data converted to numpy array")
            except:
                issues.append("Y data cannot be converted to numpy array")
        
        # NaN/Infinite validation
        if channel.ydata is not None:
            nan_count = np.sum(np.isnan(channel.ydata))
            inf_count = np.sum(np.isinf(channel.ydata))
            total_count = len(channel.ydata)
            
            if nan_count > 0:
                nan_ratio = nan_count / total_count
                if nan_ratio > 0.5:
                    issues.append(f"Channel has {nan_ratio*100:.1f}% NaN values")
                elif nan_ratio > 0.1:
                    warnings.append(f"Channel has {nan_ratio*100:.1f}% NaN values")
            
            if inf_count > 0:
                issues.append(f"Channel has {inf_count} infinite values")
        
        # X-data validation for time-based alignment
        if channel.xdata is not None:
            # Check for monotonic time
            if not np.all(np.diff(channel.xdata) >= 0):
                warnings.append("X data is not monotonically increasing")
            
            # Check for duplicate time points
            unique_x_count = len(np.unique(channel.xdata))
            if unique_x_count < len(channel.xdata):
                duplicate_ratio = 1 - (unique_x_count / len(channel.xdata))
                if duplicate_ratio > 0.1:
                    warnings.append(f"X data has {duplicate_ratio*100:.1f}% duplicate time points")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            data_quality_score=self._calculate_quality_score(channel, issues, warnings)
        )
    
    def _calculate_quality_score(self, channel: Channel, issues: list, warnings: list) -> float:
        """Calculate data quality score (0-1)"""
        score = 1.0
        
        # Penalize for issues
        score -= len(issues) * 0.2
        
        # Penalize for warnings
        score -= len(warnings) * 0.1
        
        # Penalize for NaN values
        if channel.ydata is not None:
            nan_ratio = np.sum(np.isnan(channel.ydata)) / len(channel.ydata)
            score -= nan_ratio * 0.3
        
        return max(0.0, min(1.0, score))


class ParameterValidator:
    """Validate alignment parameters and configurations"""
    
    def validate_alignment_params(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate alignment parameters from wizard"""
        issues = []
        warnings = []
        
        # Handle both comparison wizard format and signal mixer wizard format
        alignment_method = params.get('alignment_method')
        mode = params.get('mode')
        
        # Check for valid alignment method
        if alignment_method not in ['index', 'time']:
            issues.append(f"Invalid alignment method: {alignment_method}")
        
        if alignment_method == 'index':
            start_idx = params.get('start_index', 0)
            end_idx = params.get('end_index', 0)
            offset = params.get('offset', 0)
            
            if not isinstance(start_idx, int) or start_idx < 0:
                issues.append("Start index must be non-negative integer")
            
            if not isinstance(end_idx, int) or end_idx < 0:
                issues.append("End index must be non-negative integer")
            
            if start_idx >= end_idx:
                issues.append("Start index must be less than end index")
                
            if abs(offset) > 1000000:
                warnings.append("Large offset value may cause performance issues")
            
            # Validate mode for index-based alignment
            if mode not in ['truncate', 'custom']:
                warnings.append(f"Unknown index mode: {mode}, using 'truncate'")
        
        elif alignment_method == 'time':
            start_time = params.get('start_time', 0.0)
            end_time = params.get('end_time', 0.0)
            
            if start_time >= end_time:
                issues.append("Start time must be less than end time")
            
            # Check for resolution or round_to
            resolution = params.get('resolution', params.get('round_to', 0.1))
            if resolution <= 0:
                issues.append("Resolution must be positive")
            
            interpolation = params.get('interpolation', 'linear')
            if interpolation not in ['linear', 'nearest', 'cubic']:
                warnings.append(f"Unknown interpolation method: {interpolation}")
            
            # Validate mode for time-based alignment
            if mode not in ['overlap', 'custom']:
                warnings.append(f"Unknown time mode: {mode}, using 'overlap'")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )


class DateTimeConverter:
    """Advanced datetime detection and conversion capabilities"""
    
    def __init__(self):
        self.datetime_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%j',  # Julian date
            '%Y%m%d',  # YYYYMMDD
            '%Y%m%d%H%M%S',  # YYYYMMDDHHMMSS
        ]
    
    def detect_and_convert_datetime(self, channel: Channel) -> Channel:
        """Detect and convert datetime data in channel"""
        if channel.xdata is None:
            return channel
        
        # Check if already converted
        if channel.metadata.get('x_is_datetime'):
            return channel
        
        # Try to detect datetime patterns
        conversion_result = self._attempt_datetime_conversion(channel.xdata)
        
        if conversion_result['success']:
            # Update channel with converted data
            channel.xdata = conversion_result['numeric_values']
            channel.metadata.update({
                'x_is_datetime': True,
                'datetime_original': conversion_result['original_datetime'],
                'datetime_format': conversion_result['format_used'],
                'datetime_reference': conversion_result['reference_time'],
                'datetime_sampling_stats': conversion_result['sampling_stats']
            })
            
            # Update xlabel to reflect datetime conversion
            ref_time = conversion_result['reference_time']
            if isinstance(ref_time, pd.Timestamp):
                channel.xlabel = f"Time (seconds from {ref_time.strftime('%Y-%m-%d %H:%M:%S')})"
            else:
                channel.xlabel = f"Time (seconds from reference)"
        
        return channel
    
    def _attempt_datetime_conversion(self, xdata: np.ndarray) -> Dict[str, Any]:
        """Attempt to convert x-data to datetime"""
        try:
            # First try pandas auto-detection
            if isinstance(xdata[0], str):
                datetime_index = pd.to_datetime(xdata, errors='coerce')
                if not datetime_index.isnull().all():
                    datetime_series = pd.Series(datetime_index)
                    return self._process_datetime_series(datetime_series, 'pandas_auto')
            
            # Try specific formats
            for fmt in self.datetime_formats:
                try:
                    datetime_index = pd.to_datetime(xdata, format=fmt, errors='coerce')
                    if not datetime_index.isnull().all():
                        datetime_series = pd.Series(datetime_index)
                        return self._process_datetime_series(datetime_series, fmt)
                except:
                    continue
            
            # Try Unix timestamp detection
            if self._could_be_unix_timestamp(xdata):
                datetime_index = pd.to_datetime(xdata, unit='s', errors='coerce')
                if not datetime_index.isnull().all():
                    datetime_series = pd.Series(datetime_index)
                    return self._process_datetime_series(datetime_series, 'unix_timestamp')
            
            return {'success': False, 'reason': 'No datetime format detected'}
            
        except Exception as e:
            return {'success': False, 'reason': f'Conversion error: {e}'}
    
    def _process_datetime_series(self, datetime_series: pd.Series, format_used: str) -> Dict[str, Any]:
        """Process successfully converted datetime series"""
        # Remove NaN values
        clean_series = datetime_series.dropna()
        if len(clean_series) < 2:
            return {'success': False, 'reason': 'Insufficient valid datetime values'}
        
        # Use first datetime as reference
        reference_time = clean_series.iloc[0]
        
        # Convert to numeric (seconds from reference)
        numeric_values = (clean_series - reference_time).dt.total_seconds().values
        
        # Calculate sampling statistics
        sampling_stats = self._compute_datetime_sampling_stats(clean_series)
        
        return {
            'success': True,
            'numeric_values': numeric_values,
            'original_datetime': clean_series.values,
            'format_used': format_used,
            'reference_time': reference_time,
            'sampling_stats': sampling_stats
        }
    
    def _compute_datetime_sampling_stats(self, clean_series: pd.Series) -> Dict[str, Any]:
        """Compute sampling statistics for datetime series"""
        if len(clean_series) < 2:
            return {'sampling_rate': None, 'irregularity': None}
        
        # Calculate time differences
        time_diffs = clean_series.diff().dt.total_seconds()
        time_diffs = time_diffs.dropna()
        
        if len(time_diffs) == 0:
            return {'sampling_rate': None, 'irregularity': None}
        
        # Calculate stats
        mean_interval = time_diffs.mean()
        std_interval = time_diffs.std()
        
        return {
            'sampling_rate': 1.0 / mean_interval if mean_interval > 0 else None,
            'mean_interval': mean_interval,
            'std_interval': std_interval,
            'irregularity': std_interval / mean_interval if mean_interval > 0 else None
        }
    
    def _could_be_unix_timestamp(self, xdata: np.ndarray) -> bool:
        """Check if data could be Unix timestamps"""
        if not np.all(np.isfinite(xdata)):
            return False
        
        # Check if values are in reasonable Unix timestamp range
        # (1970-01-01 to 2100-01-01)
        min_val = np.min(xdata)
        max_val = np.max(xdata)
        
        return (0 < min_val < 4e9) and (0 < max_val < 4e9)


class AlignmentResultValidator:
    """Validate alignment results for quality and consistency"""
    
    def validate_alignment_result(self, result: AlignmentResult, 
                                 original_ref: Channel, original_test: Channel) -> ValidationResult:
        """Comprehensive validation of alignment results"""
        issues = []
        warnings = []
        quality_metrics = {}
        
        # Basic result validation
        if not result.success:
            issues.append(f"Alignment failed: {result.error_message}")
            return ValidationResult(False, issues, warnings, quality_metrics)
        
        # Data presence validation
        if result.ref_data is None or result.test_data is None:
            issues.append("Alignment result contains null data")
            return ValidationResult(False, issues, warnings, quality_metrics)
        
        # Data length validation
        if len(result.ref_data) != len(result.test_data):
            issues.append("Aligned data arrays have different lengths")
        
        if len(result.ref_data) == 0:
            issues.append("Alignment result contains no data points")
        
        # Data loss assessment
        original_ref_len = len(original_ref.ydata) if original_ref.ydata is not None else 0
        original_test_len = len(original_test.ydata) if original_test.ydata is not None else 0
        aligned_len = len(result.ref_data)
        
        if original_ref_len > 0:
            ref_retention = aligned_len / original_ref_len
            quality_metrics['ref_data_retention'] = ref_retention
            if ref_retention < 0.1:
                warnings.append(f"Low reference data retention: {ref_retention*100:.1f}%")
        
        if original_test_len > 0:
            test_retention = aligned_len / original_test_len
            quality_metrics['test_data_retention'] = test_retention
            if test_retention < 0.1:
                warnings.append(f"Low test data retention: {test_retention*100:.1f}%")
        
        # Data quality validation
        if len(result.ref_data) > 0:
            ref_nan_count = np.sum(np.isnan(result.ref_data))
            test_nan_count = np.sum(np.isnan(result.test_data))
            
            if ref_nan_count > 0:
                ref_nan_ratio = ref_nan_count / len(result.ref_data)
                quality_metrics['ref_nan_ratio'] = ref_nan_ratio
                if ref_nan_ratio > 0.1:
                    warnings.append(f"Reference data has {ref_nan_ratio*100:.1f}% NaN values after alignment")
            
            if test_nan_count > 0:
                test_nan_ratio = test_nan_count / len(result.test_data)
                quality_metrics['test_nan_ratio'] = test_nan_ratio
                if test_nan_ratio > 0.1:
                    warnings.append(f"Test data has {test_nan_ratio*100:.1f}% NaN values after alignment")
        
        # Statistical validation
        if len(result.ref_data) > 1:
            ref_std = np.std(result.ref_data)
            test_std = np.std(result.test_data)
            
            if ref_std == 0:
                warnings.append("Reference data has zero variance after alignment")
            if test_std == 0:
                warnings.append("Test data has zero variance after alignment")
        
        # Performance metrics
        quality_metrics['final_length'] = aligned_len
        quality_metrics['alignment_success'] = True
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            quality_metrics=quality_metrics
        )


class DataAligner:
    """Enhanced data aligner with comprehensive validation and fallback strategies"""
    
    def __init__(self):
        self._alignment_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize validators and converters
        self.data_validator = DataQualityValidator()
        self.param_validator = ParameterValidator()
        self.datetime_converter = DateTimeConverter()
        self.result_validator = AlignmentResultValidator()
        
        # Performance tracking
        self.alignment_stats = {
            'total_alignments': 0,
            'successful_alignments': 0,
            'failed_alignments': 0,
            'datetime_conversions': 0,
            'validation_failures': 0,
            'fallback_usage': 0
        }
    
    def align_from_wizard_params(self, ref_channel: Channel, test_channel: Channel, 
                                alignment_params: Dict[str, Any]) -> AlignmentResult:
        """Enhanced alignment with full validation and robust fallback strategy"""
        self.alignment_stats['total_alignments'] += 1
        error_messages = []
        
        # DEBUG: Print input data lengths
        print(f"[DEBUG] DataAligner input - ref_channel length: {len(ref_channel.ydata) if ref_channel.ydata is not None else 0}")
        print(f"[DEBUG] DataAligner input - test_channel length: {len(test_channel.ydata) if test_channel.ydata is not None else 0}")
        print(f"[DEBUG] DataAligner input - alignment_params: {alignment_params}")
        
        # CRITICAL: Check if channels have data - if not, fail immediately
        if (ref_channel.ydata is None or len(ref_channel.ydata) == 0 or
            test_channel.ydata is None or len(test_channel.ydata) == 0):
            self.alignment_stats['failed_alignments'] += 1
            return AlignmentResult(
                ref_data=np.array([]),
                test_data=np.array([]),
                success=False,
                error_message="Cannot align: one or both channels have no data"
            )
        
        try:
            # Step 1: Validate input parameters - but continue with fallback if invalid
            param_validation = self.param_validator.validate_alignment_params(alignment_params)
            if not param_validation.is_valid:
                error_messages.append(f"Parameter validation: {', '.join(param_validation.issues)}")
                alignment_params = self._get_fallback_params(alignment_params, ref_channel, test_channel)
                self.alignment_stats['fallback_usage'] += 1
            
            # Step 2: Validate input channels - but continue with cleanup if issues
            ref_validation = self.data_validator.validate_channel_data(ref_channel)
            test_validation = self.data_validator.validate_channel_data(test_channel)
            
            if not ref_validation.is_valid or not test_validation.is_valid:
                error_messages.append(f"Channel validation: {', '.join(ref_validation.issues + test_validation.issues)}")
                # Clean up channels before proceeding
                ref_channel = self._cleanup_channel_data(ref_channel)
                test_channel = self._cleanup_channel_data(test_channel)
                self.alignment_stats['fallback_usage'] += 1
            
            # DEBUG: Print data lengths after cleanup
            print(f"[DEBUG] DataAligner after cleanup - ref_channel length: {len(ref_channel.ydata) if ref_channel.ydata is not None else 0}")
            print(f"[DEBUG] DataAligner after cleanup - test_channel length: {len(test_channel.ydata) if test_channel.ydata is not None else 0}")
            
            # Step 3: Apply datetime conversion if needed - but continue if it fails
            try:
                enhanced_ref = self.datetime_converter.detect_and_convert_datetime(ref_channel)
                enhanced_test = self.datetime_converter.detect_and_convert_datetime(test_channel)
                
                if enhanced_ref.metadata.get('x_is_datetime') or enhanced_test.metadata.get('x_is_datetime'):
                    self.alignment_stats['datetime_conversions'] += 1
            except Exception as e:
                error_messages.append(f"DateTime conversion failed: {e}")
                enhanced_ref = ref_channel
                enhanced_test = test_channel
                self.alignment_stats['fallback_usage'] += 1
            
            # Step 4: Perform alignment - this is the core operation
            alignment_result = self._perform_enhanced_alignment(enhanced_ref, enhanced_test, alignment_params)
            
            # DEBUG: Print alignment result lengths
            print(f"[DEBUG] DataAligner result - ref_data length: {len(alignment_result.ref_data) if alignment_result.success else 0}")
            print(f"[DEBUG] DataAligner result - test_data length: {len(alignment_result.test_data) if alignment_result.success else 0}")
            
            # Step 5: Validate alignment result - but don't fail if validation fails
            if alignment_result.success:
                try:
                    result_validation = self.result_validator.validate_alignment_result(
                        alignment_result, enhanced_ref, enhanced_test
                    )
                    
                    if not result_validation.is_valid:
                        error_messages.append(f"Result validation: {', '.join(result_validation.issues)}")
                        # Don't fail - just add warnings
                        alignment_result.warnings = result_validation.warnings
                    else:
                        # Add quality metrics to result
                        alignment_result.quality_metrics = result_validation.quality_metrics
                        alignment_result.warnings = result_validation.warnings
                except Exception as e:
                    error_messages.append(f"Result validation failed: {e}")
                    self.alignment_stats['fallback_usage'] += 1
                
                self.alignment_stats['successful_alignments'] += 1
            else:
                self.alignment_stats['failed_alignments'] += 1
            
            # Add accumulated error messages to result
            if error_messages:
                if alignment_result.error_message:
                    alignment_result.error_message += f" | Validation issues: {'; '.join(error_messages)}"
                else:
                    alignment_result.error_message = f"Validation issues: {'; '.join(error_messages)}"
            
            return alignment_result
            
        except Exception as e:
            # Final fallback - try basic alignment
            error_messages.append(f"Enhanced alignment failed: {e}")
            self.alignment_stats['fallback_usage'] += 1
            
            try:
                # Convert to basic alignment config and try again
                basic_config = self._convert_wizard_params_to_config(alignment_params)
                fallback_result = self._basic_align_channels(ref_channel, test_channel, basic_config)
                
                if fallback_result.success:
                    self.alignment_stats['successful_alignments'] += 1
                    fallback_result.error_message = f"Used fallback alignment: {'; '.join(error_messages)}"
                    return fallback_result
                else:
                    self.alignment_stats['failed_alignments'] += 1
                    fallback_result.error_message = f"All alignment methods failed: {'; '.join(error_messages + [fallback_result.error_message or 'Unknown error'])}"
                    return fallback_result
                    
            except Exception as final_e:
                self.alignment_stats['failed_alignments'] += 1
                return AlignmentResult(
                    ref_data=np.array([]),
                    test_data=np.array([]),
                    success=False,
                    error_message=f"Complete alignment failure: {'; '.join(error_messages + [str(final_e)])}"
                )
    
    def _perform_enhanced_alignment(self, ref_channel: Channel, test_channel: Channel, 
                                   alignment_params: Dict[str, Any]) -> AlignmentResult:
        """Perform enhanced alignment with all validation features"""
        # Convert parameters to config and use basic alignment
        alignment_config = self._convert_wizard_params_to_config(alignment_params)
        return self._basic_align_channels(ref_channel, test_channel, alignment_config)
    
    def _get_fallback_params(self, params: Dict[str, Any], ref_channel: Channel, test_channel: Channel) -> Dict[str, Any]:
        """Get safe fallback parameters when validation fails"""
        fallback_params = params.copy()
        
        alignment_method = params.get('alignment_method', 'time')
        
        if alignment_method == 'index' or alignment_method not in ['index', 'time']:
            # Safe index-based fallback
            ref_len = len(ref_channel.ydata) if ref_channel.ydata is not None else 0
            test_len = len(test_channel.ydata) if test_channel.ydata is not None else 0
            safe_len = min(ref_len, test_len, 1000)  # Limit to prevent memory issues
            
            fallback_params.update({
                'alignment_method': 'index',
                'mode': 'truncate',  # Use valid mode
                'start_index': 0,
                'end_index': safe_len,
                'offset': 0
            })
        
        elif alignment_method == 'time':
            # Safe time-based fallback
            if ref_channel.xdata is not None and test_channel.xdata is not None:
                ref_time_range = (ref_channel.xdata[0], ref_channel.xdata[-1])
                test_time_range = (test_channel.xdata[0], test_channel.xdata[-1])
                
                # Find safe overlap
                safe_start = max(ref_time_range[0], test_time_range[0])
                safe_end = min(ref_time_range[1], test_time_range[1])
                
                if safe_start < safe_end:
                    fallback_params.update({
                        'alignment_method': 'time',
                        'mode': 'overlap',  # Use valid mode
                        'start_time': safe_start,
                        'end_time': safe_end,
                        'offset': 0.0,
                        'interpolation': 'nearest',
                        'resolution': 1.0
                    })
                else:
                    # Fall back to index-based
                    return self._get_fallback_params({'alignment_method': 'index'}, ref_channel, test_channel)
            else:
                # No time data - fall back to index-based
                return self._get_fallback_params({'alignment_method': 'index'}, ref_channel, test_channel)
        
        return fallback_params
    
    def _cleanup_channel_data(self, channel: Channel) -> Channel:
        """Clean up channel data to make it more suitable for alignment"""
        if channel.ydata is not None:
            # Remove NaN and infinite values
            finite_mask = np.isfinite(channel.ydata)
            if np.any(finite_mask):
                channel.ydata = channel.ydata[finite_mask]
                if channel.xdata is not None:
                    channel.xdata = channel.xdata[finite_mask]
            
            # Ensure data is not empty after cleanup
            if len(channel.ydata) == 0:
                # Create minimal dummy data to prevent complete failure
                channel.ydata = np.array([0.0])
                if channel.xdata is not None:
                    channel.xdata = np.array([0.0])
        
        return channel
    
    def _convert_wizard_params_to_config(self, alignment_params: Dict[str, Any]) -> 'AlignmentConfig':
        """Convert wizard alignment parameters to AlignmentConfig"""
        from pair import AlignmentConfig, AlignmentMethod
        
        # Get alignment method from params
        alignment_method = alignment_params.get('alignment_method', 'time')
        mode = alignment_params.get('mode', 'truncate')
        
        if alignment_method == 'index':
            # Index-based alignment
            start_index = alignment_params.get('start_index', 0)
            end_index = alignment_params.get('end_index', 100)
            offset = alignment_params.get('offset', 0)
            
            return AlignmentConfig(
                method=AlignmentMethod.INDEX,
                mode=mode,  # Use the actual mode from params ('truncate' or 'custom')
                offset=offset,
                start_index=start_index,
                end_index=end_index,
                start_time=None,
                end_time=None,
                interpolation=None,
                round_to=None
            )
            
        elif alignment_method == 'time':
            # Time-based alignment
            start_time = alignment_params.get('start_time', 0.0)
            end_time = alignment_params.get('end_time', 10.0)
            offset = alignment_params.get('offset', 0.0)
            interpolation = alignment_params.get('interpolation', 'nearest')
            resolution = alignment_params.get('resolution', alignment_params.get('round_to', 0.1))
            
            # Store resolution for time grid creation
            round_to = None
            if resolution is not None and resolution > 0:
                if resolution < 1:
                    round_to = abs(int(np.log10(resolution)))
                else:
                    round_to = 0
            
            return AlignmentConfig(
                method=AlignmentMethod.TIME,
                mode=mode,  # Use the actual mode from params ('overlap' or 'custom')
                offset=offset,
                start_index=None,
                end_index=None,
                start_time=start_time,
                end_time=end_time,
                interpolation=interpolation,
                round_to=round_to,
                resolution=resolution
            )
            
        else:
            raise ValueError(f"Unknown alignment method: {alignment_method}")
    
    def _basic_align_channels(self, ref_channel: Channel, test_channel: Channel, 
                      alignment_config: AlignmentConfig) -> AlignmentResult:
        """Basic alignment method with minimal validation"""
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
        
        # Check for self-comparison - only check channel_id since it's unique
        if ref_channel.channel_id == test_channel.channel_id:
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
        
        # DEBUG: Print alignment method details
        print(f"[DEBUG] DataAligner._align_by_index - mode: {alignment_config.mode}")
        print(f"[DEBUG] DataAligner._align_by_index - ref_data length: {len(ref_data) if ref_data is not None else 0}")
        print(f"[DEBUG] DataAligner._align_by_index - test_data length: {len(test_data) if test_data is not None else 0}")
        if alignment_config.mode == 'custom':
            print(f"[DEBUG] DataAligner._align_by_index - start_index: {alignment_config.start_index}, end_index: {alignment_config.end_index}")
        print(f"[DEBUG] DataAligner._align_by_index - offset: {alignment_config.offset}")
        
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
            print(f"[DEBUG] DataAligner._align_by_index truncate - min_length: {min_length}")
            
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
        
        # DEBUG: Print final aligned data lengths
        print(f"[DEBUG] DataAligner._align_by_index final - aligned_ref length: {len(aligned_ref)}")
        print(f"[DEBUG] DataAligner._align_by_index final - aligned_test length: {len(aligned_test)}")
        
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
            uniform_time_grid = np.arange(time_start, time_end, 
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

    def get_alignment_stats(self) -> Dict[str, Any]:
        """Get alignment performance statistics"""
        return {
            'total_alignments': self.alignment_stats['total_alignments'],
            'successful_alignments': self.alignment_stats['successful_alignments'],
            'failed_alignments': self.alignment_stats['failed_alignments'],
            'datetime_conversions': self.alignment_stats['datetime_conversions'],
            'validation_failures': self.alignment_stats['validation_failures'],
            'fallback_usage': self.alignment_stats['fallback_usage'],
            'success_rate': self.alignment_stats['successful_alignments'] / self.alignment_stats['total_alignments'] if self.alignment_stats['total_alignments'] > 0 else 0,
            'fallback_rate': self.alignment_stats['fallback_usage'] / self.alignment_stats['total_alignments'] if self.alignment_stats['total_alignments'] > 0 else 0
        } 