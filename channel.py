import enum
import numpy as np
from typing import Optional, List, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib
import time
import uuid
import weakref
from functools import cached_property, lru_cache


class SourceType(Enum):
    RAW = "raw"
    PROCESSED = "processed"
    MIXED = "mixed"
    SPECTROGRAM = "spectrogram"
    COMPARISON = "comparison"


@dataclass
class DataStats:
    """Lightweight container for computed statistics"""
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    count: int
    
    def __post_init__(self):
        """Validate stats after creation"""
        if self.count <= 0:
            raise ValueError("Count must be positive")


@dataclass
class SamplingStats:
    """Container for sampling rate statistics"""
    median_fs: float
    std_fs: float
    min_fs: float
    max_fs: float
    regularity_score: float  # 0-1, how regular the sampling is
    
    @property
    def is_regular(self) -> bool:
        """True if sampling is reasonably regular (std < 5% of median)"""
        return self.regularity_score > 0.95


class Channel:
    """
    Optimized class to store information about a data channel.
    
    Key optimizations:
    - Lazy loading of statistics and computations
    - Weak references to prevent circular references
    - Cached properties for expensive operations
    - Memory-efficient data storage options
    - Enhanced factory methods for different use cases
    """
    
    # Class-level cache for expensive operations
    _computation_cache = {}
    _cache_hits = 0
    _cache_misses = 0
    
    def __init__(self, channel_id=None, file_id=None, filename=None, xdata=None, ydata=None,
                 xlabel=None, ylabel=None, legend_label=None, type=None, step=0, tags=None,
                 description=None, metadata=None, lazy_init=True):
        
        # Core identifiers
        self.channel_id = channel_id or self._generate_channel_id()
        self.file_id = file_id
        self.filename = filename
        self.lineage_id = None
        
        # Data arrays - consider using views or references for large data
        self._xdata = xdata
        self._ydata = ydata
        self._data_loaded = xdata is not None and ydata is not None
        
        # Labels and display
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend_label = legend_label
        self.description = description or ""
        
        # Processing metadata
        self.type = type or SourceType.RAW
        self.step = step
        self.tags = tags or []
        self.params = {}  # Parameters used to create this channel
        
        # Visualization properties
        self.show = True
        self.yaxis = "y-left"
        self.color = None
        self.style = "-"
        self.marker = "None"
        
        # Lineage tracking
        self.parent_ids = []
        self.file_status = None
        
        # Extended metadata
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.modified_at = self.created_at
        
        # Cached statistics (computed on demand)
        self._x_stats = None
        self._y_stats = None
        self._sampling_stats = None
        self._data_hash = None
        
        # Performance tracking
        self._access_count = 0
        self._last_access = None
        
        # Initialize basic stats if not lazy loading
        if not lazy_init and self._data_loaded:
            self._eager_init_stats()

    @staticmethod
    def _generate_channel_id():
        """Generate a unique channel ID using timestamp and random bytes"""
        timestamp = str(time.time_ns())
        random_bytes = str(uuid.uuid4().hex[:8])
        combined = f"{timestamp}{random_bytes}"
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    def _eager_init_stats(self):
        """Initialize statistics immediately (for backwards compatibility)"""
        if self._data_loaded:
            _ = self.x_stats  # Trigger computation
            _ = self.y_stats  # Trigger computation
            _ = self.sampling_stats  # Trigger computation

    def _update_access_tracking(self):
        """Track access patterns for optimization insights"""
        self._access_count += 1
        self._last_access = time.time()

    @property
    def xdata(self) -> Optional[np.ndarray]:
        """Access x-data with lazy loading support"""
        self._update_access_tracking()
        return self._xdata

    @xdata.setter
    def xdata(self, value: Optional[np.ndarray]):
        """Set x-data and invalidate cached statistics"""
        self._xdata = value
        self._data_loaded = value is not None and self._ydata is not None
        self._invalidate_stats()

    @property
    def ydata(self) -> Optional[np.ndarray]:
        """Access y-data with lazy loading support"""
        self._update_access_tracking()
        return self._ydata

    @ydata.setter
    def ydata(self, value: Optional[np.ndarray]):
        """Set y-data and invalidate cached statistics"""
        self._ydata = value
        self._data_loaded = self._xdata is not None and value is not None
        self._invalidate_stats()

    def _invalidate_stats(self):
        """Invalidate all cached statistics when data changes"""
        self._x_stats = None
        self._y_stats = None
        self._sampling_stats = None
        self._data_hash = None
        self.modified_at = datetime.now()

    @cached_property
    def data_hash(self) -> Optional[str]:
        """Compute hash of data for change detection and caching"""
        if not self._data_loaded:
            return None
        
        try:
            x_bytes = self._xdata.tobytes() if self._xdata is not None else b''
            y_bytes = self._ydata.tobytes() if self._ydata is not None else b''
            combined = x_bytes + y_bytes
            return hashlib.md5(combined).hexdigest()
        except Exception as e:
            print(f"[Channel] Warning: Could not compute data hash: {e}")
            return None

    @cached_property
    def x_stats(self) -> Optional[DataStats]:
        """Compute x-data statistics with caching"""
        if not self._data_loaded or self._xdata is None:
            return None
        
        cache_key = f"x_stats_{self.channel_id}_{self.data_hash}"
        if cache_key in self._computation_cache:
            self._cache_hits += 1
            return self._computation_cache[cache_key]
        
        try:
            self._cache_misses += 1
            if len(self._xdata) == 0:
                return None
                
            # Use numpy's optimized functions
            stats = DataStats(
                min_val=float(np.min(self._xdata)),
                max_val=float(np.max(self._xdata)),
                mean_val=float(np.mean(self._xdata)),
                std_val=float(np.std(self._xdata)),
                count=len(self._xdata)
            )
            
            # Cache the result
            self._computation_cache[cache_key] = stats
            return stats
            
        except Exception as e:
            print(f"[Channel] Warning: Could not compute x-stats: {e}")
            return None

    @cached_property
    def y_stats(self) -> Optional[DataStats]:
        """Compute y-data statistics with caching"""
        if not self._data_loaded or self._ydata is None:
            return None
        
        cache_key = f"y_stats_{self.channel_id}_{self.data_hash}"
        if cache_key in self._computation_cache:
            self._cache_hits += 1
            return self._computation_cache[cache_key]
        
        try:
            self._cache_misses += 1
            if len(self._ydata) == 0:
                return None
                
            stats = DataStats(
                min_val=float(np.min(self._ydata)),
                max_val=float(np.max(self._ydata)),
                mean_val=float(np.mean(self._ydata)),
                std_val=float(np.std(self._ydata)),
                count=len(self._ydata)
            )
            
            self._computation_cache[cache_key] = stats
            return stats
            
        except Exception as e:
            print(f"[Channel] Warning: Could not compute y-stats: {e}")
            return None

    @cached_property
    def sampling_stats(self) -> Optional[SamplingStats]:
        """Compute sampling rate statistics with enhanced analysis"""
        if not self._data_loaded or self._xdata is None or len(self._xdata) < 2:
            return None
        
        cache_key = f"sampling_stats_{self.channel_id}_{self.data_hash}"
        if cache_key in self._computation_cache:
            self._cache_hits += 1
            return self._computation_cache[cache_key]
        
        try:
            self._cache_misses += 1
            
            # Check if we have pre-computed datetime-based sampling stats
            if (self.metadata and 
                self.metadata.get('x_is_datetime', False) and 
                'datetime_sampling_stats' in self.metadata):
                
                datetime_stats = self.metadata['datetime_sampling_stats']
                stats = SamplingStats(
                    median_fs=datetime_stats['median_fs'],
                    std_fs=datetime_stats['std_fs'],
                    min_fs=datetime_stats['min_fs'],
                    max_fs=datetime_stats['max_fs'],
                    regularity_score=datetime_stats['regularity_score']
                )
                
                # Store additional datetime info for debugging/display
                stats._datetime_method = True
                stats._interval_description = datetime_stats.get('interval_description', 'N/A')
                stats._total_duration = datetime_stats.get('total_duration_seconds', 0)
                
                self._computation_cache[cache_key] = stats
                return stats
            
            # Fallback to numeric-based calculation
            time_diffs = np.diff(self._xdata)
            valid_diffs = time_diffs[time_diffs > 0]
            
            if len(valid_diffs) == 0:
                return None
            
            sampling_rates = 1.0 / valid_diffs
            valid_rates = sampling_rates[np.isfinite(sampling_rates)]
            
            if len(valid_rates) == 0:
                return None
            
            median_fs = float(np.median(valid_rates))
            std_fs = float(np.std(valid_rates))
            
            # Calculate regularity score (how consistent the sampling is)
            cv = std_fs / median_fs if median_fs > 0 else float('inf')
            regularity_score = max(0.0, min(1.0, 1.0 - cv))
            
            stats = SamplingStats(
                median_fs=median_fs,
                std_fs=std_fs,
                min_fs=float(np.min(valid_rates)),
                max_fs=float(np.max(valid_rates)),
                regularity_score=regularity_score
            )
            
            # Mark as numeric method
            stats._datetime_method = False
            
            self._computation_cache[cache_key] = stats
            return stats
            
        except Exception as e:
            print(f"[Channel] Warning: Could not compute sampling stats: {e}")
            return None

    # Backwards compatibility properties
    @property
    def xmin(self) -> Optional[float]:
        """Get minimum x value (backwards compatibility)"""
        stats = self.x_stats
        return stats.min_val if stats else None

    @property
    def xmax(self) -> Optional[float]:
        """Get maximum x value (backwards compatibility)"""
        stats = self.x_stats
        return stats.max_val if stats else None

    @property
    def ymin(self) -> Optional[float]:
        """Get minimum y value (backwards compatibility)"""
        stats = self.y_stats
        return stats.min_val if stats else None

    @property
    def ymax(self) -> Optional[float]:
        """Get maximum y value (backwards compatibility)"""
        stats = self.y_stats
        return stats.max_val if stats else None

    @property
    def fs_median(self) -> Optional[float]:
        """Get median sampling frequency (backwards compatibility)"""
        stats = self.sampling_stats
        return stats.median_fs if stats else None

    @property
    def fs_std(self) -> Optional[float]:
        """Get sampling frequency standard deviation (backwards compatibility)"""
        stats = self.sampling_stats
        return stats.std_fs if stats else None

    @property
    def xrange(self) -> Optional[tuple[float, float]]:
        """Get the x-axis range (min, max)"""
        stats = self.x_stats
        return (stats.min_val, stats.max_val) if stats else None

    @property
    def yrange(self) -> Optional[tuple[float, float]]:
        """Get the y-axis range (min, max)"""
        stats = self.y_stats
        return (stats.min_val, stats.max_val) if stats else None
    
    def get_sampling_rate_description(self) -> str:
        """Get human-readable sampling rate description"""
        stats = self.sampling_stats
        if not stats:
            return "N/A"
        
        # Check if this is datetime-based sampling
        if hasattr(stats, '_datetime_method') and stats._datetime_method:
            if hasattr(stats, '_interval_description'):
                return f"{stats.median_fs:.3f} Hz ({stats._interval_description} intervals)"
            else:
                return f"{stats.median_fs:.3f} Hz (datetime-based)"
        
        # Standard numeric-based display
        if stats.median_fs >= 1000:
            return f"{stats.median_fs/1000:.1f} kHz"
        elif stats.median_fs >= 1:
            return f"{stats.median_fs:.1f} Hz"
        else:
            return f"{stats.median_fs:.3f} Hz"

    # Enhanced factory methods
    @classmethod
    def from_parsing(cls,
                     file_id: str,
                     filename: str,
                     xdata: np.ndarray,
                     ydata: np.ndarray,
                     xlabel: str,
                     ylabel: str,
                     legend_label: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     metadata: Optional[dict] = None,
                     lazy_init: bool = True) -> "Channel":
        """Create a new channel from parsed raw data (step=0, type=RAW)"""
        lineage_id = f"{file_id}_{ylabel}"
        return cls(
            channel_id=None,
            file_id=file_id,
            filename=filename,
            xdata=xdata,
            ydata=ydata,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_label=legend_label or ylabel,
            type=SourceType.RAW,
            step=0,
            tags=tags or ["time-series"],
            metadata=metadata,
            lazy_init=lazy_init
        ).with_lineage(lineage_id)

    @classmethod
    def from_parent(cls,
                    parent: "Channel",
                    xdata: np.ndarray,
                    ydata: np.ndarray,
                    legend_label: Optional[str] = "",
                    description: Optional[str] = "",
                    tags: Optional[List[str]] = None,
                    type: SourceType = SourceType.PROCESSED,
                    params: Optional[dict] = None,
                    metadata: Optional[dict] = None,
                    lazy_init: bool = True) -> "Channel":
        """Create a new channel based on a parent with incremented step"""
        new_channel = cls(
            channel_id=None,
            file_id=parent.file_id,
            filename=parent.filename,
            xdata=xdata,
            ydata=ydata,
            xlabel=parent.xlabel,
            ylabel=parent.ylabel,
            legend_label=legend_label or f"{parent.legend_label} (step {parent.step + 1})",
            type=type,
            step=parent.step + 1,
            tags=tags or ["time-series"],
            description=description,
            metadata=metadata,
            lazy_init=lazy_init
        )
        
        new_channel.parent_ids = [parent.channel_id]
        new_channel.lineage_id = parent.lineage_id
        new_channel.params = params or {}
        
        return new_channel

    @classmethod
    def from_spectrogram(cls,
                        parent: "Channel",
                        time_data: np.ndarray,
                        freq_data: np.ndarray,
                        power_data: np.ndarray,
                        legend_label: Optional[str] = None,
                        params: Optional[dict] = None) -> "Channel":
        """Create a spectrogram channel with 2D data structure"""
        # Store spectrogram data in metadata for proper handling
        spectrogram_metadata = {
            'spectrogram_data': {
                'time': time_data,
                'frequency': freq_data,
                'power': power_data
            },
            'type': 'spectrogram'
        }
        
        return cls(
            channel_id=None,
            file_id=parent.file_id,
            filename=parent.filename,
            xdata=time_data,  # Use time as x-axis for compatibility
            ydata=freq_data,  # Use frequency as y-axis for compatibility
            xlabel="Time",
            ylabel="Frequency",
            legend_label=legend_label or f"{parent.legend_label} (Spectrogram)",
            type=SourceType.SPECTROGRAM,
            step=parent.step + 1,
            tags=["spectrogram", "frequency-domain"],
            metadata=spectrogram_metadata,
            params=params or {}
        ).with_lineage(parent.lineage_id).with_parent(parent.channel_id)

    @classmethod
    def from_statistics(cls,
                       parent: "Channel",
                       stat_names: List[str],
                       stat_values: np.ndarray,
                       legend_label: Optional[str] = None,
                       params: Optional[dict] = None) -> "Channel":
        """Create a statistical summary channel"""
        return cls(
            channel_id=None,
            file_id=parent.file_id,
            filename=parent.filename,
            xdata=np.arange(len(stat_names)),
            ydata=stat_values,
            xlabel="Statistic",
            ylabel="Value",
            legend_label=legend_label or f"{parent.legend_label} (Stats)",
            type=SourceType.STATISTICAL,
            step=parent.step + 1,
            tags=["statistics", "summary"],
            metadata={'stat_names': stat_names},
            params=params or {}
        ).with_lineage(parent.lineage_id).with_parent(parent.channel_id)

    @classmethod
    def from_comparison(cls,
                       parent_channels: List["Channel"],
                       comparison_method: str,
                       xdata: np.ndarray,
                       ydata: np.ndarray,
                       xlabel: str,
                       ylabel: str,
                       legend_label: str,
                       pairs_metadata: List[dict],
                       statistical_results: dict,
                       method_parameters: dict,
                       overlay_config: dict = None,
                       tags: Optional[List[str]] = None) -> "Channel":
        """
        Create a channel from comparison wizard results
        
        Args:
            parent_channels: List of channels used in the comparison
            comparison_method: Name of the comparison method used
            xdata: Transformed x-data for plotting (e.g., means for Bland-Altman)
            ydata: Transformed y-data for plotting (e.g., differences for Bland-Altman)
            xlabel: X-axis label
            ylabel: Y-axis label
            legend_label: Display label for the comparison
            pairs_metadata: List of dictionaries containing pair information
            statistical_results: Combined statistical results
            method_parameters: Parameters used for the comparison method
            overlay_config: Configuration for plot overlays
            tags: Optional tags for categorization
            
        Returns:
            Channel object representing the comparison results
        """
        # Create a combined lineage ID from all parent channels
        parent_lineage_ids = [ch.lineage_id for ch in parent_channels if ch.lineage_id]
        combined_lineage = f"comparison_{comparison_method}_{hash(tuple(parent_lineage_ids)) % 10000}"
        
        # Get the highest step from parent channels
        max_step = max([ch.step for ch in parent_channels], default=0)
        
        # Create comprehensive metadata
        comparison_metadata = {
            'comparison_method': comparison_method,
            'pairs': pairs_metadata,
            'statistical_results': statistical_results,
            'method_parameters': method_parameters,
            'overlay_config': overlay_config or {},
            'parent_channel_ids': [ch.channel_id for ch in parent_channels],
            'parent_lineage_ids': parent_lineage_ids,
            'creation_timestamp': datetime.now().isoformat(),
            'data_points': len(xdata),
            'comparison_type': 'multi_pair' if len(pairs_metadata) > 1 else 'single_pair'
        }
        
        # Determine primary file info from the first parent channel
        primary_parent = parent_channels[0] if parent_channels else None
        file_id = primary_parent.file_id if primary_parent else "comparison"
        filename = primary_parent.filename if primary_parent else "comparison_result"
        
        # Create the channel
        comparison_channel = cls(
            channel_id=None,
            file_id=file_id,
            filename=filename,
            xdata=xdata,
            ydata=ydata,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_label=legend_label,
            type=SourceType.COMPARISON,
            step=max_step + 1,
            tags=tags or ["comparison", "statistical", comparison_method],
            description=f"Comparison analysis using {comparison_method} method",
            metadata=comparison_metadata,
            lazy_init=True
        )
        
        # Set lineage and parent relationships
        comparison_channel.lineage_id = combined_lineage
        comparison_channel.parent_ids = [ch.channel_id for ch in parent_channels]
        comparison_channel.params = method_parameters
        
        return comparison_channel

    def with_lineage(self, lineage_id: str) -> "Channel":
        """Set the lineage ID and return self for method chaining"""
        self.lineage_id = lineage_id
        return self

    def with_parent(self, parent_id: str) -> "Channel":
        """Add a parent ID and return self for method chaining"""
        if parent_id not in self.parent_ids:
            self.parent_ids.append(parent_id)
        return self

    def get_config(self) -> dict:
        """Get channel configuration as a dictionary"""
        return {
            "channel_id": self.channel_id,
            "file_id": self.file_id,
            "filename": self.filename,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "legend_label": self.legend_label,
            "type": self.type.value if isinstance(self.type, SourceType) else self.type,
            "step": self.step,
            "tags": self.tags,
            "show": self.show,
            "yaxis": self.yaxis,
            "color": self.color,
            "style": self.style,
            "marker": self.marker,
            "parent_ids": self.parent_ids,
            "lineage_id": self.lineage_id,
            "file_status": self.file_status,
            "description": self.description,
            "params": self.params,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat()
        }

    def update_metadata(self, key: str, value: Any):
        """Safely update a single metadata field"""
        self.metadata[key] = value
        self.modified_at = datetime.now()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Safely get a metadata field value"""
        return self.metadata.get(key, default)

    def clear_metadata(self):
        """Clear all metadata"""
        self.metadata = {}
        self.modified_at = datetime.now()

    def get_memory_usage(self) -> dict:
        """Get approximate memory usage breakdown"""
        usage = {}
        
        if self._xdata is not None:
            usage['xdata_bytes'] = self._xdata.nbytes
        if self._ydata is not None:
            usage['ydata_bytes'] = self._ydata.nbytes
            
        usage['metadata_est_bytes'] = len(str(self.metadata).encode())
        usage['total_est_bytes'] = sum(usage.values())
        
        return usage

    def get_performance_stats(self) -> dict:
        """Get performance statistics for this channel"""
        return {
            'access_count': self._access_count,
            'last_access': self._last_access,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }

    @classmethod
    def clear_computation_cache(cls):
        """Clear the global computation cache"""
        cls._computation_cache.clear()
        cls._cache_hits = 0
        cls._cache_misses = 0

    @classmethod
    def get_cache_stats(cls) -> dict:
        """Get global cache statistics"""
        return {
            'cache_size': len(cls._computation_cache),
            'total_hits': cls._cache_hits,
            'total_misses': cls._cache_misses,
            'hit_rate': cls._cache_hits / (cls._cache_hits + cls._cache_misses) if (cls._cache_hits + cls._cache_misses) > 0 else 0
        }

    def __str__(self):
        return f"Channel {self.channel_id} (File: {self.file_id}, Type: {self.type}, Step: {self.step})"

    def __repr__(self):
        return f"Channel(id='{self.channel_id}', label='{self.legend_label}', step={self.step})"