import enum
import numpy as np
from typing import Optional, List, Any, Union, Callable, Dict
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib
import time
import uuid
import weakref
from functools import cached_property, lru_cache

# Import alignment classes - these should be defined in this file or imported from another module
class AlignmentMethod(Enum):
    """Alignment method enumeration"""
    INDEX = "index"
    TIME = "time"

@dataclass
class AlignmentConfig:
    """Configuration for data alignment"""
    method: AlignmentMethod
    mode: str
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    offset: float = 0.0
    interpolation: Optional[str] = None
    round_to: Optional[int] = None
    resolution: Optional[float] = None




@dataclass
class PairStats:
    """Lightweight container for computed pair statistics"""
    r_squared: Optional[float]
    correlation_coefficient: Optional[float]
    mean_difference: Optional[float]
    std_difference: Optional[float]
    bias: Optional[float]
    limits_of_agreement: Optional[tuple[float, float]]
    sample_count: int
    
    def __post_init__(self):
        """Validate stats after creation"""
        if self.sample_count <= 0:
            raise ValueError("Sample count must be positive")


class Pair:
    """
    Class to store information about a comparison pair between two data channels.
    
    Key features:
    - Visual properties for plotting
    - Metadata and configuration storage
    - Aligned data storage
    - Basic validation and statistics
    """
    
    def __init__(self, pair_id=None, name=None, ref_channel_id=None, test_channel_id=None,
                 ref_file_id=None, test_file_id=None, ref_channel_name=None, test_channel_name=None,
                 alignment_config=None, show=True, color=None, marker_type=None, marker_color=None, 
                 line_style="-", alpha=1.0, marker_size=50, legend_label=None, 
                 tags=None, description=None, metadata=None):
        
        # Core identifiers
        self.pair_id = pair_id or self._generate_pair_id()
        self.name = name or f"Pair_{self.pair_id[:8]}"
        self.description = description or ""
        
        # Channel references
        self.ref_channel_id = ref_channel_id
        self.test_channel_id = test_channel_id
        self.ref_file_id = ref_file_id
        self.test_file_id = test_file_id
        self.ref_channel_name = ref_channel_name
        self.test_channel_name = test_channel_name
        
        # Alignment configuration
        self.alignment_config = alignment_config or AlignmentConfig(AlignmentMethod.INDEX, "truncate")
        
        # Visual properties
        self.show = show
        self.color = color or self._get_default_color()
        self.marker_type = marker_type or "o"
        self.marker_color = marker_color or "🔵 Blue"
        self.line_style = line_style
        self.alpha = alpha
        self.marker_size = marker_size
        self.legend_label = legend_label or self.name
        
        # Metadata
        self.tags = tags or []
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.modified_at = self.created_at
        
        # Aligned data storage
        self.aligned_ref_data = None
        self.aligned_test_data = None
        self.alignment_metadata = {}
        
        # Computed properties
        self._stats = None
        self._data_hash = None

    @staticmethod
    def _generate_pair_id():
        """Generate a unique pair ID using timestamp and random bytes"""
        timestamp = str(time.time_ns())
        random_bytes = str(uuid.uuid4().hex[:8])
        combined = f"{timestamp}{random_bytes}"
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def _get_default_color(self):
        """Get a default color for this pair based on its ID"""
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # Use pair ID to get consistent color assignment
        color_index = hash(self.pair_id) % len(default_colors)
        return default_colors[color_index]

    def _invalidate_cache(self):
        """Invalidate cached properties when data changes"""
        self._stats = None
        self._data_hash = None
        self.modified_at = datetime.now()

    @cached_property
    def data_hash(self) -> Optional[str]:
        """Compute hash of pair configuration for change detection"""
        try:
            config_str = f"{self.ref_channel_id}:{self.test_channel_id}:{self.alignment_config.method.value}:{self.alignment_config.mode}"
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception as e:
            print(f"[Pair] Warning: Could not compute data hash: {e}")
            return None

    @cached_property
    def stats(self) -> Optional[PairStats]:
        """Compute pair statistics"""
        # For now, return basic stats - this would be computed by comparison methods
        return PairStats(
            r_squared=None,
            correlation_coefficient=None,
            mean_difference=None,
            std_difference=None,
            bias=None,
            limits_of_agreement=None,
            sample_count=0
        )

    def get_style_config(self) -> dict:
        """Get style configuration for this pair"""
        return {
            'color': self.color,
            'marker_type': self.marker_type,
            'marker_color': self.marker_color,
            'line_style': self.line_style,
            'alpha': self.alpha,
            'marker_size': self.marker_size,
            'legend_label': self.legend_label,
            'show': self.show
        }
    
    def update_style(self, style_config: dict):
        """Update style properties from configuration"""
        if 'color' in style_config:
            self.color = style_config['color']
        if 'marker_type' in style_config:
            self.marker_type = style_config['marker_type']
        if 'marker_color' in style_config:
            self.marker_color = style_config['marker_color']
        if 'line_style' in style_config:
            self.line_style = style_config['line_style']
        if 'alpha' in style_config:
            self.alpha = style_config['alpha']
        if 'marker_size' in style_config:
            self.marker_size = style_config['marker_size']
        if 'legend_label' in style_config:
            self.legend_label = style_config['legend_label']
        if 'show' in style_config:
            self.show = style_config['show']
        
        self.modified_at = datetime.now()
    
    def set_aligned_data(self, ref_data, test_data, metadata=None):
        """Set aligned data and automatically compute basic stats"""
        self.aligned_ref_data = ref_data
        self.aligned_test_data = test_data
        self.alignment_metadata = metadata or {}
        
        # Auto-compute basic stats when data is set
        if ref_data is not None and test_data is not None:
            self.compute_basic_stats()
        
        self.modified_at = datetime.now()

    def compute_basic_stats(self):
        """Compute basic comparison statistics from aligned data"""
        if not self.has_aligned_data():
            return None
            
        try:
            ref_data = self.aligned_ref_data
            test_data = self.aligned_test_data
            
            # Ensure data is not None and has proper length
            if ref_data is None or test_data is None:
                return None
                
            if len(ref_data) < 2 or len(test_data) < 2:
                return None
            
            # Basic statistics
            r_squared = self._calculate_r_squared(ref_data, test_data)
            correlation = self._calculate_correlation(ref_data, test_data)
            mean_diff = np.mean(test_data - ref_data)
            std_diff = np.std(test_data - ref_data)
            
            # Additional metrics
            bias = mean_diff  # Same as mean difference for Bland-Altman
            limits_of_agreement = (bias - 1.96 * std_diff, bias + 1.96 * std_diff)
            
            # Store in metadata for quick access
            self.metadata['basic_stats'] = {
                'r_squared': r_squared,
                'correlation': correlation,
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'bias': bias,
                'limits_of_agreement': limits_of_agreement,
                'sample_count': len(ref_data),
                'computed_at': datetime.now().isoformat()
            }
            
            print(f"[Pair] Computed basic stats for {self.name}: R²={r_squared:.3f}, r={correlation:.3f}")
            return self.metadata['basic_stats']
            
        except Exception as e:
            print(f"[Pair] Error computing basic stats for {self.name}: {e}")
            return None

    def _calculate_r_squared(self, ref_data, test_data):
        """Calculate R-squared value between two datasets"""
        try:
            if len(ref_data) < 2 or len(test_data) < 2:
                return None
                
            # Calculate correlation coefficient
            correlation = self._calculate_correlation(ref_data, test_data)
            if correlation is None:
                return None
                
            # R-squared is the square of correlation coefficient
            return correlation ** 2
            
        except Exception as e:
            print(f"[Pair] Error calculating R-squared: {e}")
            return None

    def _calculate_correlation(self, ref_data, test_data):
        """Calculate Pearson correlation coefficient between two datasets"""
        try:
            if len(ref_data) != len(test_data) or len(ref_data) < 2:
                return None
                
            # Check for constant data (zero variance)
            ref_std = np.std(ref_data)
            test_std = np.std(test_data)
            
            if ref_std == 0 or test_std == 0:
                return None
                
            # Calculate correlation
            correlation = np.corrcoef(ref_data, test_data)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                return None
                
            return correlation
            
        except Exception as e:
            print(f"[Pair] Error calculating correlation: {e}")
            return None

    @property
    def r_squared(self) -> Optional[float]:
        """Get R-squared value from computed stats"""
        stats = self.metadata.get('basic_stats')
        return stats.get('r_squared') if stats else None

    @property
    def correlation(self) -> Optional[float]:
        """Get correlation coefficient from computed stats"""
        stats = self.metadata.get('basic_stats')
        return stats.get('correlation') if stats else None

    @property
    def mean_difference(self) -> Optional[float]:
        """Get mean difference from computed stats"""
        stats = self.metadata.get('basic_stats')
        return stats.get('mean_difference') if stats else None

    @property
    def std_difference(self) -> Optional[float]:
        """Get standard deviation of difference from computed stats"""
        stats = self.metadata.get('basic_stats')
        return stats.get('std_difference') if stats else None

    @property
    def bias(self) -> Optional[float]:
        """Get bias (mean difference) from computed stats"""
        stats = self.metadata.get('basic_stats')
        return stats.get('bias') if stats else None

    @property
    def limits_of_agreement(self) -> Optional[tuple[float, float]]:
        """Get limits of agreement from computed stats"""
        stats = self.metadata.get('basic_stats')
        return stats.get('limits_of_agreement') if stats else None
    
    def get_aligned_data(self):
        """Get aligned data for this pair"""
        return {
            'ref_data': self.aligned_ref_data,
            'test_data': self.aligned_test_data,
            'metadata': self.alignment_metadata
        }
    
    def has_aligned_data(self):
        """Check if this pair has aligned data"""
        return (self.aligned_ref_data is not None and 
                self.aligned_test_data is not None)
    
    def clear_aligned_data(self):
        """Clear aligned data (e.g., when alignment config changes)"""
        self.aligned_ref_data = None
        self.aligned_test_data = None
        self.alignment_metadata = {}
        self.modified_at = datetime.now()
    
    def get_display_name(self) -> str:
        """Get formatted display name for the pair"""
        if self.name and self.name != f"Pair_{self.pair_id[:8]}":
            return self.name
        elif self.ref_channel_name and self.test_channel_name:
            return f"{self.ref_channel_name} vs {self.test_channel_name}"
        else:
            return f"Pair {self.pair_id[:8]}"

    def get_tooltip_text(self) -> str:
        """Get detailed tooltip text for the pair"""
        lines = [
            f"Name: {self.name}",
            f"Reference: {self.ref_channel_name} ({self.ref_file_id})",
            f"Test: {self.test_channel_name} ({self.test_file_id})",
            f"Alignment: {self.alignment_config.method.value} - {self.alignment_config.mode}",
        ]
        
        if self.description:
            lines.append(f"Description: {self.description}")
            
        return "\n".join(lines)

    def validate(self) -> List[str]:
        """Validate pair configuration and return list of errors"""
        errors = []
        
        # Check required fields
        if not self.ref_channel_id:
            errors.append("Reference channel ID is required")
        if not self.test_channel_id:
            errors.append("Test channel ID is required")
        if not self.ref_file_id:
            errors.append("Reference file ID is required")
        if not self.test_file_id:
            errors.append("Test file ID is required")
            
        # Check for self-comparison
        if (self.ref_channel_id == self.test_channel_id and 
            self.ref_file_id == self.test_file_id):
            errors.append("Cannot compare channel to itself")
            
        # Validate alignment config
        try:
            if isinstance(self.alignment_config, dict):
                # Convert dict to AlignmentConfig object
                self.alignment_config = AlignmentConfig(**self.alignment_config)
        except Exception as e:
            errors.append(f"Invalid alignment configuration: {e}")
            
        return errors

    def get_config(self) -> dict:
        """Export pair configuration as dictionary"""
        return {
            'pair_id': self.pair_id,
            'name': self.name,
            'description': self.description,
            'ref_channel_id': self.ref_channel_id,
            'test_channel_id': self.test_channel_id,
            'ref_file_id': self.ref_file_id,
            'test_file_id': self.test_file_id,
            'ref_channel_name': self.ref_channel_name,
            'test_channel_name': self.test_channel_name,
            'alignment_config': {
                'method': self.alignment_config.method.value if isinstance(self.alignment_config.method, AlignmentMethod) else self.alignment_config.method,
                'mode': self.alignment_config.mode,
                'start_index': self.alignment_config.start_index,
                'end_index': self.alignment_config.end_index,
                'start_time': self.alignment_config.start_time,
                'end_time': self.alignment_config.end_time,
                'offset': self.alignment_config.offset,
                'interpolation': self.alignment_config.interpolation,
                'round_to': self.alignment_config.round_to
            },
            'show': self.show,
            'color': self.color,
            'marker_type': self.marker_type,
            'marker_color': self.marker_color,
            'line_style': self.line_style,
            'alpha': self.alpha,
            'marker_size': self.marker_size,
            'legend_label': self.legend_label,
            'tags': self.tags,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat()
        }

    @classmethod
    def from_config(cls, config: dict) -> 'Pair':
        """Create Pair object from configuration dictionary"""
        # Convert alignment config
        alignment_config = config.get('alignment_config', {})
        if isinstance(alignment_config, dict):
            method = alignment_config.get('method')
            if isinstance(method, str):
                try:
                    method = AlignmentMethod(method)
                except ValueError:
                    method = AlignmentMethod.INDEX
            alignment_config['method'] = method
            alignment_config = AlignmentConfig(**alignment_config)
        
        # Convert timestamps
        created_at = config.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()
            
        modified_at = config.get('modified_at')
        if isinstance(modified_at, str):
            modified_at = datetime.fromisoformat(modified_at)
        else:
            modified_at = datetime.now()
        
        return cls(
            pair_id=config.get('pair_id'),
            name=config.get('name'),
            description=config.get('description'),
            ref_channel_id=config.get('ref_channel_id'),
            test_channel_id=config.get('test_channel_id'),
            ref_file_id=config.get('ref_file_id'),
            test_file_id=config.get('test_file_id'),
            ref_channel_name=config.get('ref_channel_name'),
            test_channel_name=config.get('test_channel_name'),
            alignment_config=alignment_config,
            show=config.get('show', True),
            color=config.get('color'),
            marker_type=config.get('marker_type'),
            marker_color=config.get('marker_color'),
            line_style=config.get('line_style', '-'),
            alpha=config.get('alpha', 1.0),
            marker_size=config.get('marker_size', 50),
            legend_label=config.get('legend_label'),
            tags=config.get('tags', []),
            metadata=config.get('metadata', {}),
        )

    def update_metadata(self, key: str, value: Any):
        """Update metadata with new key-value pair"""
        self.metadata[key] = value
        self.modified_at = datetime.now()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key"""
        return self.metadata.get(key, default)

    def clear_metadata(self):
        """Clear all metadata"""
        self.metadata.clear()
        self.modified_at = datetime.now()

    def get_memory_usage(self) -> dict:
        """Get memory usage information"""
        return {
            'pair_id': self.pair_id,
            'name_size': len(self.name) if self.name else 0,
            'description_size': len(self.description) if self.description else 0,
            'alignment_config_size': len(str(self.alignment_config)),
            'tags_size': sum(len(tag) for tag in self.tags),
            'metadata_size': len(str(self.metadata)),
            'access_count': 0 # Removed performance tracking
        }

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'pair_id': self.pair_id,
            'access_count': 0, # Removed performance tracking
            'last_access': None, # Removed performance tracking
            'created_at': self.created_at,
            'modified_at': self.modified_at
        }

    @classmethod
    def clear_computation_cache(cls):
        """Clear class-level computation cache"""
        # Removed cache infrastructure
        pass

    @classmethod
    def get_cache_stats(cls) -> dict:
        """Get cache statistics"""
        return {
            'cache_size': 0, # Removed cache infrastructure
            'cache_hits': 0, # Removed cache infrastructure
            'cache_misses': 0, # Removed cache infrastructure
            'hit_rate': 0 # Removed cache infrastructure
        }

    def __str__(self):
        """String representation"""
        return f"Pair({self.get_display_name()})"

    def __repr__(self):
        """Detailed string representation"""
        return f"Pair(pair_id='{self.pair_id}', name='{self.name}', alignment='{self.alignment_config.method.value}')" 