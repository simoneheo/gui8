from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class ParseStrategy(Enum):
    """Available parsing strategies in order of preference."""
    PANDAS_AUTO = "pandas_auto"
    DELIMITER_DETECTION = "delimiter_detection" 
    MANUAL_PARSING = "manual_parsing"


class DataType(Enum):
    """Data type classifications."""
    NUMERIC = "numeric"
    DATETIME = "datetime"
    TEXT = "text"
    MIXED = "mixed"


@dataclass
class ColumnInfo:
    """Information about a parsed column."""
    name: str
    data_type: DataType
    is_time_column: bool = False
    row_count: int = 0
    null_count: int = 0
    sample_values: List[Any] = field(default_factory=list)
    
    @property
    def null_ratio(self) -> float:
        """Ratio of null values."""
        return self.null_count / self.row_count if self.row_count > 0 else 0.0
    
    @property
    def data_quality(self) -> str:
        """Data quality assessment."""
        if self.null_ratio < 0.05:
            return "High"
        elif self.null_ratio < 0.20:
            return "Medium"
        else:
            return "Low"


@dataclass
class ParseResult:
    """Results from parsing operation."""
    success: bool = False
    strategy_used: Optional[ParseStrategy] = None
    rows_parsed: int = 0
    columns_found: int = 0
    channels_created: int = 0
    parse_time_ms: float = 0.0
    error_message: Optional[str] = None
    encoding_detected: str = "utf-8"
    metadata_lines_skipped: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'success': self.success,
            'strategy_used': self.strategy_used.value if self.strategy_used else None,
            'rows_parsed': self.rows_parsed,
            'columns_found': self.columns_found,
            'channels_created': self.channels_created,
            'parse_time_ms': self.parse_time_ms,
            'error_message': self.error_message,
            'encoding_detected': self.encoding_detected,
            'metadata_lines_skipped': self.metadata_lines_skipped
        }


@dataclass
class ParseConfig:
    """Configuration for parsing operations."""
    
    # Parsing limits
    max_preview_lines: int = 1000
    max_sample_lines: int = 20
    min_confidence_threshold: float = 0.5
    
    # Metadata detection
    metadata_prefixes: List[str] = field(default_factory=lambda: [
        '#', '//', '--', '%', ';', '"""', ';;;', '>', '::'
    ])
    
    # Time column detection
    time_column_indicators: List[str] = field(default_factory=lambda: [
        'time', 'timestamp', 'datetime', 'date', 'ms', 'sec', 'seconds',
        'hour', 'minute', 'epoch', 'utc', 'gmt', 'clock', 't', 'x',
        'index', 'idx', 'sample', 'n', 'step'
    ])
    
    time_column_patterns: List[str] = field(default_factory=lambda: [
        r'time\(?s?\)?', r'time_', r'_time', r't_\d*', r'x_?\d*'
    ])
    
    # Data quality thresholds
    min_numeric_ratio: float = 0.7  # Minimum ratio of numeric values in a column
    min_data_ratio: float = 0.1     # Minimum ratio of non-null data
    
    # Parsing attempts
    encoding_attempts: List[str] = field(default_factory=lambda: [
        'utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii', 'iso-8859-1'
    ])
    
    delimiter_candidates: List[str] = field(default_factory=lambda: [
        ',', '\t', ';', '|', ' '
    ])
    
    # Results storage
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parse_result: Optional[ParseResult] = None
    
    def add_column_info(self, name: str, data_type: DataType, 
                       is_time_column: bool = False, row_count: int = 0,
                       null_count: int = 0, sample_values: List[Any] = None):
        """Add information about a parsed column."""
        self.columns[name] = ColumnInfo(
            name=name,
            data_type=data_type,
            is_time_column=is_time_column,
            row_count=row_count,
            null_count=null_count,
            sample_values=sample_values or []
        )
    
    def get_time_columns(self) -> List[str]:
        """Get list of columns identified as time columns."""
        return [name for name, info in self.columns.items() if info.is_time_column]
    
    def get_data_columns(self) -> List[str]:
        """Get list of columns identified as data (non-time) columns."""
        return [name for name, info in self.columns.items() if not info.is_time_column]
    
    def set_parse_result(self, result: ParseResult):
        """Set the parsing result."""
        self.parse_result = result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the parsing configuration and results."""
        summary = {
            'total_columns': len(self.columns),
            'time_columns': len(self.get_time_columns()),
            'data_columns': len(self.get_data_columns()),
            'column_quality': {}
        }
        
        # Add column quality information
        for name, info in self.columns.items():
            summary['column_quality'][name] = {
                'data_type': info.data_type.value,
                'quality': info.data_quality,
                'null_ratio': info.null_ratio,
                'is_time_column': info.is_time_column
            }
        
        # Add parse result if available
        if self.parse_result:
            summary['parse_result'] = self.parse_result.to_dict()
        
        return summary
    
    def reset(self):
        """Reset configuration for new parsing operation."""
        self.columns.clear()
        self.metadata.clear()
        self.parse_result = None


# Global default configuration instance
DEFAULT_PARSE_CONFIG = ParseConfig()


def create_parse_config(**kwargs) -> ParseConfig:
    """Create a new ParseConfig with optional parameter overrides."""
    config = ParseConfig()
    
    # Update with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config 