"""
Base Comparison Class

This module defines the base class for all comparison methods, providing a consistent
interface and common functionality for data comparison operations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import warnings

class BaseComparison(ABC):
    """
    Abstract base class for all comparison methods.
    
    This class defines the interface that all comparison methods must implement,
    ensuring consistency across different comparison techniques.
    """
    
    # Class attributes that subclasses should override
    name = "Base Comparison"
    description = "Base class for comparison methods"
    category = "Base"
    version = "1.0.0"
    
    # Parameters that the comparison method accepts
    parameters = {}
    
    # Output types this comparison produces
    output_types = ["statistics", "plot_data"]
    
    # Plot configuration
    plot_type = "scatter"  # Default plot type
    requires_pairs = False  # Whether this comparison needs pair data instead of combined data
    
    def __init__(self, **kwargs):
        """
        Initialize the comparison method with parameters.
        
        Args:
            **kwargs: Parameter values for the comparison method
        """
        self.params = self._validate_parameters(kwargs)
        self.results = {}
        self.metadata = {
            'method': self.name,
            'version': self.version,
            'parameters': self.params.copy()
        }
        
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set default values for parameters.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Dictionary of validated parameters with defaults
        """
        validated = {}
        
        for param_name, param_config in self.parameters.items():
            if param_name in params:
                value = params[param_name]
                # Type validation
                expected_type = param_config.get('type', str)
                if not isinstance(value, expected_type):
                    try:
                        value = expected_type(value)
                    except (ValueError, TypeError):
                        raise ValueError(f"Parameter '{param_name}' must be of type {expected_type.__name__}")
                
                # Range validation
                if 'min' in param_config and value < param_config['min']:
                    raise ValueError(f"Parameter '{param_name}' must be >= {param_config['min']}")
                if 'max' in param_config and value > param_config['max']:
                    raise ValueError(f"Parameter '{param_name}' must be <= {param_config['max']}")
                
                # Choices validation
                if 'choices' in param_config and value not in param_config['choices']:
                    raise ValueError(f"Parameter '{param_name}' must be one of {param_config['choices']}")
                
                validated[param_name] = value
            else:
                # Use default value
                if 'default' in param_config:
                    validated[param_name] = param_config['default']
                elif param_config.get('required', False):
                    raise ValueError(f"Required parameter '{param_name}' not provided")
        
        return validated
    
    @abstractmethod
    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform the comparison between reference and test data.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing comparison results
        """
        pass
    
    def _validate_input_data(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and clean input data arrays.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            
        Returns:
            Tuple of cleaned reference and test data arrays
        """
        # Convert to numpy arrays
        ref_data = np.asarray(ref_data)
        test_data = np.asarray(test_data)
        
        # Check dimensions
        if ref_data.ndim != 1:
            raise ValueError("Reference data must be 1-dimensional")
        if test_data.ndim != 1:
            raise ValueError("Test data must be 1-dimensional")
        
        # Check length compatibility
        if len(ref_data) != len(test_data):
            warnings.warn(f"Data arrays have different lengths: ref={len(ref_data)}, test={len(test_data)}")
            # Truncate to shorter length
            min_len = min(len(ref_data), len(test_data))
            ref_data = ref_data[:min_len]
            test_data = test_data[:min_len]
        
        # Check for empty arrays
        if len(ref_data) == 0:
            raise ValueError("Data arrays are empty")
        
        return ref_data, test_data
    
    def _remove_invalid_data(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Remove NaN and infinite values from data arrays.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            
        Returns:
            Tuple of (cleaned_ref_data, cleaned_test_data, valid_ratio)
        """
        # Find valid data points
        valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
        
        # Calculate valid data ratio
        valid_ratio = np.sum(valid_mask) / len(valid_mask)
        
        # Filter data
        ref_clean = ref_data[valid_mask]
        test_clean = test_data[valid_mask]
        
        if len(ref_clean) == 0:
            raise ValueError("No valid data points found after removing NaN/infinite values")
        
        if valid_ratio < 0.5:
            warnings.warn(f"Only {valid_ratio*100:.1f}% of data points are valid")
        
        return ref_clean, test_clean, valid_ratio
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """
        Get information about the parameters this comparison method accepts.
        
        Returns:
            Dictionary with parameter information
        """
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'parameters': self.parameters,
            'output_types': self.output_types
        }
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the results from the last comparison operation.
        
        Returns:
            Dictionary containing comparison results
        """
        return self.results.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the comparison method and last operation.
        
        Returns:
            Dictionary containing metadata
        """
        return self.metadata.copy()
    
    def generate_plot_content(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any] = None, 
                             checked_pairs: List[Dict[str, Any]] = None) -> None:
        """
        Generate plot content for this comparison method.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array (combined from all pairs)
            test_data: Test data array (combined from all pairs)
            plot_config: Plot configuration dictionary
            checked_pairs: List of checked pair configurations (for methods that need pair-specific data)
        """
        # Default implementation - basic scatter plot
        try:
            if len(ref_data) == 0:
                ax.text(0.5, 0.5, f'No valid data for {self.name}', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Add identity line for reference
            min_val = min(np.min(ref_data), np.min(test_data))
            max_val = max(np.max(ref_data), np.max(test_data))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Agreement')
            
            # Basic plot title and labels
            ax.set_xlabel('Reference')
            ax.set_ylabel('Test')
            ax.set_title(f'{self.name} Analysis')
            
        except Exception as e:
            print(f"[{self.name}] Error generating plot: {e}")
            ax.text(0.5, 0.5, f'Error generating {self.name} plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def __str__(self) -> str:
        """String representation of the comparison method"""
        return f"{self.name} (v{self.version})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"<{self.__class__.__name__}: {self.name}>" 