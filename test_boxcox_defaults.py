#!/usr/bin/env python3
"""
Test script to verify Box-Cox intelligent defaults work correctly
"""

import numpy as np
import sys
import os

# Add the steps directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'steps'))

from default_config import _get_boxcox_transform_defaults, _get_boxcox_shift_default

class MockChannel:
    def __init__(self, ydata):
        self.ydata = ydata
        self.fs_median = 100.0

def test_boxcox_defaults():
    """Test Box-Cox intelligent defaults with various signal types"""
    
    print("Testing Box-Cox intelligent defaults...")
    
    # Test 1: Signal with negative values
    print("\nTest 1: Signal with negative values")
    y_negative = np.array([-0.5, -0.2, 0.1, 0.3, 0.8])
    channel_negative = MockChannel(y_negative)
    
    defaults = _get_boxcox_transform_defaults(100.0, len(y_negative), channel_negative)
    print(f"Signal: {y_negative}")
    print(f"Defaults: {defaults}")
    
    shift_value = _get_boxcox_shift_default(channel_negative)
    print(f"Shift value: {shift_value}")
    
    # Test 2: Signal with all positive values
    print("\nTest 2: Signal with all positive values")
    y_positive = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
    channel_positive = MockChannel(y_positive)
    
    defaults = _get_boxcox_transform_defaults(100.0, len(y_positive), channel_positive)
    print(f"Signal: {y_positive}")
    print(f"Defaults: {defaults}")
    
    shift_value = _get_boxcox_shift_default(channel_positive)
    print(f"Shift value: {shift_value}")
    
    # Test 3: Signal with zero values
    print("\nTest 3: Signal with zero values")
    y_zero = np.array([0.0, 0.1, 0.3, 0.5, 0.8])
    channel_zero = MockChannel(y_zero)
    
    defaults = _get_boxcox_transform_defaults(100.0, len(y_zero), channel_zero)
    print(f"Signal: {y_zero}")
    print(f"Defaults: {defaults}")
    
    shift_value = _get_boxcox_shift_default(channel_zero)
    print(f"Shift value: {shift_value}")
    
    # Test 4: Signal with mixed positive/negative values (like the error case)
    print("\nTest 4: Signal with mixed values (like the error case)")
    y_mixed = np.array([-0.39703407561556314, 0.1, 0.3, 0.5, 0.8])
    channel_mixed = MockChannel(y_mixed)
    
    defaults = _get_boxcox_transform_defaults(100.0, len(y_mixed), channel_mixed)
    print(f"Signal: {y_mixed}")
    print(f"Defaults: {defaults}")
    
    shift_value = _get_boxcox_shift_default(channel_mixed)
    print(f"Shift value: {shift_value}")
    
    # Verify the shift would work
    min_val = np.min(y_mixed)
    suggested_shift = abs(min_val) + 1e-6
    print(f"Minimum value: {min_val}")
    print(f"Suggested shift: {suggested_shift}")
    print(f"Shifted signal would be: {y_mixed + suggested_shift}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_boxcox_defaults() 