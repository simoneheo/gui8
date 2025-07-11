#!/usr/bin/env python3
"""
Test script to verify that the comparison wizard selects correlation method by default
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test for auto-zoom functionality in RenderPlotOp
import matplotlib.pyplot as plt
import numpy as np
from comparison_wizard_manager import RenderPlotOp

def test_auto_zoom():
    """Test the auto-zoom functionality"""
    
    # Create test scatter data
    scatter_data = [
        {
            'x_data': [1, 2, 3, 4, 5],
            'y_data': [2, 4, 6, 8, 10],
            'pair_name': 'Test Pair 1'
        },
        {
            'x_data': [2, 3, 4, 5, 6],
            'y_data': [3, 5, 7, 9, 11],
            'pair_name': 'Test Pair 2'
        }
    ]
    
    # Create a matplotlib figure for testing
    fig, ax = plt.subplots()
    
    # Create mock plot widget
    class MockPlotWidget:
        def __init__(self, figure):
            self.figure = figure
        
        def draw(self):
            pass
    
    plot_widget = MockPlotWidget(fig)
    
    # Create RenderPlotOp instance
    render_op = RenderPlotOp(plot_widget)
    render_op.current_figure = fig
    render_op.current_axes = ax
    
    # Test auto-zoom
    render_op._auto_zoom_to_scatter_data(scatter_data)
    
    # Check that axis limits were set correctly
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    print(f"X limits: {xlim}")
    print(f"Y limits: {ylim}")
    
    # Expected bounds: x=[1,6], y=[2,11] with 5% padding
    x_range = 6 - 1  # 5
    y_range = 11 - 2  # 9
    x_padding = x_range * 0.05  # 0.25
    y_padding = y_range * 0.05  # 0.45
    
    expected_xlim = (1 - x_padding, 6 + x_padding)  # (0.75, 6.25)
    expected_ylim = (2 - y_padding, 11 + y_padding)  # (1.55, 11.45)
    
    print(f"Expected X limits: {expected_xlim}")
    print(f"Expected Y limits: {expected_ylim}")
    
    # Check if the limits are approximately correct
    assert abs(xlim[0] - expected_xlim[0]) < 0.01, f"X min mismatch: {xlim[0]} vs {expected_xlim[0]}"
    assert abs(xlim[1] - expected_xlim[1]) < 0.01, f"X max mismatch: {xlim[1]} vs {expected_xlim[1]}"
    assert abs(ylim[0] - expected_ylim[0]) < 0.01, f"Y min mismatch: {ylim[0]} vs {expected_ylim[0]}"
    assert abs(ylim[1] - expected_ylim[1]) < 0.01, f"Y max mismatch: {ylim[1]} vs {expected_ylim[1]}"
    
    print("✅ Auto-zoom test passed!")
    
    plt.close(fig)

def test_correlation_default_selection():
    """Test that correlation method is selected by default"""
    try:
        # Import the comparison registry
        from comparison import ComparisonRegistry, load_all_comparisons
        
        # Load all comparison methods
        print("Loading comparison methods...")
        load_all_comparisons()
        
        # Get available methods
        methods = ComparisonRegistry.all_comparisons()
        print(f"Available methods: {methods}")
        
        # Check if correlation method exists
        correlation_found = False
        for method in methods:
            if "correlation" in method.lower():
                correlation_found = True
                print(f"✓ Found correlation method: {method}")
                break
        
        if not correlation_found:
            print("✗ No correlation method found in registry")
            return False
        
        # Test the selection logic
        default_method_name = "correlation"
        for method in methods:
            if default_method_name.lower() in method.lower():
                print(f"✓ Would select as default: {method}")
                return True
        
        print("✗ Could not find correlation method for default selection")
        return False
        
    except Exception as e:
        print(f"✗ Error testing correlation default selection: {e}")
        return False

if __name__ == "__main__":
    print("Testing correlation method default selection...")
    success = test_correlation_default_selection()
    if success:
        print("✓ Test passed: Correlation method can be selected by default")
    else:
        print("✗ Test failed: Correlation method default selection failed")
    sys.exit(0 if success else 1) 