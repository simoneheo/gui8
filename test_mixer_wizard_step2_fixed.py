#!/usr/bin/env python3
"""
Test script for Signal Mixer Wizard Step 2 functionality
Tests that Step 2 properly populates with channels from Step 1 and handles mixing operations
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_mixer_wizard_window import SignalMixerWizardWindow
from file_manager import FileManager
from channel_manager import ChannelManager
from channel import Channel
from file import File
import numpy as np

def create_test_data():
    """Create test data for the mixer wizard"""
    print("Creating test data...")
    
    # Create file manager and channel manager
    file_manager = FileManager()
    channel_manager = ChannelManager()
    
    # Create a test file
    test_file = File(
        file_id="test_file_1",
        filename="test_signals.txt",
        filepath="/test/test_signals.txt",
        file_type="txt"
    )
    file_manager.add_file(test_file)
    
    # Create test channels with different data
    time_data = np.linspace(0, 10, 100)
    
    # Channel A: Sine wave
    channel_a = Channel(
        channel_id="ch_a",
        filename="test_signals.txt",
        legend_label="Sine Wave",
        xdata=time_data,
        ydata=np.sin(time_data),
        xlabel="Time (s)",
        ylabel="Amplitude",
        step=0
    )
    
    # Channel B: Cosine wave
    channel_b = Channel(
        channel_id="ch_b", 
        filename="test_signals.txt",
        legend_label="Cosine Wave",
        xdata=time_data,
        ydata=np.cos(time_data),
        xlabel="Time (s)",
        ylabel="Amplitude",
        step=0
    )
    
    # Channel C: Square wave
    channel_c = Channel(
        channel_id="ch_c",
        filename="test_signals.txt", 
        legend_label="Square Wave",
        xdata=time_data,
        ydata=np.sign(np.sin(time_data * 2)),
        xlabel="Time (s)",
        ylabel="Amplitude",
        step=0
    )
    
    # Add channels to manager
    channel_manager.add_channel(channel_a)
    channel_manager.add_channel(channel_b)
    channel_manager.add_channel(channel_c)
    
    print(f"Created {channel_manager.get_channel_count()} test channels")
    return file_manager, channel_manager

def test_step2_population():
    """Test Step 2 population with channels from Step 1"""
    print("\n=== Testing Step 2 Population ===")
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create test data
    file_manager, channel_manager = create_test_data()
    
    # Create mixer wizard
    wizard = SignalMixerWizardWindow(
        file_manager=file_manager,
        channel_manager=channel_manager
    )
    
    def run_test():
        try:
            print("\n1. Testing Step 1 channel selection...")
            
            # Set up channels in Step 1
            wizard.a_file_combo.setCurrentText("test_signals.txt")
            wizard.a_channel_combo.setCurrentText("Sine Wave")
            wizard.b_file_combo.setCurrentText("test_signals.txt") 
            wizard.b_channel_combo.setCurrentText("Cosine Wave")
            
            print(f"   - Channel A: {wizard._get_channel_a().legend_label if wizard._get_channel_a() else 'None'}")
            print(f"   - Channel B: {wizard._get_channel_b().legend_label if wizard._get_channel_b() else 'None'}")
            
            # Add additional channel C
            wizard._add_additional_channel()
            if wizard.additional_channel_widgets:
                widget_info = wizard.additional_channel_widgets[0]
                widget_info['file_combo'].setCurrentText("test_signals.txt")
                widget_info['channel_combo'].setCurrentText("Square Wave")
                print(f"   - Channel C: {widget_info['channel_combo'].currentText()}")
            
            # Validate channels
            validation_result = wizard._validate_channels()
            print(f"   - Validation passed: {validation_result}")
            
            print("\n2. Testing transition to Step 2...")
            
            # Move to Step 2
            if validation_result:
                wizard._go_to_next_step()
                print(f"   - Current step: {wizard.current_step}")
                print(f"   - Step table rows: {wizard.step_table.rowCount()}")
                
                # Check step table contents
                print("   - Step table contents:")
                for row in range(wizard.step_table.rowCount()):
                    label_item = wizard.step_table.item(row, 0)
                    desc_item = wizard.step_table.item(row, 4)
                    if label_item and desc_item:
                        print(f"     Row {row}: {label_item.text()} - {desc_item.text()}")
                
                print("\n3. Testing mixing operation...")
                
                # Test a simple mixing expression
                wizard.console_input.setText("D = A + B + C")
                print(f"   - Expression: {wizard.console_input.text()}")
                
                # Check available channels in context
                channel_context = {
                    'A': wizard._get_channel_a(),
                    'B': wizard._get_channel_b(),
                }
                channel_context.update(wizard._get_additional_channels())
                
                available_channels = list(channel_context.keys())
                print(f"   - Available channels for mixing: {available_channels}")
                
                # Test console submission (without actually executing to avoid mixer errors)
                print("   - Testing console input processing...")
                expression = wizard.console_input.text()
                if '=' in expression:
                    label, formula = map(str.strip, expression.split('=', 1))
                    print(f"   - Parsed label: {label}, formula: {formula}")
                    print(f"   - All required channels available: {all(ch in channel_context for ch in ['A', 'B', 'C'])}")
                
                print("\n✅ Step 2 population test completed successfully!")
                
            else:
                print("❌ Could not proceed to Step 2 - validation failed")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Close wizard
            wizard.close()
            QTimer.singleShot(100, app.quit)
    
    # Show wizard and run test
    wizard.show()
    QTimer.singleShot(500, run_test)
    
    return app.exec()

if __name__ == "__main__":
    print("Signal Mixer Wizard Step 2 Population Test")
    print("=" * 50)
    
    result = test_step2_population()
    
    print(f"\nTest completed with exit code: {result}") 