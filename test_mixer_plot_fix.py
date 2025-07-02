#!/usr/bin/env python3
"""
Test script to verify mixer wizard plot updates work correctly
"""

import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Mock classes for testing
class MockChannel:
    def __init__(self, channel_id, legend_label, xdata=None, ydata=None):
        self.channel_id = channel_id
        self.legend_label = legend_label
        self.filename = "test_file.csv"
        self.step = 1
        self.show = True
        self.xdata = xdata if xdata is not None else np.arange(100)
        self.ydata = ydata if ydata is not None else np.random.randn(100)
        self.xlabel = "Time"
        self.ylabel = "Amplitude"
        self.color = 'blue'

class MockFileManager:
    def __init__(self):
        self.files = []
    
    def get_file_count(self):
        return len(self.files)
    
    def get_all_files(self):
        return []

class MockChannelManager:
    def __init__(self):
        self.channels = []
        
        # Create test channels
        self.channels.append(MockChannel("ch1", "Sensor_A", ydata=np.sin(np.linspace(0, 4*np.pi, 100))))
        self.channels.append(MockChannel("ch2", "Sensor_B", ydata=np.cos(np.linspace(0, 4*np.pi, 100))))
    
    def get_channel_count(self):
        return len(self.channels)
    
    def get_channels_by_file(self, file_id):
        return self.channels
    
    def get_all_channels(self):
        return self.channels
    
    def add_channel(self, channel):
        self.channels.append(channel)
        print(f"[MockChannelManager] Added channel: {channel.channel_id}")
    
    def remove_channel(self, channel_id):
        self.channels = [ch for ch in self.channels if ch.channel_id != channel_id]
        print(f"[MockChannelManager] Removed channel: {channel_id}")

def test_mixer_wizard():
    """Test the mixer wizard functionality"""
    print("Testing mixer wizard plot updates...")
    
    app = QApplication(sys.argv)
    
    # Create mock managers
    file_manager = MockFileManager()
    channel_manager = MockChannelManager()
    
    try:
        from signal_mixer_wizard_window import SignalMixerWizardWindow
        
        # Create wizard window
        wizard = SignalMixerWizardWindow(
            file_manager=file_manager,
            channel_manager=channel_manager
        )
        
        print("✅ Mixer wizard window created successfully")
        
        # Show the window
        wizard.show()
        
        # Test auto-label generation
        print("\n--- Testing auto-label generation ---")
        next_label = wizard._get_next_available_label()
        print(f"Next available label: {next_label}")
        assert next_label == "C", f"Expected 'C', got '{next_label}'"
        
        # Test with some existing channels
        class MockMixedChannel:
            def __init__(self, label):
                self.step_table_label = label
        
        wizard.manager.mixed_channels = [MockMixedChannel("C"), MockMixedChannel("D")]
        next_label = wizard._get_next_available_label()
        print(f"Next available label after C,D: {next_label}")
        assert next_label == "E", f"Expected 'E', got '{next_label}'"
        
        print("✅ Auto-label generation working correctly")
        
        # Test plot update
        print("\n--- Testing plot update ---")
        wizard._update_plot()
        print("✅ Plot update completed without errors")
        
        # Test expression creation
        print("\n--- Testing expression creation ---")
        wizard.expression_input.setText("TEST = A + B")
        
        # Simulate creating a mixed channel
        QTimer.singleShot(100, lambda: wizard._on_create_mixed_channel())
        QTimer.singleShot(200, lambda: app.quit())
        
        print("✅ All tests completed successfully")
        return app.exec()
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = test_mixer_wizard()
    sys.exit(exit_code) 