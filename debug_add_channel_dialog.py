#!/usr/bin/env python3
# Debug script to identify the dialog popup issue when adding channels

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from PySide6.QtWidgets import QApplication
from channel import Channel
from file_manager import FileManager  
from channel_manager import ChannelManager
from signal_mixer_wizard_window import SignalMixerWizardWindow
from file import File


def create_test_data():
    """Create test data with multiple channels"""
    # Generate test data
    time = np.linspace(0, 10, 1000)
    
    # File 1 with 3 channels
    data1 = {
        'time': time,
        'ch1': np.sin(2 * np.pi * 0.5 * time) + 0.1 * np.random.randn(len(time)),
        'ch2': np.cos(2 * np.pi * 0.7 * time) + 0.1 * np.random.randn(len(time)),
        'ch3': np.sin(2 * np.pi * 1.2 * time) * np.exp(-time/5) + 0.1 * np.random.randn(len(time))
    }
    
    # File 2 with 2 channels
    data2 = {
        'time': time[:800],  # Different length
        'ch1': np.square(np.sin(2 * np.pi * 0.3 * time[:800])) + 0.1 * np.random.randn(800),
        'ch2': sawtooth(2 * np.pi * 0.8 * time[:800]) + 0.1 * np.random.randn(800)
    }
    
    return data1, data2


def create_test_files_and_channels(file_manager, channel_manager):
    """Create test files and channels"""
    data1, data2 = create_test_data()
    
    # Create File 1
    file1 = File(
        filename="test_signal_1.dat",
        filepath="/tmp/test_signal_1.dat",
        file_id="file1"
    )
    file_manager.add_file(file1)
    
    # Create channels for File 1
    for i, (ch_name, ch_data) in enumerate([('ch1', data1['ch1']), ('ch2', data1['ch2']), ('ch3', data1['ch3'])]):
        channel = Channel(
            channel_id=f"file1_{ch_name}",
            filename="test_signal_1.dat",
            legend_label=f"Signal 1 - {ch_name.upper()}",
            xdata=data1['time'],
            ydata=ch_data,
            xlabel="Time (s)",
            ylabel="Amplitude (V)",
            step=0
        )
        channel_manager.add_channel(channel)
    
    # Create File 2
    file2 = File(
        filename="test_signal_2.dat",
        filepath="/tmp/test_signal_2.dat",
        file_id="file2"
    )
    file_manager.add_file(file2)
    
    # Create channels for File 2
    for i, (ch_name, ch_data) in enumerate([('ch1', data2['ch1']), ('ch2', data2['ch2'])]):
        channel = Channel(
            channel_id=f"file2_{ch_name}",
            filename="test_signal_2.dat",
            legend_label=f"Signal 2 - {ch_name.upper()}",
            xdata=data2['time'],
            ydata=ch_data,
            xlabel="Time (s)",
            ylabel="Amplitude (V)",
            step=0
        )
        channel_manager.add_channel(channel)
    
    print("Created test files and channels:")
    print(f"- File 1: {len(data1['ch1'])} samples, 3 channels")
    print(f"- File 2: {len(data2['ch1'])} samples, 2 channels")


def main():
    """Main debug function"""
    print("=== Debug Add Channel Dialog Issue ===")
    print("This script will help identify what's causing the dialog popup.")
    print("Watch the console output for any error messages when clicking 'Add Channel'.")
    print()
    
    app = QApplication(sys.argv)
    
    # Create managers
    file_manager = FileManager()
    channel_manager = ChannelManager()
    
    # Create test data
    create_test_files_and_channels(file_manager, channel_manager)
    
    # Create mixer wizard window
    mixer_wizard = SignalMixerWizardWindow(
        file_manager=file_manager,
        channel_manager=channel_manager
    )
    
    # Show the window
    mixer_wizard.show()
    
    print("\n=== Debug Instructions ===")
    print("1. The mixer wizard window should now be open")
    print("2. Click the '+ Add Channel' button")
    print("3. Watch for any dialog popup that appears and closes quickly")
    print("4. Check the console output for any error messages")
    print("5. If you see a dialog, try to read what it says before it closes")
    print("6. Report any error messages or dialog content you see")
    
    print("\n=== Expected Behavior ===")
    print("✓ Channel C should be added to the Additional Channels section")
    print("✓ Legend should update to show channel C")
    print("✓ Plot should update (though C won't be visible until you select a file/channel)")
    print("✓ No dialogs should appear")
    
    print("\n=== If You See a Dialog ===")
    print("• Note the title and content of the dialog")
    print("• Check if it's an error, warning, or info dialog")
    print("• Look for any error messages in the console")
    print("• The dialog might be from matplotlib, Qt, or the application itself")
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 