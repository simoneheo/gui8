#!/usr/bin/env python3

import sys
import traceback

try:
    print("Testing Signal Mixer Wizard imports...")
    
    # Test basic imports
    from PySide6.QtWidgets import QApplication
    print("✅ PySide6 imports successful")
    
    from signal_mixer_wizard_window import SignalMixerWizardWindow
    print("✅ SignalMixerWizardWindow import successful")
    
    from signal_mixer_wizard_manager import SignalMixerWizardManager
    print("✅ SignalMixerWizardManager import successful")
    
    # Test basic manager imports
    from file_manager import FileManager
    from channel_manager import ChannelManager
    print("✅ Manager imports successful")
    
    # Create a test application
    app = QApplication(sys.argv)
    print("✅ QApplication created")
    
    # Try to create managers
    fm = FileManager()
    cm = ChannelManager()
    print("✅ Managers created")
    
    # Try to create the wizard window
    print("Creating Signal Mixer Wizard Window...")
    wizard = SignalMixerWizardWindow(fm, cm)
    print("✅ Signal Mixer Wizard Window created successfully")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1) 