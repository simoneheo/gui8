#!/usr/bin/env python3

import sys
import traceback

try:
    print("Testing Modified Signal Mixer Wizard...")
    
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
    print("Creating Modified Signal Mixer Wizard Window...")
    wizard = SignalMixerWizardWindow(fm, cm)
    print("✅ Signal Mixer Wizard Window created successfully")
    
    # Test that the wizard has the new attributes
    print("Testing new features...")
    
    # Check that it has the step tracking attributes
    assert hasattr(wizard, 'current_step'), "Missing current_step attribute"
    assert wizard.current_step == 1, "Should start on step 1"
    print("✅ Step tracking working")
    
    # Check that it has additional channels tracking
    assert hasattr(wizard, 'additional_channel_widgets'), "Missing additional_channel_widgets attribute"
    assert isinstance(wizard.additional_channel_widgets, list), "additional_channel_widgets should be a list"
    print("✅ Additional channels tracking working")
    
    # Check that the manager has the new methods
    assert hasattr(wizard.manager, 'set_selected_channels'), "Manager missing set_selected_channels method"
    assert hasattr(wizard.manager, 'validate_channel_compatibility'), "Manager missing validate_channel_compatibility method"
    print("✅ Manager new methods available")
    
    # Check that the UI has the step widgets
    assert hasattr(wizard, 'steps'), "Missing steps widget"
    assert hasattr(wizard, 'step1_ui'), "Missing step1_ui"
    assert hasattr(wizard, 'step2_ui'), "Missing step2_ui"
    print("✅ Step UI widgets created")
    
    # Check navigation buttons
    assert hasattr(wizard, 'back_button'), "Missing back_button"
    assert hasattr(wizard, 'next_button'), "Missing next_button"
    print("✅ Navigation buttons created")
    
    # Test additional channel methods
    assert hasattr(wizard, '_add_additional_channel'), "Missing _add_additional_channel method"
    assert hasattr(wizard, '_get_additional_channels'), "Missing _get_additional_channels method"
    print("✅ Additional channel methods available")
    
    print("✅ All tests passed! Modified Signal Mixer Wizard is working correctly.")
    
    # Optionally show the window for visual verification
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        print("Showing wizard window for visual verification...")
        wizard.show()
        app.exec()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1) 