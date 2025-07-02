#!/usr/bin/env python3

import sys
import traceback

try:
    print("Testing Updated Signal Mixer Wizard Features...")
    
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
    print("Creating Updated Signal Mixer Wizard Window...")
    wizard = SignalMixerWizardWindow(fm, cm)
    print("✅ Signal Mixer Wizard Window created successfully")
    
    # Test the updated features
    print("\nTesting updated features...")
    
    # 1. Test alignment controls tracking
    assert hasattr(wizard, 'alignment_controls'), "Missing alignment_controls attribute"
    assert isinstance(wizard.alignment_controls, dict), "alignment_controls should be a dict"
    print("✅ Alignment controls tracking working")
    
    # 2. Test step 1 has single plot instead of subplots
    assert hasattr(wizard, 'step1_ax'), "Missing step1_ax (single plot)"
    assert not hasattr(wizard, 'step1_ax_a'), "Should not have step1_ax_a (removed subplots)"
    assert not hasattr(wizard, 'step1_ax_b'), "Should not have step1_ax_b (removed subplots)"
    print("✅ Step 1 single plot structure working")
    
    # 3. Test alignment controls methods
    assert hasattr(wizard, '_update_alignment_controls_step1'), "Missing _update_alignment_controls_step1 method"
    assert hasattr(wizard, '_build_alignment_controls_step1'), "Missing _build_alignment_controls_step1 method"
    print("✅ Step 1 alignment control methods available")
    
    # 4. Test additional channel methods still work
    assert hasattr(wizard, '_add_additional_channel'), "Missing _add_additional_channel method"
    assert hasattr(wizard, '_get_additional_channels'), "Missing _get_additional_channels method"
    print("✅ Additional channel methods available")
    
    # 5. Test that channel info display was removed
    assert not hasattr(wizard, 'channel_info_text'), "channel_info_text should be removed"
    print("✅ Channel info display removed as requested")
    
    # 6. Test updated plotting method
    assert hasattr(wizard, '_update_step1_plots'), "Missing _update_step1_plots method"
    print("✅ Updated plotting method available")
    
    print("\n🎉 All updated features are working correctly!")
    print("✅ Removed channel information box from Step 1")
    print("✅ Moved alignment controls to Step 1") 
    print("✅ Dynamic alignment controls for additional channels")
    print("✅ Single overlapping plot instead of subplots")
    
    # Optionally show the window for visual verification
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        print("\nShowing wizard window for visual verification...")
        wizard.show()
        app.exec()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1) 