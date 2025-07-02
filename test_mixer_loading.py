#!/usr/bin/env python3
"""
Test script for mixer loading functionality
Tests that mixers can be loaded and registered properly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mixer_loading():
    """Test loading mixers from the mixer directory"""
    print("Testing mixer loading...")
    
    try:
        # Import mixer registry
        from mixer.mixer_registry import MixerRegistry, load_all_mixers
        print("✅ Mixer registry imported successfully")
        
        # Load all mixers
        print("\nLoading mixers from 'mixer' directory...")
        load_all_mixers("mixer")
        
        # Check what mixers were loaded
        available_mixers = MixerRegistry.all_mixers()
        print(f"✅ Loaded {len(available_mixers)} mixers: {available_mixers}")
        
        # Test getting individual mixers
        print("\nTesting individual mixer access...")
        for mixer_name in available_mixers:
            try:
                mixer_cls = MixerRegistry.get(mixer_name)
                print(f"✅ {mixer_name}: {mixer_cls}")
                
                # Check mixer properties
                if hasattr(mixer_cls, 'name'):
                    print(f"   - Name: {mixer_cls.name}")
                if hasattr(mixer_cls, 'description'):
                    print(f"   - Description: {mixer_cls.description}")
                if hasattr(mixer_cls, 'category'):
                    print(f"   - Category: {mixer_cls.category}")
                    
            except Exception as e:
                print(f"❌ Error accessing mixer {mixer_name}: {e}")
        
        print(f"\n✅ Mixer loading test completed successfully!")
        print(f"Total mixers available: {len(available_mixers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixer loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixer_creation():
    """Test creating and using a mixer"""
    print("\n" + "="*50)
    print("Testing mixer creation and usage...")
    
    try:
        from mixer.mixer_registry import MixerRegistry
        from channel import Channel
        import numpy as np
        
        # Create test channels
        time_data = np.linspace(0, 10, 100)
        
        channel_a = Channel(
            channel_id="test_a",
            filename="test.txt",
            legend_label="Test A",
            xdata=time_data,
            ydata=np.sin(time_data),
            step=0
        )
        
        channel_b = Channel(
            channel_id="test_b", 
            filename="test.txt",
            legend_label="Test B",
            xdata=time_data,
            ydata=np.cos(time_data),
            step=0
        )
        
        print("✅ Test channels created")
        
        # Test arithmetic mixer if available
        if "arithmetic" in MixerRegistry.all_mixers():
            print("\nTesting ArithmeticMixer...")
            arithmetic_mixer_cls = MixerRegistry.get("arithmetic")
            
            # Create mixer instance
            mixer = arithmetic_mixer_cls(operation="add", label="C")
            
            # Apply mixer
            channels = {"A": channel_a, "B": channel_b}
            result = mixer.apply(channels)
            
            print(f"✅ ArithmeticMixer created result channel: {result.legend_label}")
            print(f"   - Result data length: {len(result.ydata)}")
            print(f"   - Result data type: {type(result.ydata)}")
            
        else:
            print("⚠️  ArithmeticMixer not available")
        
        # Test expression mixer if available
        if "expression" in MixerRegistry.all_mixers():
            print("\nTesting ExpressionMixer...")
            expression_mixer_cls = MixerRegistry.get("expression")
            
            # Create mixer instance
            mixer = expression_mixer_cls(expression="A + B", label="D")
            
            # Apply mixer
            channels = {"A": channel_a, "B": channel_b}
            result = mixer.apply(channels)
            
            print(f"✅ ExpressionMixer created result channel: {result.legend_label}")
            print(f"   - Result data length: {len(result.ydata)}")
            
        else:
            print("⚠️  ExpressionMixer not available")
            
        print(f"\n✅ Mixer creation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Mixer creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Mixer Loading Test")
    print("=" * 50)
    
    # Test mixer loading
    loading_success = test_mixer_loading()
    
    # Test mixer creation if loading succeeded
    if loading_success:
        creation_success = test_mixer_creation()
    else:
        creation_success = False
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Mixer Loading: {'✅ PASS' if loading_success else '❌ FAIL'}")
    print(f"Mixer Creation: {'✅ PASS' if creation_success else '❌ FAIL'}")
    
    if loading_success and creation_success:
        print("\n🎉 All tests passed! Mixers are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 