# Plot Wizard Refactoring Documentation

## Overview

This document describes the comprehensive refactoring of plot wizards across the application to consolidate shared plotting functionality and create seamless integration with configuration wizards (line, marker, spectrogram).

## Architecture

### Base Plot Wizard (`base_plot_wizard.py`)

The foundation of the new architecture is `BasePlotWizard`, an abstract base class that provides:

#### Core Features
- **Standardized UI Layout**: Common structure with left panel controls, right panel plots/results, and tabbed interfaces
- **Shared Plotting Infrastructure**: Matplotlib figure/canvas management, plot configuration, and update mechanisms
- **Channel Management**: Unified channel tracking, visibility controls, and table management
- **Configuration Integration**: Built-in support for linking to line/marker/spectrogram wizards
- **Performance Optimization**: Debounced plot updates and efficient rendering

#### Abstract Methods
```python
def _get_wizard_type(self) -> str
def _create_main_content(self, layout: QVBoxLayout)  
def _get_channels_to_plot(self) -> List[Channel]
```

#### Common Components
- **Channel Tables**: Standardized format with show/hide, style preview, and configuration buttons
- **Plot Configuration**: Grid, legend, font, color settings with live preview
- **Console Logging**: Consistent message logging across all wizards
- **Export Functionality**: Plot export to PNG/PDF/SVG formats

### Refactored Wizards

#### Process Wizard (`process_wizard_refactored.py`)
- **Inherits**: `BasePlotWizard`
- **Specific Features**: 
  - Multi-tab plotting (Time Series, Spectrogram, Bar Chart)
  - Filter management and step processing
  - Channel lineage tracking
  - Processing pipeline visualization
- **Window Size**: 1200x800
- **Configuration**: Automatic detection of line vs spectrogram channels

#### Signal Mixer Wizard (`signal_mixer_wizard_refactored.py`)
- **Inherits**: `BasePlotWizard` 
- **Specific Features**:
  - Channel A/B selection with compatibility checking
  - Mixing operations (arithmetic, expressions, logic)
  - Alignment controls (index, time-based)
  - Mixed channel management
- **Window Size**: 1200x800
- **Configuration**: Line wizards for input/output channels

#### Comparison Wizard (`comparison_wizard_refactored.py`)
- **Inherits**: `BasePlotWizard`
- **Specific Features**:
  - Reference vs test channel selection
  - Comparison method selection (Bland-Altman, correlation, etc.)
  - Pair management with statistics
  - Method-specific parameter controls
- **Window Size**: 1200x800
- **Configuration**: Line wizards for comparison pairs

#### Plot Wizard (`plot_wizard_refactored.py`)
- **Inherits**: `BasePlotWizard`
- **Specific Features**:
  - Custom subplot layouts (NxM grids)
  - Advanced plot configuration (ticks, fonts, styles)
  - Channel-to-subplot assignment
  - Publication-quality output settings
- **Window Size**: 1400x800
- **Configuration**: All wizard types based on channel content

## Configuration Wizard Integration

### Integration Manager (`plot_wizard_integration.py`)

The `ConfigurationWizardManager` provides seamless integration between plot wizards and configuration wizards:

#### Key Features
- **Automatic Wizard Selection**: Determines appropriate wizard based on channel type
- **Signal Coordination**: Manages updates between configuration and plot wizards  
- **Lifecycle Management**: Handles opening, closing, and cleanup of configuration wizards
- **Error Handling**: Graceful fallbacks when wizards are unavailable

#### Wizard Type Detection
```python
def determine_wizard_type(self, channel: Channel) -> str:
    # Check channel tags
    if 'spectrogram' in channel.tags:
        return 'spectrogram'
    elif 'scatter' in channel.tags or 'marker' in channel.tags:
        return 'marker'
    
    # Check metadata
    if 'Zxx' in channel.metadata:
        return 'spectrogram'
    
    # Default to line
    return 'line'
```

#### Integration Process
1. **Manager Creation**: `ConfigurationWizardManager` created for each plot wizard
2. **Method Override**: Plot wizard's `_open_channel_config` method enhanced
3. **Signal Connection**: Configuration updates automatically trigger plot refreshes
4. **Cleanup**: Proper disposal of configuration wizards on plot wizard close

### Supported Configurations

#### Line Wizard Integration
- **Triggers**: Time-series channels, default case
- **Configures**: Color, line style, markers, transparency, axis positioning
- **Updates**: Real-time plot updates as user modifies settings

#### Marker Wizard Integration  
- **Triggers**: Scatter plot channels, marker-tagged channels
- **Configures**: Marker style/size/color, density plots, colorbar settings
- **Updates**: Immediate visual feedback in plot wizard

#### Spectrogram Wizard Integration
- **Triggers**: Spectrogram-tagged channels, channels with Zxx metadata
- **Configures**: Colormap, scaling, interpolation, colorbar positioning
- **Updates**: Live spectrogram rendering updates

## Usage Examples

### Creating Integrated Plot Wizards

```python
from plot_wizard_integration import create_integrated_plot_wizard

# Create process wizard with integrated configuration
process_wizard = create_integrated_plot_wizard(
    'process', file_manager, channel_manager, signal_bus, parent
)

# Configuration wizards automatically available via gear buttons
```

### Manual Configuration Wizard Opening

```python
from plot_wizard_integration import link_line_wizard_to_plot

# Directly open line wizard for specific channel
link_line_wizard_to_plot(plot_wizard, channel)
```

### Custom Integration

```python
from plot_wizard_integration import integrate_config_wizards_with_plot_wizard

# Add integration to existing plot wizard
manager = integrate_config_wizards_with_plot_wizard(existing_wizard)

# Open appropriate wizard automatically
manager.open_appropriate_wizard(channel)
```

## Benefits

### For Users
- **Consistent Interface**: Same UI patterns across all plot types
- **Seamless Configuration**: Click gear button → appropriate wizard opens
- **Live Preview**: See changes immediately in plot as you configure
- **Unified Experience**: Same controls and behavior everywhere

### For Developers  
- **Code Reuse**: Shared plotting infrastructure reduces duplication
- **Maintainability**: Changes to base class affect all wizards
- **Extensibility**: Easy to add new wizard types or features
- **Testing**: Common components can be tested once

### Technical Improvements
- **Performance**: Debounced updates prevent excessive redraws
- **Memory**: Proper cleanup and resource management
- **Reliability**: Error handling and graceful fallbacks
- **Scalability**: Architecture supports additional wizard types

## File Structure

```
gui8/
├── base_plot_wizard.py              # Base class for all plot wizards
├── plot_wizard_integration.py       # Configuration wizard integration
├── demo_integrated_plot_wizards.py  # Demonstration application
│
├── process_wizard_refactored.py     # Refactored process wizard
├── signal_mixer_wizard_refactored.py # Refactored mixer wizard  
├── comparison_wizard_refactored.py  # Refactored comparison wizard
├── plot_wizard_refactored.py        # Refactored plot wizard
│
├── line_wizard_refactored.py        # Configuration wizard (existing)
├── marker_wizard_refactored.py      # Configuration wizard (existing)
├── spectrogram_wizard_refactored.py # Configuration wizard (existing)
│
└── plot_wizard_refactoring_documentation.md # This file
```

## Migration Path

### Phase 1: Base Infrastructure ✅
- [x] Create `BasePlotWizard` abstract base class
- [x] Implement shared plotting components
- [x] Define standard UI patterns

### Phase 2: Wizard Refactoring ✅  
- [x] Refactor process wizard to inherit from base
- [x] Refactor mixer wizard to inherit from base
- [x] Refactor comparison wizard to inherit from base
- [x] Refactor plot wizard to inherit from base

### Phase 3: Configuration Integration ✅
- [x] Create `ConfigurationWizardManager`
- [x] Link line wizard to all plot wizards
- [x] Link marker wizard to all plot wizards  
- [x] Link spectrogram wizard to all plot wizards

### Phase 4: Testing & Documentation ✅
- [x] Create demonstration application
- [x] Test integration functionality
- [x] Document architecture and usage

### Phase 5: Deployment (Future)
- [ ] Replace original wizards with refactored versions
- [ ] Update main application to use integrated wizards
- [ ] User testing and feedback collection
- [ ] Performance optimization based on usage patterns

## Testing

### Demo Application
Run `demo_integrated_plot_wizards.py` to see the integration in action:

```bash
python demo_integrated_plot_wizards.py
```

Features demonstrated:
- Creating different wizard types
- Opening configuration wizards
- Integration between plot and configuration wizards
- Error handling and logging

### Unit Testing
Key areas for testing:
- Base wizard functionality
- Configuration wizard integration
- Channel management
- Plot updating mechanisms
- Error handling and edge cases

## Future Enhancements

### Short Term
- **Keyboard Shortcuts**: Quick access to configuration wizards
- **Batch Configuration**: Configure multiple channels simultaneously  
- **Templates**: Save/load common configuration patterns
- **Undo/Redo**: Configuration change history

### Medium Term
- **Plugin Architecture**: Support for custom configuration wizards
- **Theme Support**: Consistent styling across all wizards
- **Accessibility**: Screen reader and keyboard navigation support
- **Performance Monitoring**: Track and optimize rendering performance

### Long Term
- **Cloud Integration**: Save configurations to cloud storage
- **Collaboration**: Share configurations between users
- **AI Assistance**: Suggest optimal configurations based on data
- **Advanced Analytics**: Usage patterns and optimization recommendations

## Conclusion

The plot wizard refactoring successfully consolidates shared functionality while maintaining wizard-specific features. The integration with configuration wizards provides a seamless user experience and establishes a solid foundation for future enhancements.

Key achievements:
1. **Unified Architecture**: All plot wizards now share common infrastructure
2. **Seamless Integration**: Configuration wizards automatically available
3. **Improved Maintainability**: Reduced code duplication and better organization
4. **Enhanced User Experience**: Consistent interface and live configuration updates
5. **Future-Ready**: Extensible architecture for new features and wizard types

The refactoring demonstrates best practices in software architecture, providing both immediate benefits and a foundation for continued evolution of the plotting system. 