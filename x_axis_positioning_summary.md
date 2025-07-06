# X-Axis Positioning Feature Summary

## Overview
This document describes the implementation of top x-axis positioning functionality in the Line, Marker, and Spectrogram wizards. When users select "Top" for the x-axis position in any wizard, the actual plot will display ticks and values on the top x-axis instead of (or in addition to) the bottom x-axis.

## Implementation Details

### 1. Wizard Configuration Storage

#### Line Wizard (`line_wizard.py`)
- Stores x-axis preference in `channel.xaxis` property
- Values: `"x-bottom"` (default) or `"x-top"`
- Updated when user changes X-axis position radio buttons

#### Spectrogram Wizard (`spectrogram_wizard.py`)
- Stores x-axis preference in `channel.xaxis` property
- Values: `"x-bottom"` (default) or `"x-top"`
- Updated when user changes X-axis position radio buttons

#### Marker Wizard (`marker_wizard.py`)
- Stores x-axis preference in marker config dict as `x_axis`
- Values: `"bottom"` (default) or `"top"`
- Converted to channel format (`"x-bottom"` or `"x-top"`) when updating plot items

### 2. Plot Rendering Implementation

#### New Method: `_configure_x_axis_positioning()`
Located in `plot_wizard_manager.py`, this method:

1. **Analyzes all items** in a subplot to determine x-axis preferences
2. **Checks multiple sources**:
   - `channel.xaxis` property (for line/spectrogram wizards)
   - `item['x_axis']` property (for marker wizard)

3. **Configures matplotlib axes** based on preferences:
   - **Top only**: `ax.xaxis.tick_top()` and `ax.xaxis.set_label_position('top')`
   - **Both top and bottom**: `ax.xaxis.set_ticks_position('both')`
   - **Bottom only** (default): `ax.xaxis.tick_bottom()` and `ax.xaxis.set_label_position('bottom')`

### 3. Integration Points

#### Plot Update Flow
```
Wizard Updates → Plot Items → _configure_x_axis_positioning() → Matplotlib Axes
```

1. **Wizard saves settings**: X-axis preference stored in channel or config
2. **Plot manager updates**: `_on_*_updated_from_wizard()` methods store x-axis preferences
3. **Plot rendering**: `_configure_x_axis_positioning()` called during `_update_plot()`
4. **Matplotlib configuration**: Axes configured with appropriate tick positions

## Usage Examples

### Single Plot Type Scenarios

#### Top X-Axis Only
```
User selects "Top" in Line Wizard
→ channel.xaxis = "x-top"
→ Plot shows ticks and labels on top x-axis only
```

#### Bottom X-Axis Only (Default)
```
User selects "Bottom" in Spectrogram Wizard
→ channel.xaxis = "x-bottom"
→ Plot shows ticks and labels on bottom x-axis only
```

### Mixed Plot Type Scenarios

#### Mixed Preferences
```
Line 1: x-axis = "top"
Line 2: x-axis = "bottom"
→ Plot shows ticks on both top and bottom
→ Labels remain on bottom (default)
```

#### Consistent Preferences
```
Spectrogram: x-axis = "top"
Line overlay: x-axis = "top"
→ Plot shows ticks and labels on top x-axis only
```

## Technical Implementation

### Code Structure

#### 1. Wizard Property Management
```python
# Line/Spectrogram Wizards
self.channel.xaxis = "x-bottom" if self.bottom_x_axis_radio.isChecked() else "x-top"

# Marker Wizard (in config dict)
x_axis = "bottom" if self.bottom_x_axis_radio.isChecked() else "top"
self.pair_config['x_axis'] = x_axis
```

#### 2. Plot Manager Integration
```python
def _on_marker_updated_from_wizard(self, item_index, marker_config):
    # Store x-axis preference for marker plots
    x_axis = marker_config.get('x_axis', 'bottom')
    item['x_axis'] = f"x-{x_axis}"  # Convert to channel format
```

#### 3. Matplotlib Configuration
```python
def _configure_x_axis_positioning(self, ax_left, ax_right, items):
    # Analyze preferences
    has_top_x_axis = any(check_for_top_preference(item) for item in items)
    has_bottom_x_axis = any(check_for_bottom_preference(item) for item in items)
    
    # Configure axes
    if has_top_x_axis and not has_bottom_x_axis:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    elif has_top_x_axis and has_bottom_x_axis:
        ax.xaxis.set_ticks_position('both')
    else:
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
```

## Benefits

### 1. **Consistent User Experience**
- All three wizards (Line, Marker, Spectrogram) support x-axis positioning
- Settings are preserved and applied correctly in plots
- Mixed plot types handle x-axis preferences intelligently

### 2. **Flexible Visualization**
- Users can position x-axis where it makes most sense for their data
- Supports scientific plotting conventions
- Handles complex multi-plot scenarios

### 3. **Intelligent Behavior**
- When plots have mixed x-axis preferences, shows ticks on both top and bottom
- Maintains readability by keeping labels in consistent position
- Graceful fallback to default bottom positioning

## Future Enhancements

### Possible Improvements
1. **Label Position Control**: Allow users to choose label position independently from tick position
2. **Subplot-Level Override**: Allow subplot-level x-axis positioning settings
3. **Advanced Tick Control**: More granular control over tick appearance on top/bottom
4. **Preview Updates**: Show x-axis positioning in wizard preview sections

### Integration Opportunities
1. **Subplot Wizard**: Add x-axis positioning controls to subplot configuration
2. **Export Settings**: Include x-axis positioning in export configurations
3. **Templates**: Save x-axis preferences in plot templates

## Conclusion

The x-axis positioning feature provides users with essential control over plot appearance while maintaining the consistent UI patterns established by the base configuration wizard. The implementation correctly handles all three wizard types and provides intelligent behavior for mixed plot scenarios. 