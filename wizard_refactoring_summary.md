# Wizard Refactoring Summary

## Overview
This document outlines the refactoring of the configuration wizards (Line, Marker, Spectrogram) to use a common base class `BaseConfigWizard` for consistent UI presentation and shared functionality.

## Base Configuration Wizard (`base_config_wizard.py`)

### Key Features
- **Abstract Base Class**: Enforces consistent implementation across all wizards
- **Standardized UI Layout**: Common title, tabs, preview, and button sections
- **Shared Components**: Common controls like transparency, axis position, bring-to-front
- **Colorbar Integration**: Standardized colorbar controls for spectrograms and density plots
- **Property Management**: Unified backup/restore and update mechanisms

### Common UI Structure
All wizards now follow this standardized structure:

```
┌─────────────────────────────────────┐
│ Title: "Editing [Type]: [Name]"     │
│ Info: "Channel ID: [ID]"            │
├─────────────────────────────────────┤
│ ┌─ Basic Tab ─┐ ┌─ Axis Tab ─┐     │
│ │ • Legend    │ │ • X-Axis    │     │
│ │ • Alpha     │ │ • Y-Axis*   │     │
│ │ • Bring to  │ │             │     │
│ │   Front     │ │             │     │
│ └─────────────┘ └─────────────┘     │
│ ┌─ Type-Specific Tabs ─┐            │
│ │ • Style/Appearance    │            │
│ │ • Colorbar (if needed)│            │
│ │ • Advanced Settings   │            │
│ └───────────────────────┘            │
├─────────────────────────────────────┤
│ Preview Section                     │
│ [Live preview of current settings]  │
├─────────────────────────────────────┤
│ [OK] [Cancel] [Apply]               │
└─────────────────────────────────────┘
```

*Y-Axis only appears in Line Wizard

## Refactored Wizards

### 1. Line Wizard (`line_wizard_refactored.py`)
**Inherits from BaseConfigWizard**

#### Tabs Structure:
- **Basic**: Legend, Transparency, Bring to Front
- **Axis**: X-Axis Position, Y-Axis Selection (Left/Right)
- **Style**: Line Color, Line Style, Marker

#### Specific Features:
- Color picker with preset colors
- Y-axis selection (left/right)
- Line style and marker selection
- Visual color preview in preview section

### 2. Spectrogram Wizard (`spectrogram_wizard_refactored.py`)
**Inherits from BaseConfigWizard**

#### Tabs Structure:
- **Basic**: Legend, Transparency, Bring to Front
- **Axis**: X-Axis Position
- **Appearance**: Colormap, Interpolation
- **Colorbar**: Full colorbar controls (position, ticks, labels, etc.)
- **Scaling**: Frequency/Time scales and limits
- **Advanced**: Aspect ratio, Shading

#### Specific Features:
- Comprehensive colormap selection
- Auto/manual color range controls
- Frequency and time axis scaling
- Full colorbar customization
- Advanced rendering options

### 3. Marker Wizard (`marker_wizard_refactored.py`)
**Inherits from BaseConfigWizard**

#### Tabs Structure:
- **Basic**: Legend, Transparency, Bring to Front
- **Axis**: X-Axis Position
- **Scatter Markers**: Traditional marker settings
- **Density Plots**: Hexbin and KDE settings
- **Colorbar**: Full colorbar controls (for density plots)

#### Specific Features:
- Traditional scatter plot markers
- Hexbin density plots with grid size control
- KDE density plots with bandwidth and contour levels
- Colorscheme selection with reverse option
- Comprehensive colorbar controls for density plots

## Benefits of Refactoring

### 1. **Consistent User Experience**
- All wizards have the same layout structure
- Common controls work identically across wizards
- Predictable navigation and interaction patterns

### 2. **Code Reusability**
- Shared UI components (ColorButton, transparency controls, etc.)
- Common property management methods
- Standardized colorbar controls

### 3. **Maintainability**
- Single source of truth for common functionality
- Changes to shared components propagate automatically
- Easier to add new wizard types

### 4. **Extensibility**
- Easy to add new common features to all wizards
- Simple to create new wizard types
- Standardized extension points

### 5. **Quality Assurance**
- Consistent behavior across all wizards
- Shared validation and error handling
- Uniform backup/restore mechanisms

## Implementation Details

### Abstract Methods (Must be implemented by subclasses):
```python
def _backup_properties(self) -> dict
def _get_object_name(self) -> str
def _get_object_info(self) -> str
def _create_main_tabs(self, tab_widget: QTabWidget)
def load_properties(self)
def _update_properties(self)
def _restore_properties(self)
```

### Common Methods (Available to all subclasses):
```python
def _load_common_properties(self)
def _update_common_properties(self)
def _create_colorbar_controls_group(self) -> QGroupBox
def _load_colorbar_properties(self)
def _update_colorbar_properties(self)
def update_preview(self)  # Can be overridden
```

### Utility Functions:
```python
def create_colormap_combo(default: str = 'viridis') -> QComboBox
def create_transparency_controls() -> tuple[QDoubleSpinBox, QSlider]
```

## Migration Path

### Step 1: Test Base Class
- Verify `base_config_wizard.py` works correctly
- Test abstract methods and common functionality

### Step 2: Gradual Migration
- Replace existing wizards one at a time
- Keep old versions as backups during transition
- Test each wizard thoroughly after migration

### Step 3: Integration Testing
- Verify all wizards work with plot_wizard_manager
- Test signal connections and property updates
- Validate preview functionality

### Step 4: Cleanup
- Remove old wizard files
- Update imports throughout the codebase
- Update documentation

## Future Enhancements

### Possible Additions to Base Class:
1. **Themes**: Support for light/dark themes
2. **Presets**: Save/load wizard configurations
3. **Validation**: Input validation framework
4. **Help System**: Context-sensitive help
5. **Undo/Redo**: Multi-level undo support

### New Wizard Types:
1. **Subplot Wizard**: Already exists, could be refactored
2. **Annotation Wizard**: For text and shape annotations
3. **Export Wizard**: For plot export settings
4. **Theme Wizard**: For global appearance settings

## Conclusion

The refactoring to `BaseConfigWizard` provides a solid foundation for consistent, maintainable, and extensible configuration dialogs. The standardized UI patterns improve user experience while reducing development overhead for new features and wizard types.

The three refactored wizards demonstrate how specific functionality can be cleanly separated from common UI patterns, making the codebase more organized and easier to maintain. 