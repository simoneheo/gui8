from PySide6.QtWidgets import (
    QWidget, QFormLayout, QHBoxLayout, QVBoxLayout, QComboBox, 
    QCheckBox, QGroupBox, QButtonGroup, QRadioButton, QTabWidget
)
from PySide6.QtCore import Signal
from typing import Optional

from base_config_wizard import BaseConfigWizard, ColorButton
from channel import Channel
from plot_manager import PLOT_STYLES


class LineWizard(BaseConfigWizard):
    """
    Dialog for editing channel line properties
    Inherits from BaseConfigWizard for consistent UI and shared functionality
    """
    
    # Specific signal for line wizard
    channel_updated = Signal(str)  # channel_id when changes are applied
    
    def __init__(self, channel: Channel, parent=None):
        # Initialize base class with channel object and wizard type
        super().__init__(channel, "Line", parent)
        
        # Line-specific UI components
        self.color_button = None
        self.color_combo = None
        self.style_combo = None
        self.marker_combo = None
        self.axis_button_group = None
        self.left_axis_radio = None
        self.right_axis_radio = None
        
        # Connect specific signal to base signal
        self.config_updated.connect(lambda obj: self.channel_updated.emit(obj.channel_id))
    
    def _setup_window(self):
        """Override to set line wizard specific window size"""
        super()._setup_window()
        self.setFixedSize(450, 550)  # Smaller than default for line wizard
    
    def _backup_properties(self) -> dict:
        """Backup original channel properties for cancel operation"""
        return {
            'color': self.config_object.color,
            'style': self.config_object.style,
            'marker': self.config_object.marker,
            'yaxis': self.config_object.yaxis,
            'xaxis': getattr(self.config_object, 'xaxis', 'x-bottom'),
            'legend_label': self.config_object.legend_label,
            'z_order': getattr(self.config_object, 'z_order', 0),
            'alpha': getattr(self.config_object, 'alpha', 1.0)
        }
    
    def _get_object_name(self) -> str:
        """Get display name for the channel"""
        return self.config_object.ylabel or 'Unnamed Channel'
    
    def _get_object_info(self) -> str:
        """Get info text for the channel"""
        return f"Channel ID: {getattr(self.config_object, 'channel_id', 'Unknown')}"
    
    def _create_main_tabs(self, tab_widget: QTabWidget):
        """Create line-specific tabs"""
        # Style tab (line-specific)
        style_tab = self._create_style_tab()
        tab_widget.addTab(style_tab, "Style")
    
    def _add_axis_specific_controls(self, layout: QVBoxLayout):
        """Add Y-axis selection controls specific to line wizard"""
        # Y-Axis Selection Group
        y_axis_group = QGroupBox("Y-Axis Selection")
        y_axis_layout = QVBoxLayout(y_axis_group)
        
        self.axis_button_group = QButtonGroup()
        self.left_axis_radio = QRadioButton("Left Axis")
        self.right_axis_radio = QRadioButton("Right Axis")
        
        self.axis_button_group.addButton(self.left_axis_radio, 0)
        self.axis_button_group.addButton(self.right_axis_radio, 1)
        
        y_axis_layout.addWidget(self.left_axis_radio)
        y_axis_layout.addWidget(self.right_axis_radio)
        
        layout.addWidget(y_axis_group)
        
        # Connect Y-axis signals
        self.axis_button_group.buttonToggled.connect(self.update_preview)
    
    def _create_style_tab(self) -> QWidget:
        """Create the style settings tab specific to lines"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Color Selection
        color_layout = QHBoxLayout()
        self.color_button = ColorButton()
        self.color_combo = QComboBox()
        self.color_combo.addItems(PLOT_STYLES['Colors'].keys())
        self.color_combo.currentTextChanged.connect(self.on_color_combo_changed)
        self.color_button.color_changed.connect(self.on_color_button_changed)
        
        color_layout.addWidget(self.color_button)
        color_layout.addWidget(self.color_combo)
        layout.addRow("Line Color:", color_layout)
        
        # Line Style
        self.style_combo = QComboBox()
        for name, value in PLOT_STYLES['Line Styles'].items():
            self.style_combo.addItem(name, value)
        self.style_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Line Style:", self.style_combo)
        
        # Marker
        self.marker_combo = QComboBox()
        for name, value in PLOT_STYLES['Markers'].items():
            self.marker_combo.addItem(name, value)
        self.marker_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Marker:", self.marker_combo)
        
        return tab
    
    def load_properties(self):
        """Load current channel properties into the UI"""
        # Load common properties first
        self._load_common_properties()
        
        # Load line-specific properties
        # Color
        current_color = self.config_object.color
        self.color_button.set_color(current_color)
        
        # Find matching color in combo
        for i in range(self.color_combo.count()):
            color_name = self.color_combo.itemText(i)
            if color_name in PLOT_STYLES['Colors'] and PLOT_STYLES['Colors'][color_name] == current_color:
                self.color_combo.setCurrentIndex(i)
                break
        else:
            # Custom color
            self.color_combo.setCurrentText("Custom")
        
        # Line Style
        current_style = self.config_object.style
        for i in range(self.style_combo.count()):
            if self.style_combo.itemData(i) == current_style:
                self.style_combo.setCurrentIndex(i)
                break
        
        # Marker
        current_marker = self.config_object.marker
        for i in range(self.marker_combo.count()):
            if self.marker_combo.itemData(i) == current_marker:
                self.marker_combo.setCurrentIndex(i)
                break
        
        # Y-Axis
        if self.config_object.yaxis == 'left':
            self.left_axis_radio.setChecked(True)
        else:
            self.right_axis_radio.setChecked(True)
        
        # Initial preview update
        self.update_preview()
    
    def on_color_combo_changed(self, color_name: str):
        """Handle color combo box changes"""
        if color_name in PLOT_STYLES['Colors']:
            color_value = PLOT_STYLES['Colors'][color_name]
            self.color_button.set_color(color_value)
        self.update_preview()
    
    def on_color_button_changed(self, color: str):
        """Handle color button changes"""
        # Set combo to "Custom" for custom color
        self.color_combo.setCurrentText("Custom")
        self.update_preview()
    
    def update_preview(self):
        """Update the preview display with line-specific information"""
        if not self.preview_label:
            return
        
        # Get current settings
        legend_name = self.legend_edit.text() if self.legend_edit else "Unnamed"
        color = self.color_button.get_color() if self.color_button else "#1f77b4"
        alpha = self.alpha_spin.value() if self.alpha_spin else 1.0
        style = self.style_combo.currentText() if self.style_combo else "Solid"
        marker = self.marker_combo.currentText() if self.marker_combo else "None"
        y_axis = "Left" if self.left_axis_radio and self.left_axis_radio.isChecked() else "Right"
        x_axis = "Bottom" if self.bottom_x_axis_radio and self.bottom_x_axis_radio.isChecked() else "Top"
        bring_to_front = "Yes" if self.bring_to_front_checkbox and self.bring_to_front_checkbox.isChecked() else "No"
        
        # Create preview text with visual representation
        preview_text = f"""
        <div style="padding: 5px; font-family: monospace; font-size: 10px;">
            <b>Line Configuration Preview</b><br>
            <b>Legend:</b> {legend_name}<br>
            <b>Color:</b> <span style="color: {color}; font-size: 14px;">████</span> {color}<br>
            <b>Style:</b> {style}<br>
            <b>Marker:</b> {marker}<br>
            <b>Transparency:</b> {alpha:.2f} ({int(alpha*100)}%)<br>
            <b>Y-Axis:</b> {y_axis} | <b>X-Axis:</b> {x_axis}<br>
            <b>Bring to Front:</b> {bring_to_front}
        </div>
        """
        
        self.preview_label.setText(preview_text)
    
    def _update_properties(self):
        """Update channel properties from UI"""
        # Update common properties first
        self._update_common_properties()
        
        # Update line-specific properties
        self.config_object.color = self.color_button.get_color()
        self.config_object.style = self.style_combo.currentData()
        self.config_object.marker = self.marker_combo.currentData()
        
        # Y-axis
        self.config_object.yaxis = 'left' if self.left_axis_radio.isChecked() else 'right'
    
    def _restore_properties(self):
        """Restore original channel properties"""
        for key, value in self.original_properties.items():
            setattr(self.config_object, key, value)
    
    @staticmethod
    def edit_channel(channel: Channel, parent=None) -> bool:
        """
        Static method to edit a channel's line properties
        
        Args:
            channel: Channel object to edit
            parent: Parent widget
            
        Returns:
            True if changes were applied, False if cancelled
        """
        dialog = LineWizard(channel, parent)
        result = dialog.exec()
        return result == QDialog.Accepted


# Convenience function for opening the line wizard
def open_line_wizard(channel: Channel, parent=None) -> bool:
    """
    Open the line wizard for editing a channel's line properties
    
    Args:
        channel: Channel object to edit
        parent: Parent widget
        
    Returns:
        True if changes were applied, False if cancelled
    """
    return LineWizard.edit_channel(channel, parent) 