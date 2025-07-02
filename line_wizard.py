from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QLineEdit, QComboBox, QColorDialog, QFrame,
    QGroupBox, QButtonGroup, QRadioButton, QDialogButtonBox, QCheckBox
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QPalette
from typing import Optional

from channel import Channel
from plot_manager import PLOT_STYLES


class ColorButton(QPushButton):
    """Custom button that displays and allows selection of colors"""
    
    color_changed = Signal(str)  # hex color string
    
    def __init__(self, initial_color: str = "#1f77b4"):
        super().__init__()
        self.current_color = initial_color
        self.setFixedSize(40, 30)
        self.update_button_color()
        self.clicked.connect(self.select_color)
    
    def update_button_color(self):
        """Update button appearance to show current color"""
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.current_color};
                border: 2px solid #333;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid #666;
            }}
        """)
    
    def select_color(self):
        """Open color dialog and update color"""
        color = QColorDialog.getColor(QColor(self.current_color), self)
        if color.isValid():
            self.current_color = color.name()
            self.update_button_color()
            self.color_changed.emit(self.current_color)
    
    def set_color(self, color: str):
        """Set color programmatically"""
        self.current_color = color
        self.update_button_color()
    
    def get_color(self) -> str:
        """Get current color as hex string"""
        return self.current_color


class LineWizard(QDialog):
    """
    Dialog for editing channel line properties
    """
    
    # Signals
    channel_updated = Signal(str)  # channel_id when changes are applied
    
    def __init__(self, channel: Channel, parent=None):
        super().__init__(parent)
        self.channel = channel
        self.original_properties = self._backup_channel_properties()
        
        self.setWindowTitle(f"Line Properties - {channel.ylabel or 'Unnamed'}")
        self.setModal(True)
        self.setFixedSize(400, 500)
        
        self.init_ui()
        self.load_channel_properties()
    
    def _backup_channel_properties(self) -> dict:
        """Backup original channel properties for cancel operation"""
        return {
            'color': self.channel.color,
            'style': self.channel.style,
            'marker': self.channel.marker,
            'yaxis': self.channel.yaxis,
            'legend_label': self.channel.legend_label,
            'z_order': getattr(self.channel, 'z_order', 0)
        }
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"Editing: {self.channel.ylabel or 'Unnamed Channel'}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Main form
        form_layout = QFormLayout()
        
        # Legend Name
        self.legend_edit = QLineEdit()
        self.legend_edit.setPlaceholderText("Enter legend label...")
        form_layout.addRow("Legend Name:", self.legend_edit)
        
        # Color Selection
        color_layout = QHBoxLayout()
        self.color_button = ColorButton()
        self.color_combo = QComboBox()
        self.color_combo.addItems(PLOT_STYLES['Colors'].keys())
        self.color_combo.currentTextChanged.connect(self.on_color_combo_changed)
        self.color_button.color_changed.connect(self.on_color_button_changed)
        
        color_layout.addWidget(self.color_button)
        color_layout.addWidget(self.color_combo)
        form_layout.addRow("Line Color:", color_layout)
        
        # Line Style
        self.style_combo = QComboBox()
        for name, value in PLOT_STYLES['Line Styles'].items():
            self.style_combo.addItem(name, value)
        form_layout.addRow("Line Style:", self.style_combo)
        
        # Marker
        self.marker_combo = QComboBox()
        for name, value in PLOT_STYLES['Markers'].items():
            self.marker_combo.addItem(name, value)
        form_layout.addRow("Marker:", self.marker_combo)
        
        # Bring to Front checkbox
        self.bring_to_front_checkbox = QCheckBox("Bring to Front")
        self.bring_to_front_checkbox.setToolTip("Bring this line to the top layer in the plot")
        form_layout.addRow("", self.bring_to_front_checkbox)
        
        layout.addLayout(form_layout)
        
        # Axis Selection Group
        axis_group = QGroupBox("Y-Axis")
        axis_layout = QVBoxLayout(axis_group)
        
        self.axis_button_group = QButtonGroup()
        self.left_axis_radio = QRadioButton("Left Axis")
        self.right_axis_radio = QRadioButton("Right Axis")
        
        self.axis_button_group.addButton(self.left_axis_radio, 0)
        self.axis_button_group.addButton(self.right_axis_radio, 1)
        
        axis_layout.addWidget(self.left_axis_radio)
        axis_layout.addWidget(self.right_axis_radio)
        layout.addWidget(axis_group)
        
        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("Line preview will appear here")
        self.preview_label.setMinimumHeight(40)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: white;
                padding: 10px;
                border-radius: 4px;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)
        
        # Spacer
        layout.addStretch()
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(button_box)
        
        # Connect change signals for live preview
        self.legend_edit.textChanged.connect(self.update_preview)
        self.style_combo.currentTextChanged.connect(self.update_preview)
        self.marker_combo.currentTextChanged.connect(self.update_preview)
        self.axis_button_group.buttonToggled.connect(self.update_preview)
        self.bring_to_front_checkbox.toggled.connect(self.update_preview)
    
    def load_channel_properties(self):
        """Load current channel properties into the UI"""
        # Legend name
        self.legend_edit.setText(self.channel.legend_label or "")
        
        # Color
        if self.channel.color:
            self.color_button.set_color(self.channel.color)
            # Find matching color in combo
            for name, value in PLOT_STYLES['Colors'].items():
                if value == self.channel.color:
                    self.color_combo.setCurrentText(name)
                    break
        
        # Line style
        if self.channel.style:
            for i in range(self.style_combo.count()):
                if self.style_combo.itemData(i) == self.channel.style:
                    self.style_combo.setCurrentIndex(i)
                    break
        
        # Marker
        if self.channel.marker:
            for i in range(self.marker_combo.count()):
                if self.marker_combo.itemData(i) == self.channel.marker:
                    self.marker_combo.setCurrentIndex(i)
                    break
        
        # Axis
        if self.channel.yaxis == "y-left":
            self.left_axis_radio.setChecked(True)
        else:
            self.right_axis_radio.setChecked(True)
        
        # Z-order (Bring to Front)
        z_order = getattr(self.channel, 'z_order', 0)
        self.bring_to_front_checkbox.setChecked(z_order > 0)
        
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
        # Update combo to "Custom" or find matching color
        self.color_combo.setCurrentIndex(-1)  # Clear selection for custom color
        self.update_preview()
    
    def update_preview(self):
        """Update the preview display"""
        # Get current settings
        color = self.color_button.get_color()
        style_name = self.style_combo.currentText()
        marker_name = self.marker_combo.currentText()
        axis = "Left" if self.left_axis_radio.isChecked() else "Right"
        legend_name = self.legend_edit.text() or self.channel.ylabel or "Unnamed"
        bring_to_front = "Yes" if self.bring_to_front_checkbox.isChecked() else "No"
        
        # Create preview text
        preview_text = f"""
        <div style="padding: 5px;">
            <b>Legend:</b> {legend_name}<br>
            <b>Color:</b> <span style="color: {color};">████</span> {color}<br>
            <b>Style:</b> {style_name}<br>
            <b>Marker:</b> {marker_name}<br>
            <b>Axis:</b> {axis}<br>
            <b>Bring to Front:</b> {bring_to_front}
        </div>
        """
        
        self.preview_label.setText(preview_text)
    
    def apply_changes(self):
        """Apply changes to the channel without closing dialog"""
        self._update_channel_properties()
        self.channel_updated.emit(self.channel.channel_id)
    
    def accept(self):
        """Apply changes and close dialog"""
        self._update_channel_properties()
        self.channel_updated.emit(self.channel.channel_id)
        super().accept()
    
    def reject(self):
        """Cancel changes and restore original properties"""
        self._restore_channel_properties()
        super().reject()
    
    def _update_channel_properties(self):
        """Update channel properties from UI"""
        # Legend name
        self.channel.legend_label = self.legend_edit.text() or None
        
        # Color
        self.channel.color = self.color_button.get_color()
        
        # Style
        if self.style_combo.currentData():
            self.channel.style = self.style_combo.currentData()
        
        # Marker
        if self.marker_combo.currentData():
            self.channel.marker = self.marker_combo.currentData()
        
        # Handle case where both line and marker are "None"
        if (self.channel.style == "None" and 
            (self.channel.marker == "None" or not self.channel.marker)):
            # If both are None, use a default style to ensure visibility
            self.channel.style = "-"
            self.channel.marker = "None"
        
        # Axis
        self.channel.yaxis = "y-left" if self.left_axis_radio.isChecked() else "y-right"
        
        # Z-order (Bring to Front)
        self.channel.z_order = 10 if self.bring_to_front_checkbox.isChecked() else 0
        
        # Update modification time
        from datetime import datetime
        self.channel.modified_at = datetime.now()
    
    def _restore_channel_properties(self):
        """Restore original channel properties"""
        self.channel.color = self.original_properties['color']
        self.channel.style = self.original_properties['style']
        self.channel.marker = self.original_properties['marker']
        self.channel.yaxis = self.original_properties['yaxis']
        self.channel.legend_label = self.original_properties['legend_label']
        self.channel.z_order = self.original_properties['z_order']
    
    @staticmethod
    def edit_channel(channel: Channel, parent=None) -> bool:
        """
        Static method to edit a channel's properties
        
        Returns:
            True if changes were applied, False if cancelled
        """
        dialog = LineWizard(channel, parent)
        result = dialog.exec()
        return result == QDialog.Accepted


# Convenience function for opening the line wizard
def open_line_wizard(channel: Channel, parent=None) -> bool:
    """
    Open the line wizard for editing a channel
    
    Args:
        channel: Channel to edit
        parent: Parent widget
        
    Returns:
        True if changes were applied, False if cancelled
    """
    return LineWizard.edit_channel(channel, parent) 