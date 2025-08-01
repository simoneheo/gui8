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
    color_changed = Signal(str)
    def __init__(self, initial_color="#1f77b4"):
        super().__init__()
        self.current_color = initial_color
        self.setFixedSize(40, 24)
        self.update_button_color()
        self.clicked.connect(self.select_color)
    def update_button_color(self):
        self.setStyleSheet(f"background-color: {self.current_color}; border: 1px solid #333;")
    def select_color(self):
        color = QColorDialog.getColor(QColor(self.current_color), self)
        if color.isValid():
            self.current_color = color.name()
            self.update_button_color()
            self.color_changed.emit(self.current_color)
    def set_color(self, color: str):
        self.current_color = color
        self.update_button_color()
    def get_color(self) -> str:
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
        self.setFixedSize(400, 300)  # Reduced height since we removed preview box
        
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
        
        # Main form
        form_layout = QFormLayout()
        
        # Legend Name
        self.legend_edit = QLineEdit()
        self.legend_edit.setPlaceholderText("Enter legend label...")
        form_layout.addRow("Legend Name:", self.legend_edit)
        
        # Color Selection
        self.color_button = ColorButton()
        self.color_button.color_changed.connect(self.on_color_button_changed)
        form_layout.addRow("Line Color:", self.color_button)
        
        # Line Style - Show symbols instead of names
        self.style_combo = QComboBox()
        # Map symbols to values for display
        style_symbols = {
            '─': 'Solid',
            '┄┄': 'Dashed', 
            '┄─┄': 'Dash-dot',
            '┄┄┄': 'Dotted',
            '━━━': 'Solid (thick)',
            '┅┅┅': 'Dashed (long)',
            '┄┄┄┄': 'Dotted (sparse)',
            '┄─┄─': 'Dash-dot-dot',
            '━┄━┄': 'Dash-dash-dot',
            'None': 'None'
        }
        for symbol, name in style_symbols.items():
            if name in PLOT_STYLES['Line Styles']:
                self.style_combo.addItem(symbol, PLOT_STYLES['Line Styles'][name])
            elif name == 'Solid (thick)':
                self.style_combo.addItem(symbol, '-')  # Will be handled by linewidth
            elif name == 'Dashed (long)':
                self.style_combo.addItem(symbol, '--')  # Will be handled by dash pattern
            elif name == 'Dotted (sparse)':
                self.style_combo.addItem(symbol, ':')  # Will be handled by dot spacing
            elif name == 'Dash-dot-dot':
                self.style_combo.addItem(symbol, '-.')  # Will be handled by custom pattern
            elif name == 'Dash-dash-dot':
                self.style_combo.addItem(symbol, '--')  # Will be handled by custom pattern
        form_layout.addRow("Line Style:", self.style_combo)
        
        # Marker - Show symbols instead of names
        self.marker_combo = QComboBox()
        # Map symbols to values for display
        marker_symbols = {
            '●': 'Circle',
            '■': 'Square',
            '▲': 'Triangle',
            '◆': 'Diamond',
            '+': 'Plus',
            '✕': 'X',
            '★': 'Star',
            'None': 'None'
        }
        for symbol, name in marker_symbols.items():
            self.marker_combo.addItem(symbol, PLOT_STYLES['Markers'][name])
        form_layout.addRow("Marker:", self.marker_combo)
        
        # Bring to Front checkbox
        self.bring_to_front_checkbox = QCheckBox("Bring to Front")
        self.bring_to_front_checkbox.setToolTip("Bring this line to the top layer in the plot")
        form_layout.addRow("", self.bring_to_front_checkbox)
        
        layout.addLayout(form_layout)
        
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
    
    def load_channel_properties(self):
        """Load current channel properties into the UI"""
        # Legend name
        self.legend_edit.setText(self.channel.legend_label or "")
        
        # Color
        if self.channel.color:
            self.color_button.set_color(self.channel.color)
        
        # Line style - Find matching style and set to symbol
        if self.channel.style:
            for i in range(self.style_combo.count()):
                if self.style_combo.itemData(i) == self.channel.style:
                    self.style_combo.setCurrentIndex(i)
                    break
        
        # Marker - Find matching marker and set to symbol
        if self.channel.marker:
            for i in range(self.marker_combo.count()):
                if self.marker_combo.itemData(i) == self.channel.marker:
                    self.marker_combo.setCurrentIndex(i)
                    break
        
        # Z-order (Bring to Front)
        z_order = getattr(self.channel, 'z_order', 0)
        self.bring_to_front_checkbox.setChecked(z_order > 0)
    
    def on_color_button_changed(self, color: str):
        """Handle color button changes"""
        pass
    
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
        
        # Always use left Y-axis (no axis selection option)
        self.channel.yaxis = "y-left"
        
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