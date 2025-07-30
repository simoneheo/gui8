"""
Pair Marker Wizard

A focused dialog for editing visual marker properties of comparison pairs.
Designed specifically for styling scatter points in comparison plots.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QLineEdit, QComboBox, QColorDialog, QFrame,
    QGroupBox, QDialogButtonBox, QSpinBox, QDoubleSpinBox, 
    QSlider, QCheckBox
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
from typing import Optional, Dict, Any


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


class PairMarkerWizard(QDialog):
    """
    Focused dialog for editing visual marker properties of comparison pairs.
    
    Features:
    - Marker shape, color, size, transparency
    - Legend label customization
    - Z-order control (bring to front)
    """
    
    # Signals
    marker_updated = Signal(dict)  # marker properties when changes are applied
    pair_name_changed = Signal(str)  # pair name when changed
    
    def __init__(self, pair_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.pair_config = pair_config
        self.original_properties = self._backup_marker_properties()
        
        pair_name = pair_config.get('name', 'Unnamed Pair')
        self.setWindowTitle(f"Pair Styling - {pair_name}")
        self.setModal(True)
        self.setFixedSize(450, 350)
        
        self.init_ui()
        self.load_marker_properties()
    
    def _backup_marker_properties(self) -> dict:
        """Backup original marker properties for cancel operation"""
        return {
            'marker_type': self.pair_config.get('marker_type', 'o'),
            'marker_color': self.pair_config.get('marker_color', '🔵 Blue'),
            'marker_color_hex': self.pair_config.get('marker_color_hex', '#1f77b4'),
            'marker_size': self.pair_config.get('marker_size', 50),
            'marker_alpha': self.pair_config.get('marker_alpha', 0.8),
            'legend_label': self.pair_config.get('legend_label', ''),
            'z_order': self.pair_config.get('z_order', 0),
        }
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Main form
        form_layout = QFormLayout()
        
        # Marker Shape
        self._create_marker_shape_controls(form_layout)
        
        # Marker Color
        self._create_color_controls(form_layout)
        
        # Marker Size
        self._create_size_controls(form_layout)
        
        # Transparency
        self._create_transparency_controls(form_layout)
        
        # Legend Label
        self._create_legend_controls(form_layout)
        
        # Bring to Front checkbox
        self._create_zorder_controls(form_layout)
        
        layout.addLayout(form_layout)
        
        # Dialog buttons
        self._create_dialog_buttons(layout)
            
    
    def _create_marker_shape_controls(self, form_layout: QFormLayout):
        """Create marker shape selection controls"""
        self.marker_combo = QComboBox()
        marker_types = [
            ('○', 'o'),
            ('□', 's'),
            ('△', '^'),
            ('◇', 'D'),
            ('▽', 'v'),
            ('◁', '<'),
            ('▷', '>'),
            ('⬟', 'p'),
            ('✦', '*'),
            ('⬢', 'h'),
            ('+', '+'),
            ('×', 'x'),
            ('|', '|'),
            ('—', '_')
        ]
        for name, value in marker_types:
            self.marker_combo.addItem(name, value)
        form_layout.addRow("Marker Shape:", self.marker_combo)
    
    def _create_color_controls(self, form_layout: QFormLayout):
        """Create color selection controls"""
        color_layout = QHBoxLayout()
        self.color_button = ColorButton()
        self.color_combo = QComboBox()
        color_options = [
            ('🔵 Blue', '#1f77b4'),
            ('🔴 Red', '#d62728'),
            ('🟢 Green', '#2ca02c'),
            ('🟣 Purple', '#9467bd'),
            ('🟠 Orange', '#ff7f0e'),
            ('🟤 Brown', '#8c564b'),
            ('🩷 Pink', '#e377c2'),
            ('⚫ Gray', '#7f7f7f'),
            ('🟡 Yellow', '#bcbd22'),
            ('🔶 Cyan', '#17becf'),
            ('Custom', 'custom')
        ]
        for name, value in color_options:
            self.color_combo.addItem(name, value)
        self.color_combo.currentTextChanged.connect(self.on_color_combo_changed)
        self.color_button.color_changed.connect(self.on_color_button_changed)
        
        color_layout.addWidget(self.color_button)
        color_layout.addWidget(self.color_combo)
        form_layout.addRow("Marker Color:", color_layout)
    
    def _create_size_controls(self, form_layout: QFormLayout):
        """Create marker size controls"""
        size_layout = QHBoxLayout()
        self.size_spin = QSpinBox()
        self.size_spin.setRange(6, 200)
        self.size_spin.setValue(50)
        self.size_spin.setSuffix(" pts")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(6, 200)
        self.size_slider.setValue(50)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.size_spin.setValue)
        
        size_layout.addWidget(self.size_spin)
        size_layout.addWidget(self.size_slider)
        form_layout.addRow("Marker Size:", size_layout)
    
    def _create_transparency_controls(self, form_layout: QFormLayout):
        """Create transparency controls"""
        alpha_layout = QHBoxLayout()
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(0.8)
        self.alpha_spin.setDecimals(2)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(80)
        self.alpha_spin.valueChanged.connect(lambda v: self.alpha_slider.setValue(int(v * 100)))
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_spin.setValue(v / 100.0))
        
        alpha_layout.addWidget(self.alpha_spin)
        alpha_layout.addWidget(self.alpha_slider)
        form_layout.addRow("Transparency:", alpha_layout)
    
    def _create_legend_controls(self, form_layout: QFormLayout):
        """Create pair name controls"""
        self.legend_label_edit = QLineEdit()
        self.legend_label_edit.setPlaceholderText("Custom pair name (optional)")
        self.legend_label_edit.textChanged.connect(self._on_pair_name_changed)
        form_layout.addRow("Pair Name:", self.legend_label_edit)
    
    def _create_zorder_controls(self, form_layout: QFormLayout):
        """Create z-order (bring to front) controls"""
        self.bring_to_front_checkbox = QCheckBox("Bring to Front")
        self.bring_to_front_checkbox.setToolTip("Bring this pair's markers to the top layer in the plot")
        form_layout.addRow("", self.bring_to_front_checkbox)
    
    def _create_dialog_buttons(self, layout: QVBoxLayout):
        """Create dialog buttons"""
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(button_box)
    
    def load_marker_properties(self):
        """Load current marker properties into the UI"""
        # Marker type
        marker_type = self.pair_config.get('marker_type', 'o')
        idx = self.marker_combo.findData(marker_type)
        if idx >= 0:
            self.marker_combo.setCurrentIndex(idx)
        else:
            # Fallback to text search
            idx = self.marker_combo.findText(marker_type)
            if idx >= 0:
                self.marker_combo.setCurrentIndex(idx)
            else:
                self.marker_combo.setCurrentIndex(0)  # Default to circle
        
        # Color
        marker_color = self.pair_config.get('marker_color', '🔵 Blue')
        idx = self.color_combo.findText(marker_color)
        if idx >= 0:
            self.color_combo.setCurrentIndex(idx)
        else:
            self.color_combo.setCurrentText('Custom')
            hex_color = self.pair_config.get('marker_color_hex', '#1f77b4')
            self.color_button.set_color(hex_color)
        
        # Size
        size = self.pair_config.get('marker_size', 50)
        self.size_spin.setValue(size)
        
        # Alpha
        alpha = self.pair_config.get('marker_alpha', 0.8)
        self.alpha_spin.setValue(alpha)
        
        # Legend label
        legend_label = self.pair_config.get('legend_label', '')
        self.legend_label_edit.setText(legend_label)
        
        # Also load from name field if available
        pair_name = self.pair_config.get('name', '')
        if pair_name and not legend_label:
            self.legend_label_edit.setText(pair_name)
        
        # Z-order (Bring to Front)
        z_order = self.pair_config.get('z_order', 0)
        self.bring_to_front_checkbox.setChecked(z_order > 0)
    
    def on_color_combo_changed(self, color_name: str):
        """Handle color combo box changes"""
        if color_name == 'Custom':
            return
        
        color_map = {
            '🔵 Blue': '#1f77b4',
            '🔴 Red': '#d62728',
            '🟢 Green': '#2ca02c',
            '🟣 Purple': '#9467bd',
            '🟠 Orange': '#ff7f0e',
            '🟤 Brown': '#8c564b',
            '🩷 Pink': '#e377c2',
            '⚫ Gray': '#7f7f7f',
            '🟡 Yellow': '#bcbd22',
            '🔶 Cyan': '#17becf'
        }
        
        if color_name in color_map:
            self.color_button.set_color(color_map[color_name])
    
    def on_color_button_changed(self, color: str):
        """Handle color button changes"""
        self.color_combo.setCurrentText('Custom')
    
    def _on_pair_name_changed(self, new_name: str):
        """Handle pair name text field changes"""
        self.pair_name_changed.emit(new_name)
    
    def apply_changes(self):
        """Apply changes to the pair config without closing dialog"""
        self._update_marker_properties()
        self.marker_updated.emit(self.pair_config)
    
    def accept(self):
        """Apply changes and close dialog"""
        self._update_marker_properties()
        self.marker_updated.emit(self.pair_config)
        super().accept()
    
    def reject(self):
        """Cancel changes and restore original properties"""
        self._restore_marker_properties()
        super().reject()
    
    def _update_marker_properties(self):
        """Update pair config with marker properties from UI"""
        # Marker type
        self.pair_config['marker_type'] = self.marker_combo.currentData()
        
        # Color
        self.pair_config['marker_color'] = self.color_combo.currentText()
        self.pair_config['marker_color_hex'] = self.color_button.get_color()
        
        # Size and transparency
        self.pair_config['marker_size'] = self.size_spin.value()
        self.pair_config['marker_alpha'] = self.alpha_spin.value()
        
        # Legend label (now pair name)
        self.pair_config['legend_label'] = self.legend_label_edit.text()
        
        # Z-order (Bring to Front)
        self.pair_config['z_order'] = 10 if self.bring_to_front_checkbox.isChecked() else 0
        
        # Update modification time
        from datetime import datetime
        self.pair_config['modified_at'] = datetime.now().isoformat()
    
    def _restore_marker_properties(self):
        """Restore original marker properties"""
        for key, value in self.original_properties.items():
            self.pair_config[key] = value
    
    @staticmethod
    def edit_pair_marker(pair_config: Dict[str, Any], parent=None) -> bool:
        """
        Static method to edit a pair's marker properties
        
        Args:
            pair_config: Dictionary containing pair configuration
            parent: Parent widget
            
        Returns:
            True if changes were applied, False if cancelled
        """
        dialog = PairMarkerWizard(pair_config, parent)
        result = dialog.exec()
        return result == QDialog.Accepted


# Convenience function for opening the pair marker wizard
def open_pair_marker_wizard(pair_config: Dict[str, Any], parent=None) -> bool:
    """
    Open the pair marker wizard for editing a comparison pair's marker properties
    
    Args:
        pair_config: Dictionary containing pair configuration
        parent: Parent widget
        
    Returns:
        True if changes were applied, False if cancelled
    """
    return PairMarkerWizard.edit_pair_marker(pair_config, parent) 