from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QLineEdit, QComboBox, QFrame, QGroupBox, 
    QDialogButtonBox, QCheckBox, QDoubleSpinBox, QSpinBox, 
    QSlider, QTabWidget, QWidget, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont
from typing import Dict, Any


class SubplotWizard(QDialog):
    """
    Dialog for editing subplot configuration properties including:
    - Xlabel and Ylabel for all axes
    - Axis position controls (top/bottom x-axis, left/right y-axis)
    - Legend position and settings
    - Tick controls and formatting
    - Advanced axis settings (scales, limits)
    """
    
    # Signals
    subplot_updated = Signal(int, dict)  # subplot_num, config_dict when changes are applied
    
    def __init__(self, subplot_num: int, subplot_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.subplot_num = subplot_num
        self.subplot_config = subplot_config
        self.original_config = subplot_config.copy()
        
        self.setWindowTitle(f"Subplot {subplot_num} Configuration")
        self.setModal(True)
        self.setFixedSize(500, 650)
        
        self.init_ui()
        self.load_current_config()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"Configuring Subplot {self.subplot_num}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Tab widget
        tab_widget = QTabWidget()
        
        # Basic tab
        basic_tab = self._create_basic_tab()
        tab_widget.addTab(basic_tab, "Basic")
        
        # Axis tab
        axis_tab = self._create_axis_tab()
        tab_widget.addTab(axis_tab, "Axis")
        
        layout.addWidget(tab_widget)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(button_box)
    
    def _create_basic_tab(self) -> QWidget:
        """Create the basic settings tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Title
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Enter subplot title...")
        layout.addRow("Title:", self.title_edit)
        
        # Show Legend
        self.show_legend_checkbox = QCheckBox("Show Legend")
        self.show_legend_checkbox.setChecked(True)
        layout.addRow("", self.show_legend_checkbox)
        
        # Legend Position
        self.legend_position_combo = QComboBox()
        legend_positions = [
            ('best', 'Best (Auto)'),
            ('upper right', 'Upper Right'),
            ('upper left', 'Upper Left'),
            ('lower left', 'Lower Left'),
            ('lower right', 'Lower Right'),
            ('center', 'Center'),
            ('upper center', 'Upper Center'),
            ('lower center', 'Lower Center'),
            ('center left', 'Center Left'),
            ('center right', 'Center Right')
        ]
        
        for pos_value, pos_display in legend_positions:
            self.legend_position_combo.addItem(pos_display, pos_value)
        
        layout.addRow("Legend Position:", self.legend_position_combo)
        
        # Connect show legend checkbox to enable/disable position dropdown
        self.show_legend_checkbox.stateChanged.connect(self._on_show_legend_changed)
        
        # X Label
        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.setPlaceholderText("Enter X-axis label...")
        layout.addRow("X Label:", self.xlabel_edit)
        
        # Y Label
        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("Enter y-axis label...")
        layout.addRow("Y Label:", self.ylabel_edit)
        
        return tab
    
    def _create_axis_tab(self) -> QWidget:
        """Create the axis settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Axis Limits
        limits_group = QGroupBox("Axis Limits")
        limits_layout = QFormLayout(limits_group)
        
        # X Limits
        self.xlim_min_edit = QLineEdit()
        self.xlim_min_edit.setPlaceholderText("Auto")
        limits_layout.addRow("X Min:", self.xlim_min_edit)
        
        self.xlim_max_edit = QLineEdit()
        self.xlim_max_edit.setPlaceholderText("Auto")
        limits_layout.addRow("X Max:", self.xlim_max_edit)
        
        # Y Limits
        self.ylim_min_edit = QLineEdit()
        self.ylim_min_edit.setPlaceholderText("Auto")
        limits_layout.addRow("Y Min:", self.ylim_min_edit)
        
        self.ylim_max_edit = QLineEdit()
        self.ylim_max_edit.setPlaceholderText("Auto")
        limits_layout.addRow("Y Max:", self.ylim_max_edit)
        
        layout.addWidget(limits_group)
        
        # Tick Spacing
        spacing_group = QGroupBox("Tick Spacing")
        spacing_layout = QFormLayout(spacing_group)
        
        # Major tick spacing
        self.major_tick_spacing_combo = QComboBox()
        spacing_options = ['auto', 'custom']
        self.major_tick_spacing_combo.addItems(spacing_options)
        spacing_layout.addRow("Major Tick Spacing:", self.major_tick_spacing_combo)
        
        # Custom major tick value
        self.major_tick_value_spin = QDoubleSpinBox()
        self.major_tick_value_spin.setRange(0.001, 1000.0)
        self.major_tick_value_spin.setValue(1.0)
        self.major_tick_value_spin.setEnabled(False)
        spacing_layout.addRow("Major Tick Value:", self.major_tick_value_spin)
        
        # Minor tick spacing
        self.minor_tick_spacing_combo = QComboBox()
        self.minor_tick_spacing_combo.addItems(spacing_options)
        spacing_layout.addRow("Minor Tick Spacing:", self.minor_tick_spacing_combo)
        
        # Custom minor tick value
        self.minor_tick_value_spin = QDoubleSpinBox()
        self.minor_tick_value_spin.setRange(0.001, 100.0)
        self.minor_tick_value_spin.setValue(0.2)
        self.minor_tick_value_spin.setEnabled(False)
        spacing_layout.addRow("Minor Tick Value:", self.minor_tick_value_spin)
        
        # Show minor ticks
        self.minor_ticks_checkbox = QCheckBox("Show Minor Ticks")
        self.minor_ticks_checkbox.setChecked(False)
        spacing_layout.addRow("", self.minor_ticks_checkbox)
        
        layout.addWidget(spacing_group)
        
        # Connect spacing combos to enable/disable custom values
        self.major_tick_spacing_combo.currentTextChanged.connect(self._on_major_tick_spacing_changed)
        self.minor_tick_spacing_combo.currentTextChanged.connect(self._on_minor_tick_spacing_changed)
        
        return tab
    
    def _on_major_tick_spacing_changed(self, spacing_type):
        """Handle major tick spacing change"""
        if spacing_type == 'custom':
            self.major_tick_value_spin.setEnabled(True)
        else:
            self.major_tick_value_spin.setEnabled(False)
    
    def _on_minor_tick_spacing_changed(self, spacing_type):
        """Handle minor tick spacing change"""
        if spacing_type == 'custom':
            self.minor_tick_value_spin.setEnabled(True)
        else:
            self.minor_tick_value_spin.setEnabled(False)
    
    def _on_show_legend_changed(self, state):
        """Handle show legend checkbox change - enable/disable legend position dropdown"""
        self.legend_position_combo.setEnabled(state == Qt.Checked)
    
    def load_current_config(self):
        """Load current subplot configuration into the UI"""
        # Basic settings
        self.xlabel_edit.setText(self.subplot_config.get('xlabel', ''))
        self.ylabel_edit.setText(self.subplot_config.get('ylabel', ''))
        self.title_edit.setText(self.subplot_config.get('title', ''))
        
        # Legend settings
        show_legend = self.subplot_config.get('show_legend', True)
        self.show_legend_checkbox.setChecked(show_legend)
        
        # Set legend position
        legend_position = self.subplot_config.get('legend_position', 'upper right')
        for i in range(self.legend_position_combo.count()):
            if self.legend_position_combo.itemData(i) == legend_position:
                self.legend_position_combo.setCurrentIndex(i)
                break
        
        # Enable/disable legend position dropdown based on show legend checkbox
        self.legend_position_combo.setEnabled(show_legend)
        
        # Axis limits
        xlim = self.subplot_config.get('xlim', [None, None])
        ylim = self.subplot_config.get('ylim', [None, None])
        
        if xlim[0] is not None:
            self.xlim_min_edit.setText(str(xlim[0]))
        if xlim[1] is not None:
            self.xlim_max_edit.setText(str(xlim[1]))
        if ylim[0] is not None:
            self.ylim_min_edit.setText(str(ylim[0]))
        if ylim[1] is not None:
            self.ylim_max_edit.setText(str(ylim[1]))
        
        # Tick settings - individual tick label display
        self.major_tick_spacing_combo.setCurrentText(self.subplot_config.get('major_tick_spacing', 'auto'))
        self.major_tick_value_spin.setValue(self.subplot_config.get('major_tick_value', 1.0))
        self.minor_tick_spacing_combo.setCurrentText(self.subplot_config.get('minor_tick_spacing', 'auto'))
        self.minor_tick_value_spin.setValue(self.subplot_config.get('minor_tick_value', 0.2))
        self.minor_ticks_checkbox.setChecked(self.subplot_config.get('minor_ticks_on', False))
    
    def apply_changes(self):
        """Apply changes to the subplot configuration without closing dialog"""
        self._update_subplot_config()
        self.subplot_updated.emit(self.subplot_num, self.subplot_config)
    
    def accept(self):
        """Apply changes and close dialog"""
        self._update_subplot_config()
        self.subplot_updated.emit(self.subplot_num, self.subplot_config)
        super().accept()
    
    def reject(self):
        """Cancel changes and restore original configuration"""
        self.subplot_config = self.original_config.copy()
        super().reject()
    
    def _update_subplot_config(self):
        """Update the subplot configuration dictionary with current UI values"""
        # Basic settings
        self.subplot_config['xlabel'] = self.xlabel_edit.text()
        self.subplot_config['ylabel'] = self.ylabel_edit.text()
        self.subplot_config['title'] = self.title_edit.text()
        
        # Legend settings
        self.subplot_config['show_legend'] = self.show_legend_checkbox.isChecked()
        self.subplot_config['legend_position'] = self.legend_position_combo.currentData()
        
        # Axis limits
        xlim = [None, None]
        ylim = [None, None]
        
        try:
            if self.xlim_min_edit.text().strip():
                xlim[0] = float(self.xlim_min_edit.text())
        except ValueError:
            pass
        
        try:
            if self.xlim_max_edit.text().strip():
                xlim[1] = float(self.xlim_max_edit.text())
        except ValueError:
            pass
        
        try:
            if self.ylim_min_edit.text().strip():
                ylim[0] = float(self.ylim_min_edit.text())
        except ValueError:
            pass
        
        try:
            if self.ylim_max_edit.text().strip():
                ylim[1] = float(self.ylim_max_edit.text())
        except ValueError:
            pass
        
        self.subplot_config['xlim'] = xlim
        self.subplot_config['ylim'] = ylim
        
        # Tick settings - individual tick label display
        self.subplot_config['major_tick_spacing'] = self.major_tick_spacing_combo.currentText()
        self.subplot_config['major_tick_value'] = self.major_tick_value_spin.value()
        self.subplot_config['minor_tick_spacing'] = self.minor_tick_spacing_combo.currentText()
        self.subplot_config['minor_tick_value'] = self.minor_tick_value_spin.value()
        self.subplot_config['minor_ticks_on'] = self.minor_ticks_checkbox.isChecked()
    
    @staticmethod
    def edit_subplot(subplot_num: int, subplot_config: Dict[str, Any], parent=None) -> bool:
        """
        Static method to edit subplot configuration
        
        Args:
            subplot_num: The subplot number
            subplot_config: Dictionary containing subplot configuration
            parent: Parent widget
            
        Returns:
            bool: True if changes were applied, False if cancelled
        """
        wizard = SubplotWizard(subplot_num, subplot_config, parent)
        return wizard.exec() == QDialog.Accepted


def open_subplot_wizard(subplot_num: int, subplot_config: Dict[str, Any], parent=None) -> bool:
    """
    Convenience function to open the subplot wizard
    
    Args:
        subplot_num: The subplot number
        subplot_config: Dictionary containing subplot configuration
        parent: Parent widget
        
    Returns:
        bool: True if changes were applied, False if cancelled
    """
    return SubplotWizard.edit_subplot(subplot_num, subplot_config, parent) 