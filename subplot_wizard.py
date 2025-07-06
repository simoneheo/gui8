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
        
        # Legend tab
        legend_tab = self._create_legend_tab()
        tab_widget.addTab(legend_tab, "Legend")
        
        # Ticks tab
        ticks_tab = self._create_ticks_tab()
        tab_widget.addTab(ticks_tab, "Ticks")
        
        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tab_widget.addTab(advanced_tab, "Advanced")
        
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
        
        # X Label
        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.setPlaceholderText("Enter X-axis label...")
        layout.addRow("X Label:", self.xlabel_edit)
        
        # Y Label (Left)
        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("Enter left Y-axis label...")
        layout.addRow("Y Label (Left):", self.ylabel_edit)
        
        # Y Label (Right)
        self.y_right_label_edit = QLineEdit()
        self.y_right_label_edit.setPlaceholderText("Enter right Y-axis label...")
        layout.addRow("Y Label (Right):", self.y_right_label_edit)
        
        return tab
    
    def _create_axis_tab(self) -> QWidget:
        """Create the axis settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Axis Position Controls
        position_group = QGroupBox("Axis Position")
        position_layout = QFormLayout(position_group)
        
        # X-Axis Position
        x_axis_layout = QHBoxLayout()
        self.x_axis_button_group = QButtonGroup()
        self.bottom_x_axis_checkbox = QCheckBox("Bottom")
        self.top_x_axis_checkbox = QCheckBox("Top")
        
        x_axis_layout.addWidget(self.bottom_x_axis_checkbox)
        x_axis_layout.addWidget(self.top_x_axis_checkbox)
        position_layout.addRow("X-Axis:", x_axis_layout)
        
        # Y-Axis Position
        y_axis_layout = QHBoxLayout()
        self.y_axis_button_group = QButtonGroup()
        self.left_y_axis_checkbox = QCheckBox("Left")
        self.right_y_axis_checkbox = QCheckBox("Right")
        
        y_axis_layout.addWidget(self.left_y_axis_checkbox)
        y_axis_layout.addWidget(self.right_y_axis_checkbox)
        position_layout.addRow("Y-Axis:", y_axis_layout)
        
        layout.addWidget(position_group)
        
        # Scale Settings
        scale_group = QGroupBox("Scale Settings")
        scale_layout = QFormLayout(scale_group)
        
        # X Scale
        self.x_scale_combo = QComboBox()
        scales = ['linear', 'log', 'symlog', 'logit']
        self.x_scale_combo.addItems(scales)
        scale_layout.addRow("X Scale:", self.x_scale_combo)
        
        # Y Scale
        self.y_scale_combo = QComboBox()
        self.y_scale_combo.addItems(scales)
        scale_layout.addRow("Y Scale:", self.y_scale_combo)
        
        layout.addWidget(scale_group)
        
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
        
        return tab
    
    def _create_legend_tab(self) -> QWidget:
        """Create the legend settings tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Show Legend
        self.show_legend_checkbox = QCheckBox("Show Legend")
        self.show_legend_checkbox.setChecked(True)
        layout.addRow("", self.show_legend_checkbox)
        
        # Show Colorbar
        self.show_colorbar_checkbox = QCheckBox("Show Colorbar")
        self.show_colorbar_checkbox.setChecked(True)
        layout.addRow("", self.show_colorbar_checkbox)
        
        # Legend Position
        self.legend_pos_combo = QComboBox()
        positions = ['upper right', 'upper left', 'lower left', 'lower right', 
                    'right', 'center left', 'center right', 'lower center', 
                    'upper center', 'center']
        self.legend_pos_combo.addItems(positions)
        layout.addRow("Legend Position:", self.legend_pos_combo)
        
        # Colorbar Position
        self.colorbar_pos_combo = QComboBox()
        colorbar_positions = ['bottom', 'top', 'left', 'right']
        self.colorbar_pos_combo.addItems(colorbar_positions)
        layout.addRow("Colorbar Position:", self.colorbar_pos_combo)
        
        # Legend Font Size
        self.legend_fontsize_spin = QSpinBox()
        self.legend_fontsize_spin.setRange(6, 24)
        self.legend_fontsize_spin.setValue(10)
        layout.addRow("Legend Font Size:", self.legend_fontsize_spin)
        
        # Legend Columns
        self.legend_ncol_spin = QSpinBox()
        self.legend_ncol_spin.setRange(1, 10)
        self.legend_ncol_spin.setValue(1)
        layout.addRow("Legend Columns:", self.legend_ncol_spin)
        
        # Legend Frame
        self.legend_frameon_checkbox = QCheckBox("Legend Frame")
        self.legend_frameon_checkbox.setChecked(True)
        layout.addRow("", self.legend_frameon_checkbox)
        
        return tab
    
    def _create_ticks_tab(self) -> QWidget:
        """Create the ticks settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Tick Display
        display_group = QGroupBox("Tick Display")
        display_layout = QFormLayout(display_group)
        
        # Show tick labels
        self.show_x_ticks_checkbox = QCheckBox("Show X Tick Labels")
        self.show_x_ticks_checkbox.setChecked(True)
        display_layout.addRow("", self.show_x_ticks_checkbox)
        
        self.show_y_ticks_checkbox = QCheckBox("Show Y Tick Labels")
        self.show_y_ticks_checkbox.setChecked(True)
        display_layout.addRow("", self.show_y_ticks_checkbox)
        
        layout.addWidget(display_group)
        
        # Tick Spacing
        spacing_group = QGroupBox("Tick Spacing")
        spacing_layout = QFormLayout(spacing_group)
        
        # Major tick spacing
        self.major_tick_spacing_combo = QComboBox()
        spacing_options = ['auto', 'custom']
        self.major_tick_spacing_combo.addItems(spacing_options)
        spacing_layout.addRow("Major Tick Spacing:", self.major_tick_spacing_combo)
        
        # Custom tick value
        self.major_tick_value_spin = QDoubleSpinBox()
        self.major_tick_value_spin.setRange(0.001, 1000.0)
        self.major_tick_value_spin.setValue(1.0)
        self.major_tick_value_spin.setEnabled(False)
        spacing_layout.addRow("Custom Tick Value:", self.major_tick_value_spin)
        
        # Minor ticks
        self.minor_ticks_checkbox = QCheckBox("Show Minor Ticks")
        self.minor_ticks_checkbox.setChecked(False)
        spacing_layout.addRow("", self.minor_ticks_checkbox)
        
        layout.addWidget(spacing_group)
        
        # Tick Formatting
        format_group = QGroupBox("Tick Formatting")
        format_layout = QFormLayout(format_group)
        
        # Tick direction
        self.tick_direction_combo = QComboBox()
        directions = ['in', 'out', 'inout']
        self.tick_direction_combo.addItems(directions)
        format_layout.addRow("Tick Direction:", self.tick_direction_combo)
        
        # Tick length
        self.tick_length_spin = QDoubleSpinBox()
        self.tick_length_spin.setRange(1.0, 20.0)
        self.tick_length_spin.setValue(4.0)
        self.tick_length_spin.setSuffix(" pts")
        format_layout.addRow("Tick Length:", self.tick_length_spin)
        
        # Tick width
        self.tick_width_spin = QDoubleSpinBox()
        self.tick_width_spin.setRange(0.1, 5.0)
        self.tick_width_spin.setValue(1.0)
        self.tick_width_spin.setSuffix(" pts")
        format_layout.addRow("Tick Width:", self.tick_width_spin)
        
        layout.addWidget(format_group)
        
        # Connect spacing combo to enable/disable custom value
        self.major_tick_spacing_combo.currentTextChanged.connect(self._on_tick_spacing_changed)
        
        return tab
    
    def _create_advanced_tab(self) -> QWidget:
        """Create the advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Grid Settings
        grid_group = QGroupBox("Grid Settings")
        grid_layout = QFormLayout(grid_group)
        
        # Grid
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        grid_layout.addRow("", self.grid_checkbox)
        
        # Grid Alpha
        self.grid_alpha_spin = QDoubleSpinBox()
        self.grid_alpha_spin.setRange(0.0, 1.0)
        self.grid_alpha_spin.setSingleStep(0.1)
        self.grid_alpha_spin.setValue(0.3)
        self.grid_alpha_spin.setDecimals(2)
        grid_layout.addRow("Grid Transparency:", self.grid_alpha_spin)
        
        layout.addWidget(grid_group)
        
        # Spine Settings
        spine_group = QGroupBox("Spine Settings")
        spine_layout = QFormLayout(spine_group)
        
        # Spine visibility
        self.top_spine_checkbox = QCheckBox("Top Spine")
        self.top_spine_checkbox.setChecked(True)
        spine_layout.addRow("", self.top_spine_checkbox)
        
        self.bottom_spine_checkbox = QCheckBox("Bottom Spine")
        self.bottom_spine_checkbox.setChecked(True)
        spine_layout.addRow("", self.bottom_spine_checkbox)
        
        self.left_spine_checkbox = QCheckBox("Left Spine")
        self.left_spine_checkbox.setChecked(True)
        spine_layout.addRow("", self.left_spine_checkbox)
        
        self.right_spine_checkbox = QCheckBox("Right Spine")
        self.right_spine_checkbox.setChecked(True)
        spine_layout.addRow("", self.right_spine_checkbox)
        
        layout.addWidget(spine_group)
        
        layout.addStretch()
        
        return tab
    
    def _on_tick_spacing_changed(self, spacing_type):
        """Handle tick spacing change"""
        self.major_tick_value_spin.setEnabled(spacing_type == 'custom')
    
    def load_current_config(self):
        """Load current subplot configuration into the UI"""
        # Basic settings
        self.xlabel_edit.setText(self.subplot_config.get('xlabel', ''))
        self.ylabel_edit.setText(self.subplot_config.get('ylabel', ''))
        self.y_right_label_edit.setText(self.subplot_config.get('y_right_label', ''))
        self.title_edit.setText(self.subplot_config.get('title', ''))
        
        # Axis position settings
        self.bottom_x_axis_checkbox.setChecked(self.subplot_config.get('show_bottom_x_axis', True))
        self.top_x_axis_checkbox.setChecked(self.subplot_config.get('show_top_x_axis', False))
        self.left_y_axis_checkbox.setChecked(self.subplot_config.get('show_left_y_axis', True))
        self.right_y_axis_checkbox.setChecked(self.subplot_config.get('show_right_y_axis', False))
        
        # Legend settings
        self.show_legend_checkbox.setChecked(self.subplot_config.get('show_legend', True))
        self.show_colorbar_checkbox.setChecked(self.subplot_config.get('show_colorbar', True))
        self.legend_pos_combo.setCurrentText(self.subplot_config.get('legend_position', 'upper right'))
        self.colorbar_pos_combo.setCurrentText(self.subplot_config.get('colorbar_position', 'bottom'))
        self.legend_fontsize_spin.setValue(self.subplot_config.get('legend_fontsize', 10))
        self.legend_ncol_spin.setValue(self.subplot_config.get('legend_ncol', 1))
        self.legend_frameon_checkbox.setChecked(self.subplot_config.get('legend_frameon', True))
        
        # Advanced settings
        self.x_scale_combo.setCurrentText(self.subplot_config.get('x_scale', 'linear'))
        self.y_scale_combo.setCurrentText(self.subplot_config.get('y_scale', 'linear'))
        
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
        
        # Tick settings
        self.show_x_ticks_checkbox.setChecked(self.subplot_config.get('show_x_tick_labels', True))
        self.show_y_ticks_checkbox.setChecked(self.subplot_config.get('show_y_tick_labels', True))
        self.major_tick_spacing_combo.setCurrentText(self.subplot_config.get('major_tick_spacing', 'auto'))
        self.major_tick_value_spin.setValue(self.subplot_config.get('major_tick_value', 1.0))
        self.minor_ticks_checkbox.setChecked(self.subplot_config.get('minor_ticks_on', False))
        
        # Tick formatting
        self.tick_direction_combo.setCurrentText(self.subplot_config.get('tick_direction', 'in'))
        self.tick_length_spin.setValue(self.subplot_config.get('tick_length', 4.0))
        self.tick_width_spin.setValue(self.subplot_config.get('tick_width', 1.0))
        
        # Grid settings
        self.grid_checkbox.setChecked(self.subplot_config.get('grid', True))
        self.grid_alpha_spin.setValue(self.subplot_config.get('grid_alpha', 0.3))
        
        # Spine settings
        self.top_spine_checkbox.setChecked(self.subplot_config.get('show_top_spine', True))
        self.bottom_spine_checkbox.setChecked(self.subplot_config.get('show_bottom_spine', True))
        self.left_spine_checkbox.setChecked(self.subplot_config.get('show_left_spine', True))
        self.right_spine_checkbox.setChecked(self.subplot_config.get('show_right_spine', True))
    
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
        self.subplot_config['y_right_label'] = self.y_right_label_edit.text()
        self.subplot_config['title'] = self.title_edit.text()
        
        # Axis position settings
        self.subplot_config['show_bottom_x_axis'] = self.bottom_x_axis_checkbox.isChecked()
        self.subplot_config['show_top_x_axis'] = self.top_x_axis_checkbox.isChecked()
        self.subplot_config['show_left_y_axis'] = self.left_y_axis_checkbox.isChecked()
        self.subplot_config['show_right_y_axis'] = self.right_y_axis_checkbox.isChecked()
        
        # Legend settings
        self.subplot_config['show_legend'] = self.show_legend_checkbox.isChecked()
        self.subplot_config['show_colorbar'] = self.show_colorbar_checkbox.isChecked()
        self.subplot_config['legend_position'] = self.legend_pos_combo.currentText()
        self.subplot_config['colorbar_position'] = self.colorbar_pos_combo.currentText()
        self.subplot_config['legend_fontsize'] = self.legend_fontsize_spin.value()
        self.subplot_config['legend_ncol'] = self.legend_ncol_spin.value()
        self.subplot_config['legend_frameon'] = self.legend_frameon_checkbox.isChecked()
        
        # Advanced settings
        self.subplot_config['x_scale'] = self.x_scale_combo.currentText()
        self.subplot_config['y_scale'] = self.y_scale_combo.currentText()
        
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
        
        # Tick settings
        self.subplot_config['show_x_tick_labels'] = self.show_x_ticks_checkbox.isChecked()
        self.subplot_config['show_y_tick_labels'] = self.show_y_ticks_checkbox.isChecked()
        self.subplot_config['major_tick_spacing'] = self.major_tick_spacing_combo.currentText()
        self.subplot_config['major_tick_value'] = self.major_tick_value_spin.value()
        self.subplot_config['minor_ticks_on'] = self.minor_ticks_checkbox.isChecked()
        
        # Tick formatting
        self.subplot_config['tick_direction'] = self.tick_direction_combo.currentText()
        self.subplot_config['tick_length'] = self.tick_length_spin.value()
        self.subplot_config['tick_width'] = self.tick_width_spin.value()
        
        # Grid settings
        self.subplot_config['grid'] = self.grid_checkbox.isChecked()
        self.subplot_config['grid_alpha'] = self.grid_alpha_spin.value()
        
        # Spine settings
        self.subplot_config['show_top_spine'] = self.top_spine_checkbox.isChecked()
        self.subplot_config['show_bottom_spine'] = self.bottom_spine_checkbox.isChecked()
        self.subplot_config['show_left_spine'] = self.left_spine_checkbox.isChecked()
        self.subplot_config['show_right_spine'] = self.right_spine_checkbox.isChecked()
    
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