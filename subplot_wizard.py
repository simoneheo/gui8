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
        
        # Connect axis position signals to automatically update tick display
        self.top_x_axis_radio.toggled.connect(self._on_axis_position_changed)
        self.bottom_x_axis_radio.toggled.connect(self._on_axis_position_changed)
        self.left_y_axis_radio.toggled.connect(self._on_axis_position_changed)
        self.right_y_axis_radio.toggled.connect(self._on_axis_position_changed)
        
        # Connect layout option signals to handle conflicts
        self.tight_layout_checkbox.toggled.connect(self._on_layout_option_changed)
        self.constrained_layout_checkbox.toggled.connect(self._on_layout_option_changed)
        
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
        
        # X-Axis Position (mutually exclusive radio buttons)
        x_axis_layout = QHBoxLayout()
        self.x_axis_button_group = QButtonGroup()
        self.bottom_x_axis_radio = QRadioButton("Bottom")
        self.top_x_axis_radio = QRadioButton("Top")
        
        # Add radio buttons to button group for mutual exclusion
        self.x_axis_button_group.addButton(self.bottom_x_axis_radio, 0)
        self.x_axis_button_group.addButton(self.top_x_axis_radio, 1)
        
        # Set default selection
        self.bottom_x_axis_radio.setChecked(True)
        
        x_axis_layout.addWidget(self.bottom_x_axis_radio)
        x_axis_layout.addWidget(self.top_x_axis_radio)
        position_layout.addRow("X-Axis:", x_axis_layout)
        
        # Y-Axis Position (mutually exclusive radio buttons)
        y_axis_layout = QHBoxLayout()
        self.y_axis_button_group = QButtonGroup()
        self.left_y_axis_radio = QRadioButton("Left")
        self.right_y_axis_radio = QRadioButton("Right")
        
        # Add radio buttons to button group for mutual exclusion
        self.y_axis_button_group.addButton(self.left_y_axis_radio, 0)
        self.y_axis_button_group.addButton(self.right_y_axis_radio, 1)
        
        # Set default selection
        self.left_y_axis_radio.setChecked(True)
        
        y_axis_layout.addWidget(self.left_y_axis_radio)
        y_axis_layout.addWidget(self.right_y_axis_radio)
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
        
        # Individual tick label controls for each axis position
        self.show_left_ticks_checkbox = QCheckBox("Show Left Tick Labels")
        self.show_left_ticks_checkbox.setChecked(True)
        display_layout.addRow("", self.show_left_ticks_checkbox)
        
        self.show_right_ticks_checkbox = QCheckBox("Show Right Tick Labels")
        self.show_right_ticks_checkbox.setChecked(False)
        display_layout.addRow("", self.show_right_ticks_checkbox)
        
        self.show_top_ticks_checkbox = QCheckBox("Show Top Tick Labels")
        self.show_top_ticks_checkbox.setChecked(False)
        display_layout.addRow("", self.show_top_ticks_checkbox)
        
        self.show_bottom_ticks_checkbox = QCheckBox("Show Bottom Tick Labels") 
        self.show_bottom_ticks_checkbox.setChecked(True)
        display_layout.addRow("", self.show_bottom_ticks_checkbox)
        
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
        
        # Text & Font Settings
        font_group = QGroupBox("Text & Font Settings")
        font_layout = QFormLayout(font_group)
        
        # Label font size
        self.label_fontsize_spin = QSpinBox()
        self.label_fontsize_spin.setRange(6, 48)
        self.label_fontsize_spin.setValue(12)
        font_layout.addRow("Label Font Size:", self.label_fontsize_spin)
        
        # Title font size
        self.title_fontsize_spin = QSpinBox()
        self.title_fontsize_spin.setRange(8, 48)
        self.title_fontsize_spin.setValue(14)
        font_layout.addRow("Title Font Size:", self.title_fontsize_spin)
        
        # Font family
        self.font_family_combo = QComboBox()
        font_families = ['DejaVu Sans', 'Arial', 'Helvetica', 'Times New Roman', 'Courier New', 'Computer Modern']
        self.font_family_combo.addItems(font_families)
        font_layout.addRow("Font Family:", self.font_family_combo)
        
        # Font weight
        self.font_weight_combo = QComboBox()
        weights = ['normal', 'bold', 'light', 'ultralight', 'heavy']
        self.font_weight_combo.addItems(weights)
        font_layout.addRow("Font Weight:", self.font_weight_combo)
        
        layout.addWidget(font_group)
        
        # Layout & Spacing Settings
        layout_group = QGroupBox("Layout & Spacing")
        layout_layout = QFormLayout(layout_group)
        
        # Aspect ratio
        self.aspect_ratio_combo = QComboBox()
        aspects = ['auto', 'equal', '1:1', '4:3', '16:9', '16:10', '3:2']
        self.aspect_ratio_combo.addItems(aspects)
        layout_layout.addRow("Aspect Ratio:", self.aspect_ratio_combo)
        
        # Tight layout
        self.tight_layout_checkbox = QCheckBox("Use Tight Layout")
        self.tight_layout_checkbox.setChecked(True)
        self.tight_layout_checkbox.setToolTip("Automatically adjust subplot spacing for better fit")
        layout_layout.addRow("", self.tight_layout_checkbox)
        
        # Constrained layout
        self.constrained_layout_checkbox = QCheckBox("Use Constrained Layout")
        self.constrained_layout_checkbox.setChecked(False)
        self.constrained_layout_checkbox.setToolTip("Advanced layout engine for better spacing (may conflict with tight layout)")
        layout_layout.addRow("", self.constrained_layout_checkbox)
        
        # Subplot margins
        self.subplot_margin_spin = QDoubleSpinBox()
        self.subplot_margin_spin.setRange(0.0, 1.0)
        self.subplot_margin_spin.setSingleStep(0.02)
        self.subplot_margin_spin.setValue(0.1)
        self.subplot_margin_spin.setDecimals(2)
        self.subplot_margin_spin.setToolTip("Margin around subplot content")
        layout_layout.addRow("Subplot Margin:", self.subplot_margin_spin)
        
        layout.addWidget(layout_group)
        
        # Performance Options
        performance_group = QGroupBox("Performance & Quality")
        performance_layout = QFormLayout(performance_group)
        
        # Anti-aliasing
        self.antialiasing_checkbox = QCheckBox("Enable Anti-aliasing")
        self.antialiasing_checkbox.setChecked(True)
        self.antialiasing_checkbox.setToolTip("Smoother lines and text (may reduce performance)")
        performance_layout.addRow("", self.antialiasing_checkbox)
        
        # Rasterization threshold
        self.rasterize_threshold_spin = QSpinBox()
        self.rasterize_threshold_spin.setRange(0, 100000)
        self.rasterize_threshold_spin.setValue(2000)
        self.rasterize_threshold_spin.setSpecialValueText("Disabled")
        self.rasterize_threshold_spin.setToolTip("Rasterize plots with more than this many points for better performance")
        performance_layout.addRow("Rasterize Threshold:", self.rasterize_threshold_spin)
        
        # DPI setting
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 600)
        self.dpi_spin.setValue(100)
        self.dpi_spin.setToolTip("Dots per inch - higher values give better quality but larger file sizes")
        performance_layout.addRow("DPI:", self.dpi_spin)
        
        layout.addWidget(performance_group)
        
        layout.addStretch()
        
        return tab
    
    def _on_tick_spacing_changed(self, spacing_type):
        """Handle tick spacing change"""
        self.major_tick_value_spin.setEnabled(spacing_type == 'custom')
        
    def _on_axis_position_changed(self):
        """Handle axis position changes and automatically update tick display"""
        # Automatically enable/disable tick labels based on axis position
        if self.top_x_axis_radio.isChecked():
            self.show_top_ticks_checkbox.setChecked(True)
            self.show_bottom_ticks_checkbox.setChecked(False)  # Disable opposite side
            
        if self.bottom_x_axis_radio.isChecked():
            self.show_bottom_ticks_checkbox.setChecked(True)
            self.show_top_ticks_checkbox.setChecked(False)  # Disable opposite side
            
        if self.left_y_axis_radio.isChecked():
            self.show_left_ticks_checkbox.setChecked(True)
            self.show_right_ticks_checkbox.setChecked(False)  # Disable opposite side
            
        if self.right_y_axis_radio.isChecked():
            self.show_right_ticks_checkbox.setChecked(True)
            self.show_left_ticks_checkbox.setChecked(False)  # Disable opposite side
            
        print(f"[SubplotWizard] Axis position changed - auto-updated tick display")
        
    def _on_layout_option_changed(self):
        """Handle layout option changes to prevent conflicts between tight and constrained layout"""
        # If constrained layout is enabled, disable tight layout (they conflict)
        if self.constrained_layout_checkbox.isChecked():
            self.tight_layout_checkbox.setChecked(False)
            print(f"[SubplotWizard] Constrained layout enabled - disabled tight layout to avoid conflicts")
        
        # If tight layout is enabled, disable constrained layout
        elif self.tight_layout_checkbox.isChecked():
            self.constrained_layout_checkbox.setChecked(False)
            print(f"[SubplotWizard] Tight layout enabled - disabled constrained layout to avoid conflicts")
    
    def load_current_config(self):
        """Load current subplot configuration into the UI"""
        # Basic settings
        self.xlabel_edit.setText(self.subplot_config.get('xlabel', ''))
        self.ylabel_edit.setText(self.subplot_config.get('ylabel', ''))
        self.y_right_label_edit.setText(self.subplot_config.get('y_right_label', ''))
        self.title_edit.setText(self.subplot_config.get('title', ''))
        
        # Axis position settings (radio buttons - mutually exclusive)
        show_bottom_x = self.subplot_config.get('show_bottom_x_axis', True)
        show_top_x = self.subplot_config.get('show_top_x_axis', False)
        show_left_y = self.subplot_config.get('show_left_y_axis', True)
        show_right_y = self.subplot_config.get('show_right_y_axis', False)
        
        # Set X-axis radio button (prefer top if both are somehow true)
        if show_top_x:
            self.top_x_axis_radio.setChecked(True)
        else:
            self.bottom_x_axis_radio.setChecked(True)
        
        # Set Y-axis radio button (prefer right if both are somehow true)
        if show_right_y:
            self.right_y_axis_radio.setChecked(True)
        else:
            self.left_y_axis_radio.setChecked(True)
        
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
        
        # Tick settings - individual tick label display
        self.show_left_ticks_checkbox.setChecked(self.subplot_config.get('show_left_tick_labels', True))
        self.show_right_ticks_checkbox.setChecked(self.subplot_config.get('show_right_tick_labels', False))
        self.show_top_ticks_checkbox.setChecked(self.subplot_config.get('show_top_tick_labels', False))
        self.show_bottom_ticks_checkbox.setChecked(self.subplot_config.get('show_bottom_tick_labels', True))
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
        
        # Text & Font settings
        self.label_fontsize_spin.setValue(self.subplot_config.get('label_fontsize', 12))
        self.title_fontsize_spin.setValue(self.subplot_config.get('title_fontsize', 14))
        self.font_family_combo.setCurrentText(self.subplot_config.get('font_family', 'DejaVu Sans'))
        self.font_weight_combo.setCurrentText(self.subplot_config.get('font_weight', 'normal'))
        
        # Layout & Spacing settings
        self.aspect_ratio_combo.setCurrentText(self.subplot_config.get('aspect_ratio', 'auto'))
        self.tight_layout_checkbox.setChecked(self.subplot_config.get('tight_layout', True))
        self.constrained_layout_checkbox.setChecked(self.subplot_config.get('constrained_layout', False))
        self.subplot_margin_spin.setValue(self.subplot_config.get('subplot_margin', 0.1))
        
        # Performance & Quality settings
        self.antialiasing_checkbox.setChecked(self.subplot_config.get('antialiasing', True))
        self.rasterize_threshold_spin.setValue(self.subplot_config.get('rasterize_threshold', 2000))
        self.dpi_spin.setValue(self.subplot_config.get('dpi', 100))
    
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
        
        # Axis position settings (radio buttons - mutually exclusive)
        self.subplot_config['show_bottom_x_axis'] = self.bottom_x_axis_radio.isChecked()
        self.subplot_config['show_top_x_axis'] = self.top_x_axis_radio.isChecked()
        self.subplot_config['show_left_y_axis'] = self.left_y_axis_radio.isChecked()
        self.subplot_config['show_right_y_axis'] = self.right_y_axis_radio.isChecked()
        
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
        
        # Tick settings - individual tick label display
        self.subplot_config['show_left_tick_labels'] = self.show_left_ticks_checkbox.isChecked()
        self.subplot_config['show_right_tick_labels'] = self.show_right_ticks_checkbox.isChecked()
        self.subplot_config['show_top_tick_labels'] = self.show_top_ticks_checkbox.isChecked()
        self.subplot_config['show_bottom_tick_labels'] = self.show_bottom_ticks_checkbox.isChecked()
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
        
        # Text & Font settings
        self.subplot_config['label_fontsize'] = self.label_fontsize_spin.value()
        self.subplot_config['title_fontsize'] = self.title_fontsize_spin.value()
        self.subplot_config['font_family'] = self.font_family_combo.currentText()
        self.subplot_config['font_weight'] = self.font_weight_combo.currentText()
        
        # Layout & Spacing settings
        self.subplot_config['aspect_ratio'] = self.aspect_ratio_combo.currentText()
        self.subplot_config['tight_layout'] = self.tight_layout_checkbox.isChecked()
        self.subplot_config['constrained_layout'] = self.constrained_layout_checkbox.isChecked()
        self.subplot_config['subplot_margin'] = self.subplot_margin_spin.value()
        
        # Performance & Quality settings
        self.subplot_config['antialiasing'] = self.antialiasing_checkbox.isChecked()
        self.subplot_config['rasterize_threshold'] = self.rasterize_threshold_spin.value()
        self.subplot_config['dpi'] = self.dpi_spin.value()
    
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