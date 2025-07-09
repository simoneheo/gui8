from PySide6.QtWidgets import (
    QWidget, QFormLayout, QHBoxLayout, QVBoxLayout, QComboBox, 
    QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QSlider, QTabWidget,
    QLabel, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Signal
from typing import Optional, Dict, Any

from base_config_wizard import BaseConfigWizard, ColorButton, create_colormap_combo
from plot_manager import PLOT_STYLES


class MarkerWizard(BaseConfigWizard):
    """
    Dialog for editing marker properties including scatter plots and density plots
    Inherits from BaseConfigWizard for consistent UI and shared functionality
    """
    
    # Specific signal for marker wizard
    marker_updated = Signal(dict)  # Updated marker config when changes are applied
    
    def __init__(self, marker_config: Dict[str, Any], parent=None):
        # Initialize base class with marker config dict and wizard type
        super().__init__(marker_config, "Marker", parent)
        
        # Marker-specific UI components
        self.marker_style_combo = None
        self.marker_size_spin = None
        self.marker_color_button = None
        self.marker_color_combo = None
        self.edge_color_button = None
        self.edge_color_combo = None
        self.edge_width_spin = None
        
        # Density plot UI components
        self.density_type_combo = None
        self.colorscheme_combo = None
        self.reverse_colormap_checkbox = None
        self.hexbin_gridsize_spin = None
        self.kde_bandwidth_spin = None
        self.kde_levels_spin = None
        
        # Connect specific signal to base signal
        self.config_updated.connect(lambda obj: self.marker_updated.emit(obj))
    
    def _setup_window(self):
        """Override to set marker wizard specific window size"""
        super()._setup_window()
        self.setFixedSize(550, 750)  # Larger for marker wizard with density plots
    
    def _backup_properties(self) -> dict:
        """Backup original marker properties for cancel operation"""
        return self.config_object.copy()  # Since it's a dict, make a copy
    
    def _get_object_name(self) -> str:
        """Get display name for the marker config"""
        return self.config_object.get('name', 'Unnamed Marker')
    
    def _get_object_info(self) -> str:
        """Get info text for the marker config"""
        pair_name = self.config_object.get('pair_name', 'Unknown')
        return f"Pair: {pair_name}"
    
    def _create_main_tabs(self, tab_widget: QTabWidget):
        """Create marker-specific tabs"""
        # Scatter Markers tab (traditional marker settings)
        scatter_tab = self._create_scatter_tab()
        tab_widget.addTab(scatter_tab, "Scatter Markers")
        
        # Density Plots tab (hexbin and KDE)
        density_tab = self._create_density_tab()
        tab_widget.addTab(density_tab, "Density Plots")
        
        # Colorbar tab (for density plots)
        colorbar_tab = self._create_colorbar_tab()
        tab_widget.addTab(colorbar_tab, "Colorbar")
    
    def _create_scatter_tab(self) -> QWidget:
        """Create the scatter markers settings tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Marker Style
        self.marker_style_combo = QComboBox()
        for name, value in PLOT_STYLES['Markers'].items():
            self.marker_style_combo.addItem(name, value)
        self.marker_style_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Marker Style:", self.marker_style_combo)
        
        # Marker Size
        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 200)
        self.marker_size_spin.setValue(20)
        self.marker_size_spin.valueChanged.connect(self.update_preview)
        layout.addRow("Marker Size:", self.marker_size_spin)
        
        # Marker Color
        marker_color_layout = QHBoxLayout()
        self.marker_color_button = ColorButton()
        self.marker_color_combo = QComboBox()
        self.marker_color_combo.addItems(PLOT_STYLES['Colors'].keys())
        self.marker_color_combo.currentTextChanged.connect(self.on_marker_color_combo_changed)
        self.marker_color_button.color_changed.connect(self.on_marker_color_button_changed)
        
        marker_color_layout.addWidget(self.marker_color_button)
        marker_color_layout.addWidget(self.marker_color_combo)
        layout.addRow("Marker Color:", marker_color_layout)
        
        # Edge Color
        edge_color_layout = QHBoxLayout()
        self.edge_color_button = ColorButton("#000000")
        self.edge_color_combo = QComboBox()
        self.edge_color_combo.addItems(PLOT_STYLES['Colors'].keys())
        self.edge_color_combo.currentTextChanged.connect(self.on_edge_color_combo_changed)
        self.edge_color_button.color_changed.connect(self.on_edge_color_button_changed)
        
        edge_color_layout.addWidget(self.edge_color_button)
        edge_color_layout.addWidget(self.edge_color_combo)
        layout.addRow("Edge Color:", edge_color_layout)
        
        # Edge Width
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setRange(0.0, 10.0)
        self.edge_width_spin.setSingleStep(0.1)
        self.edge_width_spin.setValue(1.0)
        self.edge_width_spin.setDecimals(1)
        self.edge_width_spin.valueChanged.connect(self.update_preview)
        layout.addRow("Edge Width:", self.edge_width_spin)
        
        return tab
    
    def _create_density_tab(self) -> QWidget:
        """Create the density plots settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Density plot type
        type_group = QGroupBox("Density Plot Type")
        type_layout = QFormLayout(type_group)
        
        self.density_type_combo = QComboBox()
        density_types = ['None', 'Hexbin', 'KDE']
        self.density_type_combo.addItems(density_types)
        self.density_type_combo.currentTextChanged.connect(self._on_density_type_changed)
        type_layout.addRow("Type:", self.density_type_combo)
        
        layout.addWidget(type_group)
        
        # Colorscheme group
        colorscheme_group = QGroupBox("Colorscheme")
        colorscheme_layout = QFormLayout(colorscheme_group)
        
        # Colorscheme selection
        self.colorscheme_combo = create_colormap_combo('viridis')
        self.colorscheme_combo.currentTextChanged.connect(self.update_preview)
        colorscheme_layout.addRow("Colorscheme:", self.colorscheme_combo)
        
        # Reverse colormap
        self.reverse_colormap_checkbox = QCheckBox("Reverse Colormap")
        self.reverse_colormap_checkbox.toggled.connect(self.update_preview)
        colorscheme_layout.addRow("", self.reverse_colormap_checkbox)
        
        layout.addWidget(colorscheme_group)
        
        # Hexbin settings
        self.hexbin_group = QGroupBox("Hexbin Settings")
        hexbin_layout = QFormLayout(self.hexbin_group)
        
        self.hexbin_gridsize_spin = QSpinBox()
        self.hexbin_gridsize_spin.setRange(10, 200)
        self.hexbin_gridsize_spin.setValue(50)
        self.hexbin_gridsize_spin.valueChanged.connect(self.update_preview)
        hexbin_layout.addRow("Grid Size:", self.hexbin_gridsize_spin)
        
        layout.addWidget(self.hexbin_group)
        
        # KDE settings
        self.kde_group = QGroupBox("KDE Settings")
        kde_layout = QFormLayout(self.kde_group)
        
        self.kde_bandwidth_spin = QDoubleSpinBox()
        self.kde_bandwidth_spin.setRange(0.1, 10.0)
        self.kde_bandwidth_spin.setSingleStep(0.1)
        self.kde_bandwidth_spin.setValue(1.0)
        self.kde_bandwidth_spin.setDecimals(1)
        self.kde_bandwidth_spin.valueChanged.connect(self.update_preview)
        kde_layout.addRow("Bandwidth:", self.kde_bandwidth_spin)
        
        self.kde_levels_spin = QSpinBox()
        self.kde_levels_spin.setRange(5, 50)
        self.kde_levels_spin.setValue(10)
        self.kde_levels_spin.valueChanged.connect(self.update_preview)
        kde_layout.addRow("Contour Levels:", self.kde_levels_spin)
        
        layout.addWidget(self.kde_group)
        
        # Initially hide density-specific groups
        self.hexbin_group.setVisible(False)
        self.kde_group.setVisible(False)
        
        return tab
    
    def _create_colorbar_tab(self) -> QWidget:
        """Create the colorbar settings tab using base class functionality"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Use base class colorbar controls
        colorbar_group = self._create_colorbar_controls_group()
        layout.addWidget(colorbar_group)
        
        # Connect colorbar signals
        self._connect_colorbar_signals()
        
        return tab
    
    def _on_density_type_changed(self, density_type: str):
        """Handle density type changes"""
        self.hexbin_group.setVisible(density_type == 'Hexbin')
        self.kde_group.setVisible(density_type == 'KDE')
        self.update_preview()
    
    def on_marker_color_combo_changed(self, color_name: str):
        """Handle marker color combo box changes"""
        if color_name in PLOT_STYLES['Colors']:
            color_value = PLOT_STYLES['Colors'][color_name]
            self.marker_color_button.set_color(color_value)
        self.update_preview()
    
    def on_marker_color_button_changed(self, color: str):
        """Handle marker color button changes"""
        self.marker_color_combo.setCurrentText("Custom")
        self.update_preview()
    
    def on_edge_color_combo_changed(self, color_name: str):
        """Handle edge color combo box changes"""
        if color_name in PLOT_STYLES['Colors']:
            color_value = PLOT_STYLES['Colors'][color_name]
            self.edge_color_button.set_color(color_value)
        self.update_preview()
    
    def on_edge_color_button_changed(self, color: str):
        """Handle edge color button changes"""
        self.edge_color_combo.setCurrentText("Custom")
        self.update_preview()
    
    def load_properties(self):
        """Load current marker properties into the UI"""
        # Load common properties first
        self._load_common_properties()
        
        # Load colorbar properties
        self._load_colorbar_properties()
        
        # Load marker-specific properties
        # Marker style
        marker_style = self.config_object.get('marker_style', 'o')
        for i in range(self.marker_style_combo.count()):
            if self.marker_style_combo.itemData(i) == marker_style:
                self.marker_style_combo.setCurrentIndex(i)
                break
        
        # Marker size
        marker_size = self.config_object.get('marker_size', 20)
        self.marker_size_spin.setValue(marker_size)
        
        # Marker color
        marker_color = self.config_object.get('marker_color', '#1f77b4')
        self.marker_color_button.set_color(marker_color)
        
        # Edge color
        edge_color = self.config_object.get('edge_color', '#000000')
        self.edge_color_button.set_color(edge_color)
        
        # Edge width
        edge_width = self.config_object.get('edge_width', 1.0)
        self.edge_width_spin.setValue(edge_width)
        
        # Density plot settings
        density_type = self.config_object.get('density_type', 'None')
        index = self.density_type_combo.findText(density_type)
        if index >= 0:
            self.density_type_combo.setCurrentIndex(index)
        
        # Colorscheme
        colorscheme = self.config_object.get('colorscheme', 'viridis')
        index = self.colorscheme_combo.findText(colorscheme)
        if index >= 0:
            self.colorscheme_combo.setCurrentIndex(index)
        
        # Reverse colormap
        reverse_colormap = self.config_object.get('reverse_colormap', False)
        self.reverse_colormap_checkbox.setChecked(reverse_colormap)
        
        # Hexbin settings
        hexbin_gridsize = self.config_object.get('hexbin_gridsize', 50)
        self.hexbin_gridsize_spin.setValue(hexbin_gridsize)
        
        # KDE settings
        kde_bandwidth = self.config_object.get('kde_bandwidth', 1.0)
        self.kde_bandwidth_spin.setValue(kde_bandwidth)
        
        kde_levels = self.config_object.get('kde_levels', 10)
        self.kde_levels_spin.setValue(kde_levels)
        
        # Trigger density type change to show/hide appropriate groups
        self._on_density_type_changed(density_type)
        
        # Initial preview update
        self.update_preview()
    
    def update_preview(self):
        """Update the preview display with marker-specific information"""
        if not self.preview_label:
            return
        
        # Get current settings
        legend_name = self.legend_edit.text() if self.legend_edit else "Unnamed"
        marker_style = self.marker_style_combo.currentText() if self.marker_style_combo else "Circle"
        marker_size = self.marker_size_spin.value() if self.marker_size_spin else 20
        marker_color = self.marker_color_button.get_color() if self.marker_color_button else "#1f77b4"
        edge_color = self.edge_color_button.get_color() if self.edge_color_button else "#000000"
        edge_width = self.edge_width_spin.value() if self.edge_width_spin else 1.0
        alpha = self.alpha_spin.value() if self.alpha_spin else 1.0
        x_axis = "Bottom" if self.bottom_x_axis_radio and self.bottom_x_axis_radio.isChecked() else "Top"
        bring_to_front = "Yes" if self.bring_to_front_checkbox and self.bring_to_front_checkbox.isChecked() else "No"
        
        # Density plot info
        density_type = self.density_type_combo.currentText() if self.density_type_combo else "None"
        colorscheme = self.colorscheme_combo.currentText() if self.colorscheme_combo else "viridis"
        reverse_colormap = "Yes" if self.reverse_colormap_checkbox and self.reverse_colormap_checkbox.isChecked() else "No"
        
        # Colorbar info
        show_colorbar = "Yes" if hasattr(self, 'show_colorbar_checkbox') and self.show_colorbar_checkbox.isChecked() else "No"
        
        # Create preview text
        preview_text = f"""
        <div style="padding: 5px; font-family: monospace; font-size: 10px;">
            <b>Marker Configuration Preview</b><br>
            <b>Legend:</b> {legend_name}<br>
            <b>Marker:</b> {marker_style} (size {marker_size})<br>
            <b>Color:</b> <span style="color: {marker_color}; font-size: 14px;">●</span> {marker_color}<br>
            <b>Edge:</b> <span style="color: {edge_color}; font-size: 14px;">●</span> {edge_color} (width {edge_width})<br>
            <b>Transparency:</b> {alpha:.2f} ({int(alpha*100)}%)<br>
            <b>X-Axis:</b> {x_axis}<br>
            <b>Bring to Front:</b> {bring_to_front}<br>
            <b>Density Type:</b> {density_type}<br>
            <b>Colorscheme:</b> {colorscheme} (reversed: {reverse_colormap})<br>
            <b>Colorbar:</b> {show_colorbar}
        </div>
        """
        
        self.preview_label.setText(preview_text)
    
    def _update_properties(self):
        """Update marker properties from UI"""
        # Update common properties first
        self._update_common_properties()
        
        # Update colorbar properties
        self._update_colorbar_properties()
        
        # Update marker-specific properties
        self.config_object['marker_style'] = self.marker_style_combo.currentData()
        self.config_object['marker_size'] = self.marker_size_spin.value()
        self.config_object['marker_color'] = self.marker_color_button.get_color()
        self.config_object['edge_color'] = self.edge_color_button.get_color()
        self.config_object['edge_width'] = self.edge_width_spin.value()
        
        # Density plot properties
        self.config_object['density_type'] = self.density_type_combo.currentText()
        self.config_object['colorscheme'] = self.colorscheme_combo.currentText()
        self.config_object['reverse_colormap'] = self.reverse_colormap_checkbox.isChecked()
        self.config_object['hexbin_gridsize'] = self.hexbin_gridsize_spin.value()
        self.config_object['kde_bandwidth'] = self.kde_bandwidth_spin.value()
        self.config_object['kde_levels'] = self.kde_levels_spin.value()
    
    def _restore_properties(self):
        """Restore original marker properties"""
        self.config_object.clear()
        self.config_object.update(self.original_properties)
    
    @staticmethod
    def edit_marker(marker_config: Dict[str, Any], parent=None) -> bool:
        """
        Static method to edit marker properties
        
        Args:
            marker_config: Marker configuration dict to edit
            parent: Parent widget
            
        Returns:
            True if changes were applied, False if cancelled
        """
        dialog = MarkerWizard(marker_config, parent)
        result = dialog.exec()
        return result == QDialog.Accepted


# Convenience function for opening the marker wizard
def open_marker_wizard(marker_config: Dict[str, Any], parent=None) -> bool:
    """
    Open the marker wizard for editing marker properties
    
    Args:
        marker_config: Marker configuration dict to edit
        parent: Parent widget
        
    Returns:
        True if changes were applied, False if cancelled
    """
    return MarkerWizard.edit_marker(marker_config, parent) 