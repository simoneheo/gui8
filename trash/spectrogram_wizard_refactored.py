from PySide6.QtWidgets import (
    QWidget, QFormLayout, QHBoxLayout, QVBoxLayout, QComboBox, 
    QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QSlider, QTabWidget
)
from PySide6.QtCore import Signal
from typing import Optional

from base_config_wizard import BaseConfigWizard, create_colormap_combo
from channel import Channel


class SpectrogramWizard(BaseConfigWizard):
    """
    Dialog for editing spectrogram properties
    Inherits from BaseConfigWizard for consistent UI and shared functionality
    """
    
    # Specific signal for spectrogram wizard
    spectrogram_updated = Signal(str)  # channel_id when changes are applied
    
    def __init__(self, channel: Channel, parent=None):
        # Initialize base class with channel object and wizard type
        super().__init__(channel, "Spectrogram", parent)
        
        # Spectrogram-specific UI components
        self.colormap_combo = None
        self.interpolation_combo = None
        self.freq_scale_combo = None
        self.time_scale_combo = None
        self.freq_min_spin = None
        self.freq_max_spin = None
        self.time_min_spin = None
        self.time_max_spin = None
        self.clim_auto_checkbox = None
        self.clim_min_spin = None
        self.clim_max_spin = None
        self.aspect_combo = None
        self.shading_combo = None
        
        # Connect specific signal to base signal
        self.config_updated.connect(lambda obj: self.spectrogram_updated.emit(obj.channel_id))
    
    def _setup_window(self):
        """Override to set spectrogram wizard specific window size"""
        super()._setup_window()
        self.setFixedSize(520, 700)  # Larger for spectrogram wizard
    
    def _backup_properties(self) -> dict:
        """Backup original channel properties for cancel operation"""
        return {
            'colormap': getattr(self.config_object, 'colormap', 'viridis'),
            'clim_min': getattr(self.config_object, 'clim_min', None),
            'clim_max': getattr(self.config_object, 'clim_max', None),
            'clim_auto': getattr(self.config_object, 'clim_auto', True),
            'colorbar_position': getattr(self.config_object, 'colorbar_position', 'right'),
            'colorbar_label': getattr(self.config_object, 'colorbar_label', 'Power (dB)'),
            'show_colorbar': getattr(self.config_object, 'show_colorbar', True),
            'colorbar_pad': getattr(self.config_object, 'colorbar_pad', 0.05),
            'colorbar_shrink': getattr(self.config_object, 'colorbar_shrink', 0.8),
            'colorbar_aspect': getattr(self.config_object, 'colorbar_aspect', 20),
            'colorbar_ticks': getattr(self.config_object, 'colorbar_ticks', 5),
            'tick_format': getattr(self.config_object, 'tick_format', '%.1f'),
            'colorbar_label_fontsize': getattr(self.config_object, 'colorbar_label_fontsize', 10),
            'colorbar_tick_fontsize': getattr(self.config_object, 'colorbar_tick_fontsize', 8),
            'freq_scale': getattr(self.config_object, 'freq_scale', 'linear'),
            'time_scale': getattr(self.config_object, 'time_scale', 'linear'),
            'freq_limits': getattr(self.config_object, 'freq_limits', [None, None]),
            'time_limits': getattr(self.config_object, 'time_limits', [None, None]),
            'xaxis': getattr(self.config_object, 'xaxis', 'x-bottom'),
            'interpolation': getattr(self.config_object, 'interpolation', 'nearest'),
            'alpha': getattr(self.config_object, 'alpha', 1.0),
            'legend_label': getattr(self.config_object, 'legend_label', None),
            'aspect': getattr(self.config_object, 'aspect', 'auto'),
            'shading': getattr(self.config_object, 'shading', 'auto')
        }
    
    def _get_object_name(self) -> str:
        """Get display name for the channel"""
        return getattr(self.config_object, 'legend_label', None) or getattr(self.config_object, 'ylabel', 'Unnamed')
    
    def _get_object_info(self) -> str:
        """Get info text for the channel"""
        return f"Channel ID: {getattr(self.config_object, 'channel_id', 'Unknown')}"
    
    def _create_main_tabs(self, tab_widget: QTabWidget):
        """Create spectrogram-specific tabs"""
        # Appearance tab (spectrogram-specific)
        appearance_tab = self._create_appearance_tab()
        tab_widget.addTab(appearance_tab, "Appearance")
        
        # Colorbar tab (shared with marker wizard for density plots)
        colorbar_tab = self._create_colorbar_tab()
        tab_widget.addTab(colorbar_tab, "Colorbar")
        
        # Scaling tab (spectrogram-specific)
        scaling_tab = self._create_scaling_tab()
        tab_widget.addTab(scaling_tab, "Scaling")
        
        # Advanced tab (spectrogram-specific)
        advanced_tab = self._create_advanced_tab()
        tab_widget.addTab(advanced_tab, "Advanced")
    
    def _create_appearance_tab(self) -> QWidget:
        """Create the appearance settings tab specific to spectrograms"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Color Theme (Colormap)
        self.colormap_combo = create_colormap_combo('viridis')
        self.colormap_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Color Theme:", self.colormap_combo)
        
        # Interpolation
        self.interpolation_combo = QComboBox()
        interpolations = ['nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 
                         'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 
                         'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
        self.interpolation_combo.addItems(interpolations)
        self.interpolation_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Interpolation:", self.interpolation_combo)
        
        return tab
    
    def _create_colorbar_tab(self) -> QWidget:
        """Create the colorbar settings tab using base class functionality"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Use base class colorbar controls
        colorbar_group = self._create_colorbar_controls_group()
        layout.addWidget(colorbar_group)
        
        # Color range controls (spectrogram-specific)
        range_group = QGroupBox("Color Range")
        range_layout = QFormLayout(range_group)
        
        # Auto range checkbox
        self.clim_auto_checkbox = QCheckBox("Auto Range")
        self.clim_auto_checkbox.setChecked(True)
        self.clim_auto_checkbox.toggled.connect(self._on_auto_range_toggled)
        range_layout.addRow("", self.clim_auto_checkbox)
        
        # Manual range controls
        self.clim_min_spin = QDoubleSpinBox()
        self.clim_min_spin.setRange(-1000, 1000)
        self.clim_min_spin.setValue(0)
        self.clim_min_spin.setEnabled(False)
        self.clim_min_spin.valueChanged.connect(self.update_preview)
        range_layout.addRow("Min Value:", self.clim_min_spin)
        
        self.clim_max_spin = QDoubleSpinBox()
        self.clim_max_spin.setRange(-1000, 1000)
        self.clim_max_spin.setValue(100)
        self.clim_max_spin.setEnabled(False)
        self.clim_max_spin.valueChanged.connect(self.update_preview)
        range_layout.addRow("Max Value:", self.clim_max_spin)
        
        layout.addWidget(range_group)
        
        # Connect colorbar signals
        self._connect_colorbar_signals()
        
        return tab
    
    def _create_scaling_tab(self) -> QWidget:
        """Create the scaling settings tab specific to spectrograms"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Frequency Scale
        self.freq_scale_combo = QComboBox()
        scales = ['linear', 'log', 'symlog', 'logit']
        self.freq_scale_combo.addItems(scales)
        self.freq_scale_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Frequency Scale:", self.freq_scale_combo)
        
        # Time Scale
        self.time_scale_combo = QComboBox()
        self.time_scale_combo.addItems(scales)
        self.time_scale_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Time Scale:", self.time_scale_combo)
        
        # Frequency Limits
        freq_limits_layout = QHBoxLayout()
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0, 10000)
        self.freq_min_spin.setValue(0)
        self.freq_min_spin.setSpecialValueText("Auto")
        self.freq_min_spin.valueChanged.connect(self.update_preview)
        
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(0, 10000)
        self.freq_max_spin.setValue(1000)
        self.freq_max_spin.setSpecialValueText("Auto")
        self.freq_max_spin.valueChanged.connect(self.update_preview)
        
        freq_limits_layout.addWidget(self.freq_min_spin)
        freq_limits_layout.addWidget(QLabel("to"))
        freq_limits_layout.addWidget(self.freq_max_spin)
        layout.addRow("Frequency Limits:", freq_limits_layout)
        
        # Time Limits
        time_limits_layout = QHBoxLayout()
        self.time_min_spin = QDoubleSpinBox()
        self.time_min_spin.setRange(0, 10000)
        self.time_min_spin.setValue(0)
        self.time_min_spin.setSpecialValueText("Auto")
        self.time_min_spin.valueChanged.connect(self.update_preview)
        
        self.time_max_spin = QDoubleSpinBox()
        self.time_max_spin.setRange(0, 10000)
        self.time_max_spin.setValue(100)
        self.time_max_spin.setSpecialValueText("Auto")
        self.time_max_spin.valueChanged.connect(self.update_preview)
        
        time_limits_layout.addWidget(self.time_min_spin)
        time_limits_layout.addWidget(QLabel("to"))
        time_limits_layout.addWidget(self.time_max_spin)
        layout.addRow("Time Limits:", time_limits_layout)
        
        return tab
    
    def _create_advanced_tab(self) -> QWidget:
        """Create the advanced settings tab specific to spectrograms"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Aspect Ratio
        self.aspect_combo = QComboBox()
        aspects = ['auto', 'equal', '1', '2', '0.5']
        self.aspect_combo.addItems(aspects)
        self.aspect_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Aspect Ratio:", self.aspect_combo)
        
        # Shading
        self.shading_combo = QComboBox()
        shadings = ['auto', 'flat', 'gouraud']
        self.shading_combo.addItems(shadings)
        self.shading_combo.currentTextChanged.connect(self.update_preview)
        layout.addRow("Shading:", self.shading_combo)
        
        return tab
    
    def _on_auto_range_toggled(self, checked):
        """Handle auto range checkbox toggle"""
        self.clim_min_spin.setEnabled(not checked)
        self.clim_max_spin.setEnabled(not checked)
        self.update_preview()
    
    def load_properties(self):
        """Load current channel properties into the UI"""
        # Load common properties first
        self._load_common_properties()
        
        # Load colorbar properties
        self._load_colorbar_properties()
        
        # Load spectrogram-specific properties
        # Colormap
        colormap = getattr(self.config_object, 'colormap', 'viridis')
        index = self.colormap_combo.findText(colormap)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)
        
        # Interpolation
        interpolation = getattr(self.config_object, 'interpolation', 'nearest')
        index = self.interpolation_combo.findText(interpolation)
        if index >= 0:
            self.interpolation_combo.setCurrentIndex(index)
        
        # Color range
        clim_auto = getattr(self.config_object, 'clim_auto', True)
        self.clim_auto_checkbox.setChecked(clim_auto)
        
        clim_min = getattr(self.config_object, 'clim_min', None)
        if clim_min is not None:
            self.clim_min_spin.setValue(clim_min)
        
        clim_max = getattr(self.config_object, 'clim_max', None)
        if clim_max is not None:
            self.clim_max_spin.setValue(clim_max)
        
        # Scaling
        freq_scale = getattr(self.config_object, 'freq_scale', 'linear')
        index = self.freq_scale_combo.findText(freq_scale)
        if index >= 0:
            self.freq_scale_combo.setCurrentIndex(index)
        
        time_scale = getattr(self.config_object, 'time_scale', 'linear')
        index = self.time_scale_combo.findText(time_scale)
        if index >= 0:
            self.time_scale_combo.setCurrentIndex(index)
        
        # Limits
        freq_limits = getattr(self.config_object, 'freq_limits', [None, None])
        if freq_limits[0] is not None:
            self.freq_min_spin.setValue(freq_limits[0])
        if freq_limits[1] is not None:
            self.freq_max_spin.setValue(freq_limits[1])
        
        time_limits = getattr(self.config_object, 'time_limits', [None, None])
        if time_limits[0] is not None:
            self.time_min_spin.setValue(time_limits[0])
        if time_limits[1] is not None:
            self.time_max_spin.setValue(time_limits[1])
        
        # Advanced
        aspect = getattr(self.config_object, 'aspect', 'auto')
        index = self.aspect_combo.findText(aspect)
        if index >= 0:
            self.aspect_combo.setCurrentIndex(index)
        
        shading = getattr(self.config_object, 'shading', 'auto')
        index = self.shading_combo.findText(shading)
        if index >= 0:
            self.shading_combo.setCurrentIndex(index)
        
        # Initial preview update
        self.update_preview()
    
    def update_preview(self):
        """Update the preview display with spectrogram-specific information"""
        if not self.preview_label:
            return
        
        # Get current settings
        legend_name = self.legend_edit.text() if self.legend_edit else "Unnamed"
        colormap = self.colormap_combo.currentText() if self.colormap_combo else "viridis"
        alpha = self.alpha_spin.value() if self.alpha_spin else 1.0
        interpolation = self.interpolation_combo.currentText() if self.interpolation_combo else "nearest"
        x_axis = "Bottom" if self.bottom_x_axis_radio and self.bottom_x_axis_radio.isChecked() else "Top"
        bring_to_front = "Yes" if self.bring_to_front_checkbox and self.bring_to_front_checkbox.isChecked() else "No"
        
        # Colorbar info
        show_colorbar = "Yes" if hasattr(self, 'show_colorbar_checkbox') and self.show_colorbar_checkbox.isChecked() else "No"
        colorbar_pos = self.colorbar_position_combo.currentText() if hasattr(self, 'colorbar_position_combo') else "right"
        
        # Color range info
        auto_range = "Yes" if self.clim_auto_checkbox and self.clim_auto_checkbox.isChecked() else "No"
        range_info = "Auto" if auto_range == "Yes" else f"{self.clim_min_spin.value():.1f} to {self.clim_max_spin.value():.1f}"
        
        # Create preview text
        preview_text = f"""
        <div style="padding: 5px; font-family: monospace; font-size: 10px;">
            <b>Spectrogram Configuration Preview</b><br>
            <b>Legend:</b> {legend_name}<br>
            <b>Colormap:</b> {colormap}<br>
            <b>Interpolation:</b> {interpolation}<br>
            <b>Transparency:</b> {alpha:.2f} ({int(alpha*100)}%)<br>
            <b>X-Axis:</b> {x_axis}<br>
            <b>Bring to Front:</b> {bring_to_front}<br>
            <b>Colorbar:</b> {show_colorbar} ({colorbar_pos})<br>
            <b>Color Range:</b> {range_info}<br>
            <b>Freq Scale:</b> {self.freq_scale_combo.currentText() if self.freq_scale_combo else 'linear'}<br>
            <b>Time Scale:</b> {self.time_scale_combo.currentText() if self.time_scale_combo else 'linear'}
        </div>
        """
        
        self.preview_label.setText(preview_text)
    
    def _update_properties(self):
        """Update channel properties from UI"""
        # Update common properties first
        self._update_common_properties()
        
        # Update colorbar properties
        self._update_colorbar_properties()
        
        # Update spectrogram-specific properties
        self.config_object.colormap = self.colormap_combo.currentText()
        self.config_object.interpolation = self.interpolation_combo.currentText()
        
        # Color range
        self.config_object.clim_auto = self.clim_auto_checkbox.isChecked()
        if not self.clim_auto_checkbox.isChecked():
            self.config_object.clim_min = self.clim_min_spin.value()
            self.config_object.clim_max = self.clim_max_spin.value()
        else:
            self.config_object.clim_min = None
            self.config_object.clim_max = None
        
        # Scaling
        self.config_object.freq_scale = self.freq_scale_combo.currentText()
        self.config_object.time_scale = self.time_scale_combo.currentText()
        
        # Limits
        freq_min = self.freq_min_spin.value() if self.freq_min_spin.value() > 0 else None
        freq_max = self.freq_max_spin.value() if self.freq_max_spin.value() > 0 else None
        self.config_object.freq_limits = [freq_min, freq_max]
        
        time_min = self.time_min_spin.value() if self.time_min_spin.value() > 0 else None
        time_max = self.time_max_spin.value() if self.time_max_spin.value() > 0 else None
        self.config_object.time_limits = [time_min, time_max]
        
        # Advanced
        self.config_object.aspect = self.aspect_combo.currentText()
        self.config_object.shading = self.shading_combo.currentText()
    
    def _restore_properties(self):
        """Restore original channel properties"""
        for key, value in self.original_properties.items():
            setattr(self.config_object, key, value)
    
    @staticmethod
    def edit_spectrogram(channel: Channel, parent=None) -> bool:
        """
        Static method to edit a channel's spectrogram properties
        
        Args:
            channel: Channel object to edit
            parent: Parent widget
            
        Returns:
            True if changes were applied, False if cancelled
        """
        dialog = SpectrogramWizard(channel, parent)
        result = dialog.exec()
        return result == QDialog.Accepted


# Convenience function for opening the spectrogram wizard
def open_spectrogram_wizard(channel: Channel, parent=None) -> bool:
    """
    Open the spectrogram wizard for editing a channel's spectrogram properties
    
    Args:
        channel: Channel object to edit
        parent: Parent widget
        
    Returns:
        True if changes were applied, False if cancelled
    """
    return SpectrogramWizard.edit_spectrogram(channel, parent) 