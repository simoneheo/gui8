from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QLineEdit, QComboBox, QFrame, QGroupBox, 
    QDialogButtonBox, QCheckBox, QDoubleSpinBox, QSpinBox, 
    QSlider, QTabWidget, QWidget, QTextEdit, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont
from typing import Optional, Dict, Any
import numpy as np


class SpectrogramWizard(QDialog):
    """
    Dialog for editing spectrogram properties including:
    - Colorbar range and position
    - Color theme/colormap
    - Frequency and time axis settings
    - Display options
    """
    
    # Signals
    spectrogram_updated = Signal(str)  # channel_id when changes are applied
    
    def __init__(self, channel, parent=None):
        super().__init__(parent)
        self.channel = channel
        self.original_properties = self._backup_channel_properties()
        
        channel_name = getattr(channel, 'legend_label', None) or getattr(channel, 'ylabel', 'Unnamed')
        self.setWindowTitle(f"Spectrogram Properties - {channel_name}")
        self.setModal(True)
        self.setFixedSize(500, 650)
        
        self.init_ui()
        self.load_channel_properties()
    
    def _backup_channel_properties(self) -> dict:
        """Backup original channel properties for cancel operation"""
        return {
            'colormap': getattr(self.channel, 'colormap', 'viridis'),
            'clim_min': getattr(self.channel, 'clim_min', None),
            'clim_max': getattr(self.channel, 'clim_max', None),
            'clim_auto': getattr(self.channel, 'clim_auto', True),
            'colorbar_position': getattr(self.channel, 'colorbar_position', 'right'),
            'colorbar_label': getattr(self.channel, 'colorbar_label', 'Power (dB)'),
            'show_colorbar': getattr(self.channel, 'show_colorbar', True),
            'colorbar_pad': getattr(self.channel, 'colorbar_pad', 0.05),
            'colorbar_shrink': getattr(self.channel, 'colorbar_shrink', 0.8),
            'colorbar_aspect': getattr(self.channel, 'colorbar_aspect', 20),
            'colorbar_ticks': getattr(self.channel, 'colorbar_ticks', 5),
            'tick_format': getattr(self.channel, 'tick_format', '%.1f'),
            'colorbar_label_fontsize': getattr(self.channel, 'colorbar_label_fontsize', 10),
            'colorbar_tick_fontsize': getattr(self.channel, 'colorbar_tick_fontsize', 8),
            'freq_scale': getattr(self.channel, 'freq_scale', 'linear'),
            'time_scale': getattr(self.channel, 'time_scale', 'linear'),
            'freq_limits': getattr(self.channel, 'freq_limits', [None, None]),
            'time_limits': getattr(self.channel, 'time_limits', [None, None]),
            'xaxis': getattr(self.channel, 'xaxis', 'x-bottom'),
            'interpolation': getattr(self.channel, 'interpolation', 'nearest'),
            'alpha': getattr(self.channel, 'alpha', 1.0),
            'legend_label': getattr(self.channel, 'legend_label', None)
        }
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        channel_name = getattr(self.channel, 'legend_label', None) or getattr(self.channel, 'ylabel', 'Unnamed')
        title = QLabel(f"Editing Spectrogram: {channel_name}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Channel info
        info_text = f"Channel ID: {getattr(self.channel, 'channel_id', 'Unknown')}"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Tab widget for organizing options
        tab_widget = QTabWidget()
        
        # Basic tab
        basic_tab = self._create_basic_tab()
        tab_widget.addTab(basic_tab, "Basic")
        
        # Colorbar tab
        colorbar_tab = self._create_colorbar_tab()
        tab_widget.addTab(colorbar_tab, "Colorbar")
        
        # Axis tab
        axis_tab = self._create_axis_tab()
        tab_widget.addTab(axis_tab, "Axis")
        
        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tab_widget.addTab(advanced_tab, "Advanced")
        
        layout.addWidget(tab_widget)
        
        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("Spectrogram preview will appear here")
        self.preview_label.setMinimumHeight(80)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: white;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(button_box)
        
        # Connect signals for live preview
        self._connect_preview_signals()
    
    def _create_basic_tab(self) -> QWidget:
        """Create the basic settings tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Legend Name
        self.legend_edit = QLineEdit()
        self.legend_edit.setPlaceholderText("Enter legend label...")
        layout.addRow("Legend Name:", self.legend_edit)
        
        # Color Theme (Colormap)
        self.colormap_combo = QComboBox()
        colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
            'gray', 'bone', 'copper', 'pink',
            'Spectral', 'coolwarm', 'bwr', 'seismic',
            'RdYlBu', 'RdBu', 'RdGy', 'BrBG', 'PiYG', 'PRGn',
            'tab10', 'tab20', 'Set1', 'Set2', 'Set3'
        ]
        self.colormap_combo.addItems(colormaps)
        layout.addRow("Color Theme:", self.colormap_combo)
        
        # Transparency
        transparency_layout = QHBoxLayout()
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setDecimals(2)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(100)
        self.alpha_spin.valueChanged.connect(lambda v: self.alpha_slider.setValue(int(v * 100)))
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_spin.setValue(v / 100.0))
        
        transparency_layout.addWidget(self.alpha_spin)
        transparency_layout.addWidget(self.alpha_slider)
        layout.addRow("Transparency:", transparency_layout)
        
        # Interpolation
        self.interpolation_combo = QComboBox()
        interpolations = ['nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
        self.interpolation_combo.addItems(interpolations)
        layout.addRow("Interpolation:", self.interpolation_combo)
        
        return tab
    
    def _create_colorbar_tab(self) -> QWidget:
        """Create the colorbar settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Basic colorbar settings
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QFormLayout(basic_group)
        
        # Show Colorbar
        self.show_colorbar_checkbox = QCheckBox("Show Colorbar")
        self.show_colorbar_checkbox.setChecked(True)
        basic_layout.addRow("", self.show_colorbar_checkbox)
        
        # Colorbar Position
        self.colorbar_position_combo = QComboBox()
        positions = ['right', 'left', 'top', 'bottom']
        self.colorbar_position_combo.addItems(positions)
        basic_layout.addRow("Colorbar Position:", self.colorbar_position_combo)
        
        # Colorbar Label
        self.colorbar_label_edit = QLineEdit()
        self.colorbar_label_edit.setPlaceholderText("e.g., Power (dB)")
        basic_layout.addRow("Colorbar Label:", self.colorbar_label_edit)
        
        layout.addWidget(basic_group)
        
        # Precise positioning group
        position_group = QGroupBox("Precise Positioning")
        position_layout = QFormLayout(position_group)
        
        # Padding
        self.colorbar_pad_spin = QDoubleSpinBox()
        self.colorbar_pad_spin.setRange(0.0, 1.0)
        self.colorbar_pad_spin.setSingleStep(0.01)
        self.colorbar_pad_spin.setValue(0.05)
        self.colorbar_pad_spin.setDecimals(3)
        self.colorbar_pad_spin.setToolTip("Distance between colorbar and plot")
        position_layout.addRow("Padding:", self.colorbar_pad_spin)
        
        # Shrink
        self.colorbar_shrink_spin = QDoubleSpinBox()
        self.colorbar_shrink_spin.setRange(0.1, 1.0)
        self.colorbar_shrink_spin.setSingleStep(0.05)
        self.colorbar_shrink_spin.setValue(0.8)
        self.colorbar_shrink_spin.setDecimals(2)
        self.colorbar_shrink_spin.setToolTip("Fraction of original size")
        position_layout.addRow("Shrink:", self.colorbar_shrink_spin)
        
        # Aspect ratio
        self.colorbar_aspect_spin = QSpinBox()
        self.colorbar_aspect_spin.setRange(5, 50)
        self.colorbar_aspect_spin.setValue(20)
        self.colorbar_aspect_spin.setToolTip("Ratio of long to short dimensions")
        position_layout.addRow("Aspect Ratio:", self.colorbar_aspect_spin)
        
        layout.addWidget(position_group)
        
        # Color Range Settings
        range_group = QGroupBox("Color Range")
        range_layout = QFormLayout(range_group)
        
        # Auto Range
        self.auto_range_checkbox = QCheckBox("Auto Range")
        self.auto_range_checkbox.setChecked(True)
        self.auto_range_checkbox.toggled.connect(self._on_auto_range_toggled)
        range_layout.addRow("", self.auto_range_checkbox)
        
        # Manual Range
        self.clim_min_spin = QDoubleSpinBox()
        self.clim_min_spin.setRange(-200.0, 200.0)
        self.clim_min_spin.setValue(-60.0)
        self.clim_min_spin.setSuffix(" dB")
        self.clim_min_spin.setEnabled(False)
        range_layout.addRow("Min Value:", self.clim_min_spin)
        
        self.clim_max_spin = QDoubleSpinBox()
        self.clim_max_spin.setRange(-200.0, 200.0)
        self.clim_max_spin.setValue(0.0)
        self.clim_max_spin.setSuffix(" dB")
        self.clim_max_spin.setEnabled(False)
        range_layout.addRow("Max Value:", self.clim_max_spin)
        
        layout.addWidget(range_group)
        
        # Tick and label controls
        tick_group = QGroupBox("Ticks and Labels")
        tick_layout = QFormLayout(tick_group)
        
        # Number of ticks
        self.colorbar_ticks_spin = QSpinBox()
        self.colorbar_ticks_spin.setRange(2, 20)
        self.colorbar_ticks_spin.setValue(5)
        self.colorbar_ticks_spin.setToolTip("Number of major ticks on colorbar")
        tick_layout.addRow("Number of Ticks:", self.colorbar_ticks_spin)
        
        # Tick label format
        self.tick_format_combo = QComboBox()
        formats = ['%.1f', '%.2f', '%.0f', '%.1e', '%.2e', '{:.1f}', '{:.2f}', '{:.0f}']
        self.tick_format_combo.addItems(formats)
        self.tick_format_combo.setCurrentText('%.1f')
        self.tick_format_combo.setEditable(True)
        self.tick_format_combo.setToolTip("Format string for tick labels")
        tick_layout.addRow("Tick Format:", self.tick_format_combo)
        
        # Label font size
        self.colorbar_label_fontsize_spin = QSpinBox()
        self.colorbar_label_fontsize_spin.setRange(6, 24)
        self.colorbar_label_fontsize_spin.setValue(10)
        tick_layout.addRow("Label Font Size:", self.colorbar_label_fontsize_spin)
        
        # Tick font size
        self.colorbar_tick_fontsize_spin = QSpinBox()
        self.colorbar_tick_fontsize_spin.setRange(6, 20)
        self.colorbar_tick_fontsize_spin.setValue(8)
        tick_layout.addRow("Tick Font Size:", self.colorbar_tick_fontsize_spin)
        
        layout.addWidget(tick_group)
        
        return tab
    
    def _create_axis_tab(self) -> QWidget:
        """Create the axis settings tab"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Axis Position Group
        position_group = QGroupBox("Axis Position")
        position_layout = QFormLayout(position_group)
        
        # X-Axis Position
        x_axis_layout = QHBoxLayout()
        self.x_axis_button_group = QButtonGroup()
        self.bottom_x_axis_radio = QRadioButton("Bottom")
        self.top_x_axis_radio = QRadioButton("Top")
        
        self.x_axis_button_group.addButton(self.bottom_x_axis_radio, 0)
        self.x_axis_button_group.addButton(self.top_x_axis_radio, 1)
        
        x_axis_layout.addWidget(self.bottom_x_axis_radio)
        x_axis_layout.addWidget(self.top_x_axis_radio)
        position_layout.addRow("X-Axis:", x_axis_layout)
        
        layout.addRow(position_group)
        
        # Frequency Scale
        self.freq_scale_combo = QComboBox()
        scales = ['linear', 'log', 'symlog', 'logit']
        self.freq_scale_combo.addItems(scales)
        layout.addRow("Frequency Scale:", self.freq_scale_combo)
        
        # Time Scale
        self.time_scale_combo = QComboBox()
        self.time_scale_combo.addItems(scales)
        layout.addRow("Time Scale:", self.time_scale_combo)
        
        # Frequency Limits
        freq_limits_group = QGroupBox("Frequency Limits")
        freq_limits_layout = QFormLayout(freq_limits_group)
        
        self.freq_min_edit = QLineEdit()
        self.freq_min_edit.setPlaceholderText("Auto")
        freq_limits_layout.addRow("Min Frequency (Hz):", self.freq_min_edit)
        
        self.freq_max_edit = QLineEdit()
        self.freq_max_edit.setPlaceholderText("Auto")
        freq_limits_layout.addRow("Max Frequency (Hz):", self.freq_max_edit)
        
        layout.addRow(freq_limits_group)
        
        # Time Limits
        time_limits_group = QGroupBox("Time Limits")
        time_limits_layout = QFormLayout(time_limits_group)
        
        self.time_min_edit = QLineEdit()
        self.time_min_edit.setPlaceholderText("Auto")
        time_limits_layout.addRow("Min Time (s):", self.time_min_edit)
        
        self.time_max_edit = QLineEdit()
        self.time_max_edit.setPlaceholderText("Auto")
        time_limits_layout.addRow("Max Time (s):", self.time_max_edit)
        
        layout.addRow(time_limits_group)
        
        return tab
    
    def _create_advanced_tab(self) -> QWidget:
        """Create the advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Info section
        info_group = QGroupBox("Spectrogram Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        # Additional options
        options_group = QGroupBox("Display Options")
        options_layout = QFormLayout(options_group)
        
        # Aspect ratio
        self.aspect_combo = QComboBox()
        aspects = ['auto', 'equal', '1', '2', '0.5']
        self.aspect_combo.addItems(aspects)
        options_layout.addRow("Aspect Ratio:", self.aspect_combo)
        
        # Shading
        self.shading_combo = QComboBox()
        shadings = ['gouraud', 'flat', 'nearest']
        self.shading_combo.addItems(shadings)
        options_layout.addRow("Shading:", self.shading_combo)
        
        layout.addWidget(options_group)
        
        layout.addStretch()
        
        return tab
    
    def _connect_preview_signals(self):
        """Connect all signals for live preview updates"""
        # Basic tab
        self.legend_edit.textChanged.connect(self.update_preview)
        self.colormap_combo.currentTextChanged.connect(self.update_preview)
        self.alpha_spin.valueChanged.connect(self.update_preview)
        self.interpolation_combo.currentTextChanged.connect(self.update_preview)
        
        # Colorbar tab
        self.show_colorbar_checkbox.toggled.connect(self.update_preview)
        self.colorbar_position_combo.currentTextChanged.connect(self.update_preview)
        self.colorbar_label_edit.textChanged.connect(self.update_preview)
        self.auto_range_checkbox.toggled.connect(self.update_preview)
        self.clim_min_spin.valueChanged.connect(self.update_preview)
        self.clim_max_spin.valueChanged.connect(self.update_preview)
        self.colorbar_pad_spin.valueChanged.connect(self.update_preview)
        self.colorbar_shrink_spin.valueChanged.connect(self.update_preview)
        self.colorbar_aspect_spin.valueChanged.connect(self.update_preview)
        self.colorbar_ticks_spin.valueChanged.connect(self.update_preview)
        self.tick_format_combo.currentTextChanged.connect(self.update_preview)
        self.colorbar_label_fontsize_spin.valueChanged.connect(self.update_preview)
        self.colorbar_tick_fontsize_spin.valueChanged.connect(self.update_preview)
        
        # Axis tab
        self.x_axis_button_group.buttonToggled.connect(self.update_preview)
        self.freq_scale_combo.currentTextChanged.connect(self.update_preview)
        self.time_scale_combo.currentTextChanged.connect(self.update_preview)
        self.freq_min_edit.textChanged.connect(self.update_preview)
        self.freq_max_edit.textChanged.connect(self.update_preview)
        self.time_min_edit.textChanged.connect(self.update_preview)
        self.time_max_edit.textChanged.connect(self.update_preview)
        
        # Advanced tab
        self.aspect_combo.currentTextChanged.connect(self.update_preview)
        self.shading_combo.currentTextChanged.connect(self.update_preview)
    
    def _on_auto_range_toggled(self, checked):
        """Handle auto range checkbox toggle"""
        self.clim_min_spin.setEnabled(not checked)
        self.clim_max_spin.setEnabled(not checked)
        self.update_preview()
    
    def load_channel_properties(self):
        """Load current channel properties into the UI"""
        # Basic settings
        legend_label = getattr(self.channel, 'legend_label', None) or getattr(self.channel, 'ylabel', '')
        self.legend_edit.setText(legend_label)
        
        colormap = getattr(self.channel, 'colormap', 'viridis')
        index = self.colormap_combo.findText(colormap)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)
        
        alpha = getattr(self.channel, 'alpha', 1.0)
        self.alpha_spin.setValue(alpha)
        
        interpolation = getattr(self.channel, 'interpolation', 'nearest')
        index = self.interpolation_combo.findText(interpolation)
        if index >= 0:
            self.interpolation_combo.setCurrentIndex(index)
        
        # Colorbar settings
        show_colorbar = getattr(self.channel, 'show_colorbar', True)
        self.show_colorbar_checkbox.setChecked(show_colorbar)
        
        colorbar_position = getattr(self.channel, 'colorbar_position', 'right')
        index = self.colorbar_position_combo.findText(colorbar_position)
        if index >= 0:
            self.colorbar_position_combo.setCurrentIndex(index)
        
        colorbar_label = getattr(self.channel, 'colorbar_label', 'Power (dB)')
        self.colorbar_label_edit.setText(colorbar_label)
        
        # Precise positioning
        colorbar_pad = getattr(self.channel, 'colorbar_pad', 0.05)
        self.colorbar_pad_spin.setValue(colorbar_pad)
        
        colorbar_shrink = getattr(self.channel, 'colorbar_shrink', 0.8)
        self.colorbar_shrink_spin.setValue(colorbar_shrink)
        
        colorbar_aspect = getattr(self.channel, 'colorbar_aspect', 20)
        self.colorbar_aspect_spin.setValue(colorbar_aspect)
        
        # Tick and label settings
        colorbar_ticks = getattr(self.channel, 'colorbar_ticks', 5)
        self.colorbar_ticks_spin.setValue(colorbar_ticks)
        
        tick_format = getattr(self.channel, 'tick_format', '%.1f')
        self.tick_format_combo.setCurrentText(tick_format)
        
        colorbar_label_fontsize = getattr(self.channel, 'colorbar_label_fontsize', 10)
        self.colorbar_label_fontsize_spin.setValue(colorbar_label_fontsize)
        
        colorbar_tick_fontsize = getattr(self.channel, 'colorbar_tick_fontsize', 8)
        self.colorbar_tick_fontsize_spin.setValue(colorbar_tick_fontsize)
        
        # Color range
        clim_auto = getattr(self.channel, 'clim_auto', True)
        self.auto_range_checkbox.setChecked(clim_auto)
        
        clim_min = getattr(self.channel, 'clim_min', -60.0)
        self.clim_min_spin.setValue(clim_min if clim_min is not None else -60.0)
        
        clim_max = getattr(self.channel, 'clim_max', 0.0)
        self.clim_max_spin.setValue(clim_max if clim_max is not None else 0.0)
        
        # Axis settings
        freq_scale = getattr(self.channel, 'freq_scale', 'linear')
        index = self.freq_scale_combo.findText(freq_scale)
        if index >= 0:
            self.freq_scale_combo.setCurrentIndex(index)
        
        time_scale = getattr(self.channel, 'time_scale', 'linear')
        index = self.time_scale_combo.findText(time_scale)
        if index >= 0:
            self.time_scale_combo.setCurrentIndex(index)
        
        # X-axis position
        channel_xaxis = getattr(self.channel, 'xaxis', 'x-bottom')
        if channel_xaxis == 'x-bottom':
            self.bottom_x_axis_radio.setChecked(True)
        else:
            self.top_x_axis_radio.setChecked(True)
        
        # Frequency limits
        freq_limits = getattr(self.channel, 'freq_limits', [None, None])
        if freq_limits[0] is not None:
            self.freq_min_edit.setText(str(freq_limits[0]))
        if freq_limits[1] is not None:
            self.freq_max_edit.setText(str(freq_limits[1]))
        
        # Time limits
        time_limits = getattr(self.channel, 'time_limits', [None, None])
        if time_limits[0] is not None:
            self.time_min_edit.setText(str(time_limits[0]))
        if time_limits[1] is not None:
            self.time_max_edit.setText(str(time_limits[1]))
        
        # Advanced settings
        aspect = getattr(self.channel, 'aspect', 'auto')
        index = self.aspect_combo.findText(str(aspect))
        if index >= 0:
            self.aspect_combo.setCurrentIndex(index)
        
        shading = getattr(self.channel, 'shading', 'gouraud')
        index = self.shading_combo.findText(shading)
        if index >= 0:
            self.shading_combo.setCurrentIndex(index)
        
        # Update info text
        self._update_info_text()
        
        # Initial preview update
        self.update_preview()
    
    def _update_info_text(self):
        """Update the information text with channel details"""
        try:
            info_lines = []
            info_lines.append(f"Channel ID: {getattr(self.channel, 'channel_id', 'Unknown')}")
            info_lines.append(f"Type: {getattr(self.channel, 'type', 'Unknown')}")
            
            # Check if this is a spectrogram with metadata
            if hasattr(self.channel, 'metadata') and 'Zxx' in self.channel.metadata:
                Zxx = self.channel.metadata['Zxx']
                info_lines.append(f"Spectrogram Shape: {Zxx.shape}")
                info_lines.append(f"Data Type: Pre-computed spectrogram")
                
                # Get frequency and time axis info
                if hasattr(self.channel, 'ydata') and hasattr(self.channel, 'xdata'):
                    freq_range = f"{np.min(self.channel.ydata):.1f} - {np.max(self.channel.ydata):.1f} Hz"
                    time_range = f"{np.min(self.channel.xdata):.2f} - {np.max(self.channel.xdata):.2f} s"
                    info_lines.append(f"Frequency Range: {freq_range}")
                    info_lines.append(f"Time Range: {time_range}")
            else:
                # Regular signal that will be converted to spectrogram
                if hasattr(self.channel, 'ydata'):
                    info_lines.append(f"Signal Length: {len(self.channel.ydata)} samples")
                    info_lines.append(f"Data Type: Time-domain signal")
                    
                    # Estimate frequency range (assuming Fs = 1 for now)
                    max_freq = 0.5  # Nyquist frequency
                    info_lines.append(f"Est. Frequency Range: 0 - {max_freq:.1f} Hz")
            
            self.info_text.setText("\n".join(info_lines))
        except Exception as e:
            self.info_text.setText(f"Error loading channel info: {str(e)}")
    
    def update_preview(self):
        """Update the preview display"""
        try:
            # Get current settings
            legend_name = self.legend_edit.text() or getattr(self.channel, 'ylabel', 'Unnamed')
            colormap = self.colormap_combo.currentText()
            alpha = self.alpha_spin.value()
            interpolation = self.interpolation_combo.currentText()
            
            show_colorbar = "Yes" if self.show_colorbar_checkbox.isChecked() else "No"
            colorbar_position = self.colorbar_position_combo.currentText()
            colorbar_label = self.colorbar_label_edit.text() or "Power (dB)"
            
            auto_range = "Yes" if self.auto_range_checkbox.isChecked() else "No"
            clim_range = f"{self.clim_min_spin.value():.1f} to {self.clim_max_spin.value():.1f} dB"
            
            freq_scale = self.freq_scale_combo.currentText()
            time_scale = self.time_scale_combo.currentText()
            
            aspect = self.aspect_combo.currentText()
            shading = self.shading_combo.currentText()
            
            # Create preview text
            preview_text = f"""
            <div style="padding: 5px; font-size: 11px;">
                <b>Legend:</b> {legend_name}<br>
                <b>Colormap:</b> {colormap}<br>
                <b>Transparency:</b> {alpha:.2f} ({int(alpha*100)}%)<br>
                <b>Interpolation:</b> {interpolation}<br>
                <br>
                <b>Colorbar:</b> {show_colorbar} at {colorbar_position}<br>
                <b>Colorbar Label:</b> {colorbar_label}<br>
                <b>Auto Range:</b> {auto_range}<br>
                <b>Range:</b> {clim_range}<br>
                <br>
                <b>Freq Scale:</b> {freq_scale} | <b>Time Scale:</b> {time_scale}<br>
                <b>Aspect:</b> {aspect} | <b>Shading:</b> {shading}
            </div>
            """
            
            self.preview_label.setText(preview_text)
        except Exception as e:
            self.preview_label.setText(f"Preview error: {str(e)}")
    
    def apply_changes(self):
        """Apply changes to the channel without closing dialog"""
        self._update_channel_properties()
        self.spectrogram_updated.emit(self.channel.channel_id)
    
    def accept(self):
        """Apply changes and close dialog"""
        self._update_channel_properties()
        self.spectrogram_updated.emit(self.channel.channel_id)
        super().accept()
    
    def reject(self):
        """Cancel changes and restore original properties"""
        self._restore_channel_properties()
        super().reject()
    
    def _update_channel_properties(self):
        """Update channel properties from UI"""
        # Basic settings
        self.channel.legend_label = self.legend_edit.text() or None
        self.channel.colormap = self.colormap_combo.currentText()
        self.channel.alpha = self.alpha_spin.value()
        self.channel.interpolation = self.interpolation_combo.currentText()
        
        # Colorbar settings
        self.channel.show_colorbar = self.show_colorbar_checkbox.isChecked()
        self.channel.colorbar_position = self.colorbar_position_combo.currentText()
        self.channel.colorbar_label = self.colorbar_label_edit.text()
        
        # Precise positioning
        self.channel.colorbar_pad = self.colorbar_pad_spin.value()
        self.channel.colorbar_shrink = self.colorbar_shrink_spin.value()
        self.channel.colorbar_aspect = self.colorbar_aspect_spin.value()
        
        # Tick and label settings
        self.channel.colorbar_ticks = self.colorbar_ticks_spin.value()
        self.channel.tick_format = self.tick_format_combo.currentText()
        self.channel.colorbar_label_fontsize = self.colorbar_label_fontsize_spin.value()
        self.channel.colorbar_tick_fontsize = self.colorbar_tick_fontsize_spin.value()
        
        # Color range
        self.channel.clim_auto = self.auto_range_checkbox.isChecked()
        if not self.channel.clim_auto:
            self.channel.clim_min = self.clim_min_spin.value()
            self.channel.clim_max = self.clim_max_spin.value()
        else:
            self.channel.clim_min = None
            self.channel.clim_max = None
        
        # Axis settings
        self.channel.freq_scale = self.freq_scale_combo.currentText()
        self.channel.time_scale = self.time_scale_combo.currentText()
        
        # X-axis position
        self.channel.xaxis = "x-bottom" if self.bottom_x_axis_radio.isChecked() else "x-top"
        
        # Frequency limits
        freq_limits = [None, None]
        try:
            if self.freq_min_edit.text().strip():
                freq_limits[0] = float(self.freq_min_edit.text())
        except ValueError:
            pass
        try:
            if self.freq_max_edit.text().strip():
                freq_limits[1] = float(self.freq_max_edit.text())
        except ValueError:
            pass
        self.channel.freq_limits = freq_limits
        
        # Time limits
        time_limits = [None, None]
        try:
            if self.time_min_edit.text().strip():
                time_limits[0] = float(self.time_min_edit.text())
        except ValueError:
            pass
        try:
            if self.time_max_edit.text().strip():
                time_limits[1] = float(self.time_max_edit.text())
        except ValueError:
            pass
        self.channel.time_limits = time_limits
        
        # Advanced settings
        self.channel.aspect = self.aspect_combo.currentText()
        self.channel.shading = self.shading_combo.currentText()
        
        # Update modification time
        from datetime import datetime
        self.channel.modified_at = datetime.now()
    
    def _restore_channel_properties(self):
        """Restore original channel properties"""
        for key, value in self.original_properties.items():
            setattr(self.channel, key, value)
    
    @staticmethod
    def edit_spectrogram(channel, parent=None) -> bool:
        """
        Static method to edit a spectrogram channel's properties
        
        Args:
            channel: Channel to edit
            parent: Parent widget
            
        Returns:
            True if changes were applied, False if cancelled
        """
        dialog = SpectrogramWizard(channel, parent)
        result = dialog.exec()
        return result == QDialog.Accepted


# Convenience function for opening the spectrogram wizard
def open_spectrogram_wizard(channel, parent=None) -> bool:
    """
    Open the spectrogram wizard for editing a spectrogram channel
    
    Args:
        channel: Channel to edit
        parent: Parent widget
        
    Returns:
        True if changes were applied, False if cancelled
    """
    return SpectrogramWizard.edit_spectrogram(channel, parent) 