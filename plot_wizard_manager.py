from PySide6.QtCore import QObject, Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QMessageBox, QTableWidgetItem, QColorDialog, QComboBox, QCheckBox, 
    QSpinBox, QFileDialog, QInputDialog, QHeaderView, QDialog, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox, QDoubleSpinBox
)
from PySide6.QtGui import QColor
from plot_wizard_window import PlotWizardWindow
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import traceback
from pathlib import Path

class AdvancedSubplotDialog(QDialog):
    """Dialog for advanced subplot configuration options"""
    
    def __init__(self, subplot_num, subplot_config, current_axis=None, parent=None):
        super().__init__(parent)
        self.subplot_num = subplot_num
        self.subplot_config = subplot_config.copy()  # Work on a copy
        self.current_axis = current_axis
        
        self.setWindowTitle(f"Advanced Settings - Subplot {subplot_num}")
        self.setMinimumSize(400, 300)
        self.setModal(True)
        
        self._setup_ui()
        self._populate_fields()
        
    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Axis scales group
        scales_group = QGroupBox("Axis Scales")
        scales_layout = QFormLayout(scales_group)
        
        self.x_scale_combo = QComboBox()
        self.x_scale_combo.addItems(["linear", "log"])
        scales_layout.addRow("X Scale:", self.x_scale_combo)
        
        self.y_scale_combo = QComboBox()
        self.y_scale_combo.addItems(["linear", "log"])
        scales_layout.addRow("Y Scale:", self.y_scale_combo)
        
        layout.addWidget(scales_group)
        
        # Axis limits group
        limits_group = QGroupBox("Axis Limits")
        limits_layout = QFormLayout(limits_group)
        
        xlim_layout = QHBoxLayout()
        self.xlim_min = QLineEdit()
        self.xlim_min.setPlaceholderText("Auto")
        self.xlim_max = QLineEdit()
        self.xlim_max.setPlaceholderText("Auto")
        xlim_layout.addWidget(self.xlim_min)
        xlim_layout.addWidget(QLabel("to"))
        xlim_layout.addWidget(self.xlim_max)
        limits_layout.addRow("X Limits:", xlim_layout)
        
        ylim_layout = QHBoxLayout()
        self.ylim_min = QLineEdit()
        self.ylim_min.setPlaceholderText("Auto")
        self.ylim_max = QLineEdit()
        self.ylim_max.setPlaceholderText("Auto")
        ylim_layout.addWidget(self.ylim_min)
        ylim_layout.addWidget(QLabel("to"))
        ylim_layout.addWidget(self.ylim_max)
        limits_layout.addRow("Y Limits:", ylim_layout)
        
        layout.addWidget(limits_group)
        
        # Tick options group
        ticks_group = QGroupBox("Tick Options")
        ticks_layout = QFormLayout(ticks_group)
        
        self.x_tick_labels = QCheckBox()
        self.x_tick_labels.setChecked(True)
        ticks_layout.addRow("Show X Tick Labels:", self.x_tick_labels)
        
        self.y_tick_labels = QCheckBox()
        self.y_tick_labels.setChecked(True)
        ticks_layout.addRow("Show Y Tick Labels:", self.y_tick_labels)
        
        # Major tick spacing
        self.major_tick_spacing = QComboBox()
        self.major_tick_spacing.addItems(["auto", "multiple", "fixed", "max_n"])
        ticks_layout.addRow("Major Tick Spacing:", self.major_tick_spacing)
        
        # Major tick value
        self.major_tick_value = QDoubleSpinBox()
        self.major_tick_value.setRange(0.001, 1000000.0)
        self.major_tick_value.setDecimals(3)
        self.major_tick_value.setValue(1.0)
        ticks_layout.addRow("Major Tick Value:", self.major_tick_value)
        
        # Major tick count
        self.major_tick_count = QSpinBox()
        self.major_tick_count.setRange(2, 50)
        self.major_tick_count.setValue(10)
        ticks_layout.addRow("Major Tick Count:", self.major_tick_count)
        
        # Minor ticks
        self.minor_ticks_on = QCheckBox()
        self.minor_ticks_on.setChecked(False)
        ticks_layout.addRow("Minor Ticks On:", self.minor_ticks_on)
        
        # Minor tick spacing
        self.minor_tick_spacing = QDoubleSpinBox()
        self.minor_tick_spacing.setRange(0.001, 1000000.0)
        self.minor_tick_spacing.setDecimals(3)
        self.minor_tick_spacing.setValue(0.2)
        ticks_layout.addRow("Minor Tick Spacing:", self.minor_tick_spacing)
        
        # Connect signals for dynamic enabling/disabling
        self.major_tick_spacing.currentTextChanged.connect(self._on_tick_spacing_changed)
        self.minor_ticks_on.stateChanged.connect(self._on_minor_ticks_changed)
        
        layout.addWidget(ticks_group)
        
        # Axis labels group
        labels_group = QGroupBox("Axis Labels")
        labels_layout = QFormLayout(labels_group)
        
        self.y_left_label = QLineEdit()
        self.y_left_label.setPlaceholderText("Left Y-axis label")
        labels_layout.addRow("Y-Left Label:", self.y_left_label)
        
        self.y_right_label = QLineEdit()
        self.y_right_label.setPlaceholderText("Right Y-axis label")
        labels_layout.addRow("Y-Right Label:", self.y_right_label)
        
        layout.addWidget(labels_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
    def _on_tick_spacing_changed(self, spacing_type):
        """Handle major tick spacing type change"""
        try:
            # Enable/disable controls based on spacing type
            self.major_tick_value.setEnabled(spacing_type in ["multiple", "fixed"])
            self.major_tick_count.setEnabled(spacing_type == "max_n")
        except Exception as e:
            print(f"[AdvancedDialog] Error handling tick spacing change: {e}")
            
    def _on_minor_ticks_changed(self, state):
        """Handle minor ticks checkbox change"""
        try:
            # Enable/disable minor tick spacing based on checkbox
            self.minor_tick_spacing.setEnabled(state == 2)  # Qt.Checked
        except Exception as e:
            print(f"[AdvancedDialog] Error handling minor ticks change: {e}")
        
    def _populate_fields(self):
        """Populate fields with current values"""
        try:
            # Set scales
            self.x_scale_combo.setCurrentText(self.subplot_config.get('x_scale', 'linear'))
            self.y_scale_combo.setCurrentText(self.subplot_config.get('y_scale', 'linear'))
            
            # Set limits from current axis if available
            if self.current_axis:
                xlim = self.current_axis.get_xlim()
                ylim = self.current_axis.get_ylim()
                
                self.xlim_min.setText(f"{xlim[0]:.3f}")
                self.xlim_max.setText(f"{xlim[1]:.3f}")
                self.ylim_min.setText(f"{ylim[0]:.3f}")
                self.ylim_max.setText(f"{ylim[1]:.3f}")
            else:
                # Use stored limits if available
                xlim = self.subplot_config.get('xlim', [None, None])
                ylim = self.subplot_config.get('ylim', [None, None])
                
                if xlim[0] is not None:
                    self.xlim_min.setText(str(xlim[0]))
                if xlim[1] is not None:
                    self.xlim_max.setText(str(xlim[1]))
                if ylim[0] is not None:
                    self.ylim_min.setText(str(ylim[0]))
                if ylim[1] is not None:
                    self.ylim_max.setText(str(ylim[1]))
            
            # Set tick label options
            self.x_tick_labels.setChecked(self.subplot_config.get('show_x_tick_labels', True))
            self.y_tick_labels.setChecked(self.subplot_config.get('show_y_tick_labels', True))
            
            # Set axis labels
            self.y_left_label.setText(self.subplot_config.get('y_left_label', ''))
            self.y_right_label.setText(self.subplot_config.get('y_right_label', ''))
            
            # Set tick control options
            self.major_tick_spacing.setCurrentText(self.subplot_config.get('major_tick_spacing', 'auto'))
            self.major_tick_value.setValue(self.subplot_config.get('major_tick_value', 1.0))
            self.major_tick_count.setValue(self.subplot_config.get('major_tick_count', 10))
            self.minor_ticks_on.setChecked(self.subplot_config.get('minor_ticks_on', False))
            self.minor_tick_spacing.setValue(self.subplot_config.get('minor_tick_spacing', 0.2))
            
            # Update control states based on current values
            self._on_tick_spacing_changed(self.major_tick_spacing.currentText())
            self._on_minor_ticks_changed(2 if self.minor_ticks_on.isChecked() else 0)
            
        except Exception as e:
            print(f"[AdvancedDialog] Error populating fields: {e}")
            
    def get_config(self):
        """Get the updated configuration"""
        try:
            # Parse limits
            xlim = [None, None]
            ylim = [None, None]
            
            try:
                if self.xlim_min.text().strip():
                    xlim[0] = float(self.xlim_min.text())
                if self.xlim_max.text().strip():
                    xlim[1] = float(self.xlim_max.text())
            except ValueError:
                xlim = [None, None]
                
            try:
                if self.ylim_min.text().strip():
                    ylim[0] = float(self.ylim_min.text())
                if self.ylim_max.text().strip():
                    ylim[1] = float(self.ylim_max.text())
            except ValueError:
                ylim = [None, None]
            
            # Update config
            self.subplot_config.update({
                'x_scale': self.x_scale_combo.currentText(),
                'y_scale': self.y_scale_combo.currentText(),
                'xlim': xlim,
                'ylim': ylim,
                'show_x_tick_labels': self.x_tick_labels.isChecked(),
                'show_y_tick_labels': self.y_tick_labels.isChecked(),
                'y_left_label': self.y_left_label.text().strip(),
                'y_right_label': self.y_right_label.text().strip(),
                'major_tick_spacing': self.major_tick_spacing.currentText(),
                'major_tick_value': self.major_tick_value.value(),
                'major_tick_count': self.major_tick_count.value(),
                'minor_ticks_on': self.minor_ticks_on.isChecked(),
                'minor_tick_spacing': self.minor_tick_spacing.value()
            })
            
            return self.subplot_config
            
        except Exception as e:
            print(f"[AdvancedDialog] Error getting config: {e}")
            return self.subplot_config

class AdvancedSpectrogramDialog(QDialog):
    """Dialog for advanced spectrogram-specific configuration options"""
    
    def __init__(self, subplot_num, subplot_config, current_axis=None, parent=None):
        super().__init__(parent)
        self.subplot_num = subplot_num
        self.subplot_config = subplot_config.copy()  # Work on a copy
        self.current_axis = current_axis
        
        self.setWindowTitle(f"Advanced Spectrogram Settings - Subplot {subplot_num}")
        self.setMinimumSize(400, 300)
        self.setModal(True)
        
        self._setup_ui()
        self._populate_fields()
        
    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Colormap group
        colormap_group = QGroupBox("Colormap Settings")
        colormap_layout = QFormLayout(colormap_group)
        
        self.colormap_combo = QComboBox()
        # Popular matplotlib colormaps for spectrograms
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'coolwarm', 
                    'seismic', 'RdBu', 'jet', 'turbo', 'nipy_spectral', 'gist_rainbow']
        self.colormap_combo.addItems(colormaps)
        colormap_layout.addRow("Colormap:", self.colormap_combo)
        
        layout.addWidget(colormap_group)
        
        # Color limits group
        clim_group = QGroupBox("Color Limits (clim)")
        clim_layout = QFormLayout(clim_group)
        
        clim_layout_h = QHBoxLayout()
        self.clim_min = QLineEdit()
        self.clim_min.setPlaceholderText("Auto")
        self.clim_max = QLineEdit()
        self.clim_max.setPlaceholderText("Auto")
        clim_layout_h.addWidget(self.clim_min)
        clim_layout_h.addWidget(QLabel("to"))
        clim_layout_h.addWidget(self.clim_max)
        clim_layout.addRow("Color Range:", clim_layout_h)
        
        layout.addWidget(clim_group)
        
        # Axis scales group (same as regular advanced)
        scales_group = QGroupBox("Axis Scales")
        scales_layout = QFormLayout(scales_group)
        
        self.x_scale_combo = QComboBox()
        self.x_scale_combo.addItems(["linear", "log"])
        scales_layout.addRow("X Scale:", self.x_scale_combo)
        
        self.y_scale_combo = QComboBox()
        self.y_scale_combo.addItems(["linear", "log"])
        scales_layout.addRow("Y Scale:", self.y_scale_combo)
        
        layout.addWidget(scales_group)
        
        # Axis limits group (same as regular advanced)
        limits_group = QGroupBox("Axis Limits")
        limits_layout = QFormLayout(limits_group)
        
        xlim_layout = QHBoxLayout()
        self.xlim_min = QLineEdit()
        self.xlim_min.setPlaceholderText("Auto")
        self.xlim_max = QLineEdit()
        self.xlim_max.setPlaceholderText("Auto")
        xlim_layout.addWidget(self.xlim_min)
        xlim_layout.addWidget(QLabel("to"))
        xlim_layout.addWidget(self.xlim_max)
        limits_layout.addRow("X Limits:", xlim_layout)
        
        ylim_layout = QHBoxLayout()
        self.ylim_min = QLineEdit()
        self.ylim_min.setPlaceholderText("Auto")
        self.ylim_max = QLineEdit()
        self.ylim_max.setPlaceholderText("Auto")
        ylim_layout.addWidget(self.ylim_min)
        ylim_layout.addWidget(QLabel("to"))
        ylim_layout.addWidget(self.ylim_max)
        limits_layout.addRow("Y Limits:", ylim_layout)
        
        layout.addWidget(limits_group)
        
        # Axis labels group
        labels_group = QGroupBox("Axis Labels")
        labels_layout = QFormLayout(labels_group)
        
        self.x_label = QLineEdit()
        self.x_label.setPlaceholderText("X-axis label")
        labels_layout.addRow("X Label:", self.x_label)
        
        self.y_label = QLineEdit()
        self.y_label.setPlaceholderText("Y-axis label")
        labels_layout.addRow("Y Label:", self.y_label)
        
        self.colorbar_label = QLineEdit()
        self.colorbar_label.setPlaceholderText("Colorbar label")
        labels_layout.addRow("Colorbar Label:", self.colorbar_label)
        
        layout.addWidget(labels_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
    def _populate_fields(self):
        """Populate fields with current values"""
        try:
            # Set colormap
            self.colormap_combo.setCurrentText(self.subplot_config.get('colormap', 'viridis'))
            
            # Set color limits
            clim = self.subplot_config.get('clim', [None, None])
            if clim[0] is not None:
                self.clim_min.setText(str(clim[0]))
            if clim[1] is not None:
                self.clim_max.setText(str(clim[1]))
            
            # Set scales
            self.x_scale_combo.setCurrentText(self.subplot_config.get('x_scale', 'linear'))
            self.y_scale_combo.setCurrentText(self.subplot_config.get('y_scale', 'linear'))
            
            # Set limits from current axis if available
            if self.current_axis:
                xlim = self.current_axis.get_xlim()
                ylim = self.current_axis.get_ylim()
                
                self.xlim_min.setText(f"{xlim[0]:.3f}")
                self.xlim_max.setText(f"{xlim[1]:.3f}")
                self.ylim_min.setText(f"{ylim[0]:.3f}")
                self.ylim_max.setText(f"{ylim[1]:.3f}")
            else:
                # Use stored limits if available
                xlim = self.subplot_config.get('xlim', [None, None])
                ylim = self.subplot_config.get('ylim', [None, None])
                
                if xlim[0] is not None:
                    self.xlim_min.setText(str(xlim[0]))
                if xlim[1] is not None:
                    self.xlim_max.setText(str(xlim[1]))
                if ylim[0] is not None:
                    self.ylim_min.setText(str(ylim[0]))
                if ylim[1] is not None:
                    self.ylim_max.setText(str(ylim[1]))
            
            # Set axis labels
            self.x_label.setText(self.subplot_config.get('xlabel', ''))
            self.y_label.setText(self.subplot_config.get('ylabel', ''))
            self.colorbar_label.setText(self.subplot_config.get('colorbar_label', ''))
            
        except Exception as e:
            print(f"[AdvancedSpectrogramDialog] Error populating fields: {e}")
            
    def get_config(self):
        """Get the updated configuration"""
        try:
            # Parse limits
            xlim = [None, None]
            ylim = [None, None]
            clim = [None, None]
            
            try:
                if self.xlim_min.text().strip():
                    xlim[0] = float(self.xlim_min.text())
                if self.xlim_max.text().strip():
                    xlim[1] = float(self.xlim_max.text())
            except ValueError:
                xlim = [None, None]
                
            try:
                if self.ylim_min.text().strip():
                    ylim[0] = float(self.ylim_min.text())
                if self.ylim_max.text().strip():
                    ylim[1] = float(self.ylim_max.text())
            except ValueError:
                ylim = [None, None]
                
            try:
                if self.clim_min.text().strip():
                    clim[0] = float(self.clim_min.text())
                if self.clim_max.text().strip():
                    clim[1] = float(self.clim_max.text())
            except ValueError:
                clim = [None, None]
            
            # Update config
            self.subplot_config.update({
                'colormap': self.colormap_combo.currentText(),
                'clim': clim,
                'x_scale': self.x_scale_combo.currentText(),
                'y_scale': self.y_scale_combo.currentText(),
                'xlim': xlim,
                'ylim': ylim,
                'xlabel': self.x_label.text().strip(),
                'ylabel': self.y_label.text().strip(),
                'colorbar_label': self.colorbar_label.text().strip()
            })
            
            return self.subplot_config
            
        except Exception as e:
            print(f"[AdvancedSpectrogramDialog] Error getting config: {e}")
            return self.subplot_config

class PlotWizardManager(QObject):
    """
    Manager for the custom plot wizard that handles:
    - Channel selection and configuration
    - Multiple subplot management
    - Line styling and customization
    - Plot rendering and saving
    - Configuration persistence
    - Mixed plot type support (Bar+Line, Spectrogram+Line, etc.)
    """
    
    plot_created = Signal(dict)
    
    def __init__(self, file_manager, channel_manager, signal_bus, parent=None):
        super().__init__(parent)
        
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        
        # Plot configuration storage
        self.plot_items = []  # List of plot item configurations
        self.subplot_configs = {}  # Subplot configurations
        self.global_config = {}  # Global plot settings
        
        # Subplot dimensions
        self.subplot_rows = 1
        self.subplot_cols = 1
        
        # Plot update optimization
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._perform_plot_update)
        self.update_delay = 150  # ms delay for debouncing
        self.plot_update_in_progress = False
        
        # Cache for expensive operations
        self.last_applied_config = {}
        self.axes_cache = {}
        
        # Track user vs auto dimension changes
        self.user_set_dimensions = False
        self.updating_dimensions_programmatically = False
        
        # Create window
        self.window = PlotWizardWindow(file_manager, channel_manager)
        
        # Style options
        self.line_styles = ['-', '--', '-.', ':']
        self.line_style_names = ['Solid', 'Dashed', 'Dash-dot', 'Dotted']
        self.markers = ['o', 's', '^', 'v', 'D', '*', '+', 'x', '.', ',']
        self.marker_names = ['Circle', 'Square', 'Triangle Up', 'Triangle Down', 
                           'Diamond', 'Star', 'Plus', 'X', 'Point', 'Pixel']
        
        # Color cycle
        self.default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 
                             'pink', 'gray', 'olive', 'cyan']
        self.color_index = 0
        
        self._setup_window()
        self._connect_signals()
        self._populate_dropdowns()
        
    def _setup_window(self):
        """Setup the window with initial configurations"""
        # Configure tables
        self._setup_line_config_table()
        self._setup_subplot_config_table()
        
        # Set initial global configuration to match window placeholder
        self.global_config = {
            'sharex': False,
            'sharey': False,
            'tick_direction': 'in',
            'tick_width': 1.0,
            'tick_length': 4.0,
            'line_width': 2.0,
            'axis_linewidth': 1.5,
            'font_size': 12,
            'font_weight': 'normal',
            'font_family': 'sans-serif',
            'grid': True,
            'box_style': 'full',
            'legend_fontsize': 10,
            'legend_ncol': 1,
            'legend_frameon': True,
            'tight_layout': True
        }
        
        # Configuration controls will be updated after connection
        
    def _setup_line_config_table(self):
        """Setup the line configuration table"""
        table = self.window.line_config_table
        
        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Subplot#
        header.setSectionResizeMode(1, QHeaderView.Stretch)          # Legend
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Color
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Line
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Marker
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Y Axis
        
    def _setup_subplot_config_table(self):
        """Setup the subplot configuration table"""
        table = self.window.subplot_config_table
        
        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Subplot#
        header.setSectionResizeMode(1, QHeaderView.Stretch)          # Xlabel
        header.setSectionResizeMode(2, QHeaderView.Stretch)          # Ylabel
        header.setSectionResizeMode(3, QHeaderView.Stretch)          # Legend Label
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Legend
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Legend Pos
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Advanced
        
    def _connect_signals(self):
        """Connect window signals to manager methods"""
        # Add button
        self.window.add_btn.clicked.connect(self._on_add_to_plot)
        
        # File/channel selection
        self.window.file_dropdown.currentTextChanged.connect(self._on_file_changed)
        self.window.channel_dropdown.currentTextChanged.connect(self._on_channel_changed)
        
        # Table interactions
        self.window.line_config_table.cellDoubleClicked.connect(self._on_line_config_cell_clicked)
        self.window.subplot_config_table.cellDoubleClicked.connect(self._on_subplot_config_cell_clicked)
        self.window.subplot_config_table.cellChanged.connect(self._on_subplot_config_cell_changed)
        
        # Subplot dimension controls
        self.window.rows_spinbox.valueChanged.connect(self._on_dimension_changed)
        self.window.cols_spinbox.valueChanged.connect(self._on_dimension_changed)
        
        # Plot configuration controls
        self._connect_config_controls()
        
        # Initialize controls with current config values
        self._update_config_controls()
        
    def _connect_config_controls(self):
        """Connect all plot configuration controls to update methods"""
        try:
            # Axes settings
            self.window.sharex_checkbox.stateChanged.connect(self._on_config_changed)
            self.window.sharey_checkbox.stateChanged.connect(self._on_config_changed)
            
            # Tick settings
            self.window.tick_direction_combo.currentTextChanged.connect(self._on_config_changed)
            self.window.tick_width_spinbox.valueChanged.connect(self._on_config_changed)
            self.window.tick_length_spinbox.valueChanged.connect(self._on_config_changed)
            
            # Line and axis settings
            self.window.line_width_spinbox.valueChanged.connect(self._on_config_changed)
            self.window.axis_linewidth_spinbox.valueChanged.connect(self._on_config_changed)
            
            # Font settings
            self.window.font_size_spinbox.valueChanged.connect(self._on_config_changed)
            self.window.font_weight_combo.currentTextChanged.connect(self._on_config_changed)
            self.window.font_family_combo.currentTextChanged.connect(self._on_config_changed)
            
            # Display settings
            self.window.grid_checkbox.stateChanged.connect(self._on_config_changed)
            self.window.box_style_combo.currentTextChanged.connect(self._on_config_changed)
            
            # Legend settings
            self.window.legend_fontsize_spinbox.valueChanged.connect(self._on_config_changed)
            self.window.legend_ncol_spinbox.valueChanged.connect(self._on_config_changed)
            self.window.legend_frameon_checkbox.stateChanged.connect(self._on_config_changed)
            self.window.tight_layout_checkbox.stateChanged.connect(self._on_config_changed)
            
        except Exception as e:
            print(f"[PlotWizard] Error connecting config controls: {e}")
        
    def _populate_dropdowns(self):
        """Populate file and channel dropdowns"""
        try:
            # Clear existing items
            self.window.file_dropdown.clear()
            self.window.channel_dropdown.clear()
            
            # Get all files with parsed channels
            all_files = self.file_manager.get_all_files()
            parsed_files = []
            
            for file_info in all_files:
                channels = self.channel_manager.get_channels_by_file(file_info.file_id)
                if channels:  # File has channels
                    parsed_files.append(file_info)
            
            if not parsed_files:
                self.window.file_dropdown.addItem("No parsed files available")
                return
            
            # Add files to dropdown
            for file_info in parsed_files:
                self.window.file_dropdown.addItem(file_info.filename)
            
            # Populate channels for first file
            if parsed_files:
                self._on_file_changed(parsed_files[0].filename)
                
        except Exception as e:
            print(f"[PlotWizard] Error populating dropdowns: {str(e)}")
            
    def _on_file_changed(self, filename):
        """Handle file selection change"""
        try:
            self.window.channel_dropdown.clear()
            
            if not filename or filename == "No parsed files available":
                return
            
            # Find file by filename
            file_info = None
            for f in self.file_manager.get_all_files():
                if f.filename == filename:
                    file_info = f
                    break
            
            if not file_info:
                return
            
            # Get channels for this file
            channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            
            if channels:
                channel_names = []
                for ch in channels:
                    name = ch.legend_label or ch.channel_id
                    if name not in channel_names:
                        channel_names.append(name)
                
                self.window.channel_dropdown.addItems(channel_names)
                
                # Auto-set plot type based on first channel if available
                if channels:
                    self._update_plot_type_for_channel(channels[0])
                
        except Exception as e:
            print(f"[PlotWizard] Error updating channels: {str(e)}")
            
    def _update_plot_type_for_channel(self, channel):
        """Update plot type dropdown based on channel tags"""
        try:
            # Check channel tags to determine plot type
            if hasattr(channel, 'tags') and channel.tags:
                if 'spectrogram' in channel.tags:
                    plot_type = 'Spectrogram'
                else:
                    plot_type = 'Line'  # Default for time-series
            else:
                plot_type = 'Line'  # Default fallback
            
            # Update the dropdown
            index = self.window.type_dropdown.findText(plot_type)
            if index >= 0:
                self.window.type_dropdown.setCurrentIndex(index)
                print(f"[PlotWizard] Auto-detected plot type: {plot_type}")
            
        except Exception as e:
            print(f"[PlotWizard] Error updating plot type: {str(e)}")
            
    def _on_channel_changed(self, channel_name):
        """Handle channel selection change"""
        try:
            if not channel_name:
                return
                
            # Get current file
            filename = self.window.file_dropdown.currentText()
            if not filename or filename == "No parsed files available":
                return
                
            # Get the channel object
            channel = self._get_channel(filename, channel_name)
            if channel:
                # Update plot type based on channel
                self._update_plot_type_for_channel(channel)
                
                # Auto-populate legend entry
                if not self.window.legend_entry.text().strip():
                    self.window.legend_entry.setText(channel_name)
                
        except Exception as e:
            print(f"[PlotWizard] Error handling channel change: {str(e)}")
            
    def _on_add_to_plot(self):
        """Handle adding a channel to the plot"""
        try:
            # Get current selections
            filename = self.window.file_dropdown.currentText()
            channel_name = self.window.channel_dropdown.currentText()
            plot_type = self.window.type_dropdown.currentText()
            legend_name = self.window.legend_entry.text().strip()
            
            if not all([filename, channel_name]):
                QMessageBox.warning(self.window, "Selection Error", 
                                  "Please select both a file and channel.")
                return
            
            if filename == "No parsed files available":
                QMessageBox.warning(self.window, "No Data", 
                                  "No parsed files available.")
                return
            
            # Get the channel object
            channel = self._get_channel(filename, channel_name)
            if not channel:
                QMessageBox.warning(self.window, "Channel Error", 
                                  f"Could not find channel '{channel_name}' in file '{filename}'.")
                return
            
            # Generate legend name if not provided
            if not legend_name:
                legend_name = f"{channel_name} ({filename})"
            
            # Determine subplot number based on auto-stacking logic
            if not self.user_set_dimensions:
                # Auto-stacking mode: each new plot gets its own subplot
                existing_subplots = set(item['subplot'] for item in self.plot_items)
                if existing_subplots:
                    subplot_num = max(existing_subplots) + 1
                else:
                    subplot_num = 1
                print(f"[PlotWizard] Auto-stacking: assigning to subplot {subplot_num}")
            else:
                # User has set dimensions: default to subplot 1 (user can change later)
                subplot_num = 1
                print(f"[PlotWizard] User-dimension mode: assigning to subplot {subplot_num}")
            
            # Get next color
            color = self.default_colors[self.color_index % len(self.default_colors)]
            self.color_index += 1
            
            # Determine y-axis assignment based on mixed plot types
            y_axis = 'left'  # default
            
            # Check if this subplot already has different plot types
            existing_items = [item for item in self.plot_items if item['subplot'] == subplot_num]
            existing_types = set(item['plot_type'] for item in existing_items)
            
            if existing_types and plot_type not in existing_types:
                # We're adding a different plot type to this subplot
                if plot_type == 'Spectrogram':
                    # Spectrograms go to left axis
                    y_axis = 'left'
                    # Move existing non-spectrograms to right axis
                    for item in existing_items:
                        if item['plot_type'] != 'Spectrogram':
                            item['y_axis'] = 'right'
                elif 'Spectrogram' in existing_types:
                    # Adding non-spectrogram to subplot that has spectrograms
                    y_axis = 'right'  # Non-spectrograms go to right axis
            
            # Create plot item configuration
            plot_item = {
                'id': len(self.plot_items),
                'filename': filename,
                'channel_name': channel_name,
                'channel': channel,
                'plot_type': plot_type,
                'legend_name': legend_name,
                'subplot': subplot_num,
                'color': color,
                'line_style': '-',
                'marker': 'o' if plot_type == 'Scatter' else 'None',
                'y_axis': y_axis,
                'visible': True
            }
            
            print(f"[PlotWizard] Added {plot_type} to subplot {subplot_num} on {y_axis} axis")
            
            # Add to plot items
            self.plot_items.append(plot_item)
            
            # Ensure subplot configuration exists
            if subplot_num not in self.subplot_configs:
                self._create_subplot_config(subplot_num)
            
            # Update tables
            self._update_line_config_table()
            self._update_subplot_config_table()
            
            # Clear legend entry for next item
            self.window.legend_entry.clear()
            
            # Auto-update subplot dimensions
            self._auto_update_dimensions()
            
            # Update plot
            self._update_plot()
            
            print(f"[PlotWizard] Added {legend_name} to subplot {subplot_num}")
            
        except Exception as e:
            print(f"[PlotWizard] Error adding to plot: {str(e)}")
            traceback.print_exc()
            QMessageBox.critical(self.window, "Error", f"Failed to add to plot: {str(e)}")
            
    def _get_channel(self, filename, channel_name):
        """Get channel object by filename and channel name"""
        try:
            # Find file by filename
            file_info = None
            for f in self.file_manager.get_all_files():
                if f.filename == filename:
                    file_info = f
                    break
            
            if not file_info:
                return None
            
            # Get channels for this file
            channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            
            # Find matching channel
            for channel in channels:
                name = channel.legend_label or channel.channel_id
                if name == channel_name:
                    return channel
            return None
            
        except Exception as e:
            print(f"[PlotWizard] Error getting channel: {str(e)}")
            return None
            
    def _create_subplot_config(self, subplot_num):
        """Create default configuration for a subplot"""
        self.subplot_configs[subplot_num] = {
            'subplot': subplot_num,
            'xlabel': 'Time',
            'ylabel': 'Amplitude',
            'show_legend': True,
            'legend_position': 'upper right',
            'colorbar_position': 'bottom',  # Default for spectrograms
            'show_colorbar': True,  # Default for spectrograms
            # Advanced settings
            'x_scale': 'linear',
            'y_scale': 'linear',
            'xlim': [None, None],
            'ylim': [None, None],
            'show_x_tick_labels': True,
            'show_y_tick_labels': True,
            'y_left_label': '',
            'y_right_label': ''
        }
        
    def _update_line_config_table(self):
        """Update the line configuration table"""
        try:
            table = self.window.line_config_table
            table.setRowCount(len(self.plot_items))
            
            for i, item in enumerate(self.plot_items):
                # Subplot number (editable)
                subplot_spinbox = QSpinBox()
                subplot_spinbox.setMinimum(1)
                subplot_spinbox.setMaximum(10)
                subplot_spinbox.setValue(item['subplot'])
                subplot_spinbox.valueChanged.connect(lambda v, idx=i: self._on_subplot_changed(idx, v))
                table.setCellWidget(i, 0, subplot_spinbox)
                
                # Legend name (editable)
                table.setItem(i, 1, QTableWidgetItem(item['legend_name']))
                
                # Color (clickable)
                color_item = QTableWidgetItem(item['color'])
                color_item.setBackground(QColor(item['color']))
                table.setItem(i, 2, color_item)
                
                # Line style (dropdown) - disabled for spectrograms
                line_combo = QComboBox()
                line_combo.addItems(self.line_style_names)
                try:
                    style_idx = self.line_styles.index(item['line_style'])
                    line_combo.setCurrentIndex(style_idx)
                except ValueError:
                    line_combo.setCurrentIndex(0)
                line_combo.currentIndexChanged.connect(lambda idx, row=i: self._on_line_style_changed(row, idx))
                
                # Disable for spectrograms
                if item['plot_type'] == 'Spectrogram':
                    line_combo.setEnabled(False)
                    line_combo.setToolTip("Line style not applicable to spectrograms")
                
                table.setCellWidget(i, 3, line_combo)
                
                # Marker (dropdown) - disabled for spectrograms
                marker_combo = QComboBox()
                marker_names_with_none = ['None'] + self.marker_names
                marker_combo.addItems(marker_names_with_none)
                try:
                    if item['marker'] == 'None':
                        marker_combo.setCurrentIndex(0)
                    else:
                        marker_idx = self.markers.index(item['marker']) + 1
                        marker_combo.setCurrentIndex(marker_idx)
                except ValueError:
                    marker_combo.setCurrentIndex(0)
                marker_combo.currentIndexChanged.connect(lambda idx, row=i: self._on_marker_changed(row, idx))
                
                # Disable for spectrograms
                if item['plot_type'] == 'Spectrogram':
                    marker_combo.setEnabled(False)
                    marker_combo.setToolTip("Markers not applicable to spectrograms")
                
                table.setCellWidget(i, 4, marker_combo)
                
                # Y Axis (dropdown)
                y_axis_combo = QComboBox()
                y_axis_combo.addItems(['left', 'right'])
                y_axis_combo.setCurrentText(item['y_axis'])
                y_axis_combo.currentTextChanged.connect(lambda text, row=i: self._on_y_axis_changed(row, text))
                table.setCellWidget(i, 5, y_axis_combo)
                
        except Exception as e:
            print(f"[PlotWizard] Error updating line config table: {str(e)}")
            
    def _update_subplot_config_table(self):
        """Update the subplot configuration table with separate rows for different plot types"""
        try:
            table = self.window.subplot_config_table
            subplot_nums = sorted(set(item['subplot'] for item in self.plot_items))
            
            # Create configuration rows - one per subplot per plot type
            config_rows = []
            for subplot_num in subplot_nums:
                subplot_items = [item for item in self.plot_items if item['subplot'] == subplot_num]
                plot_types = set(item['plot_type'] for item in subplot_items)
                
                # If mixed types, create separate rows for each type
                if len(plot_types) > 1:
                    for plot_type in sorted(plot_types):  # Sort for consistent ordering
                        config_rows.append({
                            'subplot_num': subplot_num,
                            'plot_type': plot_type,
                            'mixed': True
                        })
                else:
                    # Single type, one row
                    config_rows.append({
                        'subplot_num': subplot_num,
                        'plot_type': list(plot_types)[0] if plot_types else 'Line',
                        'mixed': False
                    })
            
            table.setRowCount(len(config_rows))
            
            # Check if any subplot contains spectrograms to determine column header
            has_any_spectrogram = any(item['plot_type'] == 'Spectrogram' for item in self.plot_items)
            if has_any_spectrogram:
                # Update column header to reflect mixed usage
                table.setHorizontalHeaderLabels(["Subplot#", "Xlabel", "Ylabel", "Legend Label", "Legend", "Legend/Colorbar Pos", "Advanced"])
            else:
                # Standard legend-only header
                table.setHorizontalHeaderLabels(["Subplot#", "Xlabel", "Ylabel", "Legend Label", "Legend", "Legend Pos", "Advanced"])
            
            # Temporarily disconnect signal to avoid recursion during updates
            try:
                table.cellChanged.disconnect()
            except RuntimeError:
                # Signal wasn't connected yet, which is fine
                pass
            
            for i, row_config in enumerate(config_rows):
                subplot_num = row_config['subplot_num']
                plot_type = row_config['plot_type']
                is_mixed = row_config['mixed']
                
                if subplot_num not in self.subplot_configs:
                    self._create_subplot_config(subplot_num)
                
                config = self.subplot_configs[subplot_num]
                
                # Subplot number (read-only) - show type suffix for mixed subplots
                if is_mixed:
                    display_text = f"{subplot_num} ({plot_type})"
                else:
                    display_text = str(subplot_num)
                subplot_item = QTableWidgetItem(display_text)
                subplot_item.setFlags(subplot_item.flags() & ~Qt.ItemIsEditable)
                subplot_item.setData(Qt.UserRole, {'subplot_num': subplot_num, 'plot_type': plot_type})
                table.setItem(i, 0, subplot_item)
                
                # X Label (editable) - same for all types in subplot, only editable on first row
                xlabel_item = QTableWidgetItem(config['xlabel'])
                xlabel_item.setData(Qt.UserRole, {'subplot_num': subplot_num, 'plot_type': plot_type})
                
                # Check if this is the first row for this subplot
                is_first_row_for_subplot = not any(
                    config_rows[j]['subplot_num'] == subplot_num 
                    for j in range(i)
                )
                
                if not is_first_row_for_subplot:
                    # Disable editing for duplicate subplot labels
                    xlabel_item.setFlags(xlabel_item.flags() & ~Qt.ItemIsEditable)
                    xlabel_item.setBackground(QColor(240, 240, 240))  # Light gray background
                    xlabel_item.setToolTip("X label is shared - edit in the first row for this subplot")
                
                table.setItem(i, 1, xlabel_item)
                
                # Y Label (editable) - type-specific for mixed subplots
                if is_mixed:
                    if plot_type == 'Spectrogram':
                        ylabel_text = config.get('y_left_label', '').strip()
                        if not ylabel_text:
                            ylabel_text = 'Frequency (Hz)'
                    else:  # Line plots
                        ylabel_text = config.get('y_right_label', '').strip()
                        if not ylabel_text:
                            ylabel_text = 'Amplitude'
                else:
                    ylabel_text = config.get('y_left_label', '').strip()
                    if not ylabel_text:
                        ylabel_text = config['ylabel']
                
                ylabel_item = QTableWidgetItem(ylabel_text)
                ylabel_item.setData(Qt.UserRole, {'subplot_num': subplot_num, 'plot_type': plot_type})
                table.setItem(i, 2, ylabel_item)
                
                # Legend Label (editable) - collect legend names for this type
                legend_labels = []
                for item in self.plot_items:
                    if item['subplot'] == subplot_num and (not is_mixed or item['plot_type'] == plot_type):
                        legend_labels.append(item['legend_name'])
                
                legend_label_text = "; ".join(legend_labels) if legend_labels else ""
                legend_label_item = QTableWidgetItem(legend_label_text)
                legend_label_item.setData(Qt.UserRole, {'subplot_num': subplot_num, 'plot_type': plot_type})
                table.setItem(i, 3, legend_label_item)
                
                # Show Legend (checkbox) - type-specific for mixed subplots
                legend_checkbox = QCheckBox()
                if is_mixed and plot_type == 'Spectrogram':
                    # For spectrograms in mixed subplots, this controls colorbar visibility
                    legend_checkbox.setChecked(config.get('show_colorbar', True))
                    legend_checkbox.setToolTip("Show/hide colorbar for spectrograms")
                else:
                    # For regular plots, this controls legend visibility
                    legend_checkbox.setChecked(config['show_legend'])
                    legend_checkbox.setToolTip("Show/hide legend for line plots")
                
                legend_checkbox.stateChanged.connect(lambda state, sn=subplot_num, pt=plot_type: self._on_legend_show_changed(sn, state, pt))
                table.setCellWidget(i, 4, legend_checkbox)
                
                # Legend Position (dropdown) - or Colorbar Position for spectrograms
                legend_pos_combo = QComboBox()
                
                if plot_type == 'Spectrogram':
                    # Use colorbar positions for spectrograms
                    colorbar_positions = ['bottom', 'top', 'right', 'left']
                    legend_pos_combo.addItems(colorbar_positions)
                    current_pos = config.get('colorbar_position', 'bottom')
                    if current_pos not in colorbar_positions:
                        current_pos = 'bottom'
                    legend_pos_combo.setCurrentText(current_pos)
                    legend_pos_combo.setToolTip("Controls colorbar position for spectrograms")
                else:
                    # Use standard legend positions for regular plots
                    legend_positions = ['upper right', 'upper left', 'lower left', 'lower right', 
                                      'right', 'center left', 'center right', 'lower center', 'upper center', 'center', 'best']
                    legend_pos_combo.addItems(legend_positions)
                    legend_pos_combo.setCurrentText(config['legend_position'])
                    legend_pos_combo.setToolTip("Controls legend position")
                
                legend_pos_combo.currentTextChanged.connect(lambda text, sn=subplot_num, pt=plot_type: self._on_legend_pos_changed(sn, text, pt))
                table.setCellWidget(i, 5, legend_pos_combo)
                
                # Advanced Settings Button
                advanced_button = QPushButton("Edit")
                advanced_button.setMaximumWidth(60)
                advanced_button.clicked.connect(lambda checked, sn=subplot_num, pt=plot_type: self._on_advanced_settings_clicked(sn, pt))
                table.setCellWidget(i, 6, advanced_button)
            
            # Reconnect signal
            try:
                table.cellChanged.connect(self._on_subplot_config_cell_changed)
            except Exception as e:
                print(f"[PlotWizard] Warning: Could not reconnect cellChanged signal: {e}")
                
        except Exception as e:
            print(f"[PlotWizard] Error updating subplot config table: {str(e)}")
            
    def _on_subplot_changed(self, item_index, new_subplot):
        """Handle subplot number change"""
        try:
            if 0 <= item_index < len(self.plot_items):
                self.plot_items[item_index]['subplot'] = new_subplot
                
                # Ensure subplot config exists
                if new_subplot not in self.subplot_configs:
                    self._create_subplot_config(new_subplot)
                
                self._update_subplot_config_table()
                self._update_plot()
                
        except Exception as e:
            print(f"[PlotWizard] Error changing subplot: {str(e)}")
            
    def _on_line_style_changed(self, item_index, style_index):
        """Handle line style change"""
        try:
            if 0 <= item_index < len(self.plot_items) and 0 <= style_index < len(self.line_styles):
                self.plot_items[item_index]['line_style'] = self.line_styles[style_index]
                self._update_plot()
                
        except Exception as e:
            print(f"[PlotWizard] Error changing line style: {str(e)}")
            
    def _on_marker_changed(self, item_index, marker_index):
        """Handle marker change"""
        try:
            if 0 <= item_index < len(self.plot_items):
                if marker_index == 0:  # None
                    self.plot_items[item_index]['marker'] = 'None'
                elif 1 <= marker_index <= len(self.markers):
                    self.plot_items[item_index]['marker'] = self.markers[marker_index - 1]
                self._update_plot()
                
        except Exception as e:
            print(f"[PlotWizard] Error changing marker: {str(e)}")
            
    def _on_y_axis_changed(self, item_index, y_axis):
        """Handle Y axis change"""
        try:
            if 0 <= item_index < len(self.plot_items):
                self.plot_items[item_index]['y_axis'] = y_axis
                self._update_plot()
                
        except Exception as e:
            print(f"[PlotWizard] Error changing Y axis: {str(e)}")
            
    def _on_legend_show_changed(self, subplot_num, state, plot_type=None):
        """Handle legend show/hide change"""
        try:
            if subplot_num in self.subplot_configs:
                if plot_type == 'Spectrogram':
                    # Control colorbar visibility for spectrograms
                    self.subplot_configs[subplot_num]['show_colorbar'] = state == 2  # Qt.Checked
                    print(f"[PlotWizard] Updated colorbar visibility for subplot {subplot_num} to: {state == 2}")
                else:
                    # Control legend visibility for regular plots
                    self.subplot_configs[subplot_num]['show_legend'] = state == 2  # Qt.Checked
                    print(f"[PlotWizard] Updated legend visibility for subplot {subplot_num} to: {state == 2}")
                
                self._update_plot()
                
        except Exception as e:
            print(f"[PlotWizard] Error changing legend/colorbar visibility: {str(e)}")
            
    def _on_legend_pos_changed(self, subplot_num, position, plot_type=None):
        """Handle legend position change (or colorbar position for spectrograms)"""
        try:
            if subplot_num in self.subplot_configs:
                if plot_type == 'Spectrogram':
                    # Store as colorbar position for spectrograms
                    self.subplot_configs[subplot_num]['colorbar_position'] = position
                    print(f"[PlotWizard] Updated colorbar position for subplot {subplot_num} to: {position}")
                else:
                    # Store as legend position for regular plots
                    self.subplot_configs[subplot_num]['legend_position'] = position
                    print(f"[PlotWizard] Updated legend position for subplot {subplot_num} to: {position}")
                
                self._update_plot()
                
        except Exception as e:
            print(f"[PlotWizard] Error changing legend/colorbar position: {str(e)}")
            
    def _on_subplot_config_cell_changed(self, row, column):
        """Handle inline editing of subplot configuration cells"""
        try:
            table = self.window.subplot_config_table
            item = table.item(row, column)
            
            if not item:
                return
                
            # Get subplot info from the item data
            item_data = item.data(Qt.UserRole)
            if not item_data:
                return
                
            subplot_num = item_data['subplot_num']
            plot_type = item_data['plot_type']
            
            if subplot_num not in self.subplot_configs:
                return
                
            config = self.subplot_configs[subplot_num]
            new_value = item.text().strip()
            
            if column == 1:  # X Label - shared across all types in subplot
                config['xlabel'] = new_value
                print(f"[PlotWizard] Updated subplot {subplot_num} xlabel to: {new_value}")
                
                # Update all other rows for this subplot to show the same X label
                for other_row in range(table.rowCount()):
                    other_item = table.item(other_row, 0)  # Subplot number column
                    if other_item:
                        other_data = other_item.data(Qt.UserRole)
                        if other_data and other_data['subplot_num'] == subplot_num and other_row != row:
                            other_xlabel_item = table.item(other_row, 1)
                            if other_xlabel_item:
                                other_xlabel_item.setText(new_value)
                
                self._update_plot()
                
            elif column == 2:  # Y Label - type-specific for mixed subplots
                subplot_items = [item for item in self.plot_items if item['subplot'] == subplot_num]
                plot_types = set(item['plot_type'] for item in subplot_items)
                is_mixed = len(plot_types) > 1
                
                if is_mixed:
                    if plot_type == 'Spectrogram':
                        config['y_left_label'] = new_value
                        print(f"[PlotWizard] Updated subplot {subplot_num} spectrogram Y label (left) to: {new_value}")
                    else:  # Line plots
                        config['y_right_label'] = new_value
                        print(f"[PlotWizard] Updated subplot {subplot_num} line Y label (right) to: {new_value}")
                else:
                    # Single type - update both for compatibility
                    config['ylabel'] = new_value
                    config['y_left_label'] = new_value
                    print(f"[PlotWizard] Updated subplot {subplot_num} ylabel to: {new_value}")
                
                self._update_plot()
                
            elif column == 3:  # Legend Label - update plot items for this type
                # Parse the legend labels (separated by semicolons)
                legend_labels = [label.strip() for label in new_value.split(';') if label.strip()]
                
                # Find plot items for this subplot and type
                subplot_items = [item for item in self.plot_items 
                               if item['subplot'] == subplot_num and item['plot_type'] == plot_type]
                
                # Update legend names for each item of this type
                for i, item in enumerate(subplot_items):
                    if i < len(legend_labels):
                        item['legend_name'] = legend_labels[i]
                    else:
                        # If fewer labels provided than items, keep original
                        pass
                        
                print(f"[PlotWizard] Updated legend labels for subplot {subplot_num} {plot_type}")
                
                # Update the line config table to reflect changes
                self._update_line_config_table()
                self._update_plot()
                
        except Exception as e:
            print(f"[PlotWizard] Error handling subplot config cell change: {str(e)}")
            
    def _on_advanced_settings_clicked(self, subplot_num, plot_type=None):
        """Handle advanced settings button click"""
        try:
            if subplot_num not in self.subplot_configs:
                self._create_subplot_config(subplot_num)
            
            config = self.subplot_configs[subplot_num]
            
            # Get current axis if plot exists
            current_axis = None
            if hasattr(self, 'current_axes') and subplot_num in self.current_axes:
                current_axis = self.current_axes[subplot_num]['left']
            
            # Choose appropriate dialog based on plot type (if specified) or subplot content
            if plot_type == 'Spectrogram':
                # Use spectrogram-specific dialog
                dialog = AdvancedSpectrogramDialog(subplot_num, config, current_axis, self.window)
            elif plot_type and plot_type != 'Spectrogram':
                # Use regular advanced dialog for non-spectrogram types
                dialog = AdvancedSubplotDialog(subplot_num, config, current_axis, self.window)
            else:
                # Fallback: Check if this subplot contains spectrograms
                subplot_items = [item for item in self.plot_items if item['subplot'] == subplot_num]
                has_spectrogram = any(item['plot_type'] == 'Spectrogram' for item in subplot_items)
                
                if has_spectrogram:
                    # Use spectrogram-specific dialog
                    dialog = AdvancedSpectrogramDialog(subplot_num, config, current_axis, self.window)
                else:
                    # Use regular advanced dialog
                    dialog = AdvancedSubplotDialog(subplot_num, config, current_axis, self.window)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Update configuration with dialog results
                updated_config = dialog.get_config()
                self.subplot_configs[subplot_num].update(updated_config)
                
                print(f"[PlotWizard] Updated advanced settings for subplot {subplot_num} ({plot_type or 'mixed'})")
                
                # Update the subplot config table to reflect changes
                self._update_subplot_config_table()
                
                # Update plot to reflect changes
                self._update_plot()
                
        except Exception as e:
            print(f"[PlotWizard] Error handling advanced settings: {str(e)}")
            QMessageBox.critical(self.window, "Error", f"Failed to open advanced settings: {str(e)}")
            
    def _on_line_config_cell_clicked(self, row, column):
        """Handle line config table cell double-click"""
        try:
            if row >= len(self.plot_items):
                return
                
            item = self.plot_items[row]
            
            if column == 1:  # Legend name
                current_name = item['legend_name']
                new_name, ok = QInputDialog.getText(
                    self.window, 
                    "Edit Legend Name", 
                    "Legend name:", 
                    text=current_name
                )
                if ok and new_name.strip():
                    item['legend_name'] = new_name.strip()
                    self._update_line_config_table()
                    self._update_plot()
                    
            elif column == 2:  # Color
                current_color = QColor(item['color'])
                new_color = QColorDialog.getColor(current_color, self.window, "Choose Color")
                if new_color.isValid():
                    item['color'] = new_color.name()
                    self._update_line_config_table()
                    self._update_plot()
                    
        except Exception as e:
            print(f"[PlotWizard] Error handling cell click: {str(e)}")
            
    def _on_subplot_config_cell_clicked(self, row, column):
        """Handle subplot config table cell double-click"""
        # Inline editing is now handled by _on_subplot_config_cell_changed
        # This method is kept for any future double-click functionality
        pass
            
    def _on_config_changed(self):
        """Handle configuration control changes"""
        try:
            # Axes settings
            self.global_config['sharex'] = self.window.sharex_checkbox.isChecked()
            self.global_config['sharey'] = self.window.sharey_checkbox.isChecked()
            
            # Tick settings
            self.global_config['tick_direction'] = self.window.tick_direction_combo.currentText()
            self.global_config['tick_width'] = self.window.tick_width_spinbox.value()
            self.global_config['tick_length'] = self.window.tick_length_spinbox.value()
            
            # Line and axis settings
            self.global_config['line_width'] = self.window.line_width_spinbox.value()
            self.global_config['axis_linewidth'] = self.window.axis_linewidth_spinbox.value()
            
            # Font settings
            self.global_config['font_size'] = self.window.font_size_spinbox.value()
            self.global_config['font_weight'] = self.window.font_weight_combo.currentText()
            self.global_config['font_family'] = self.window.font_family_combo.currentText()
            
            # Display settings
            self.global_config['grid'] = self.window.grid_checkbox.isChecked()
            self.global_config['box_style'] = self.window.box_style_combo.currentText()
            
            # Legend settings
            self.global_config['legend_fontsize'] = self.window.legend_fontsize_spinbox.value()
            self.global_config['legend_ncol'] = self.window.legend_ncol_spinbox.value()
            self.global_config['legend_frameon'] = self.window.legend_frameon_checkbox.isChecked()
            self.global_config['tight_layout'] = self.window.tight_layout_checkbox.isChecked()
            
            # Update plot
            self._update_plot()
            
        except Exception as e:
            print(f"[PlotWizard] Error handling config change: {str(e)}")
            
    def _on_dimension_changed(self):
        """Handle subplot dimension changes"""
        try:
            self.subplot_rows = self.window.rows_spinbox.value()
            self.subplot_cols = self.window.cols_spinbox.value()
            
            # Only mark as user-set if this isn't a programmatic update
            if not self.updating_dimensions_programmatically:
                self.user_set_dimensions = True
                print(f"[PlotWizard] User changed subplot dimensions to {self.subplot_rows}×{self.subplot_cols}")
            else:
                print(f"[PlotWizard] Programmatically updated dimensions to {self.subplot_rows}×{self.subplot_cols}")
            
            # Check if we need to reassign subplot numbers
            max_positions = self.subplot_rows * self.subplot_cols
            current_subplot_nums = set(item['subplot'] for item in self.plot_items)
            
            if current_subplot_nums and max(current_subplot_nums) > max_positions:
                print(f"[PlotWizard] Some subplots exceed new grid capacity, reassigning...")
                self._reassign_subplot_numbers(max_positions)
                
                # Update tables to reflect changes
                self._update_line_config_table()
                self._update_subplot_config_table()
            
            # Update plot with new dimensions
            self._update_plot()
            
        except Exception as e:
            print(f"[PlotWizard] Error changing subplot dimensions: {str(e)}")
            
    def _reassign_subplot_numbers(self, max_positions):
        """Reassign subplot numbers to fit within the grid capacity"""
        try:
            # Get unique subplot numbers
            current_subplots = sorted(set(item['subplot'] for item in self.plot_items))
            
            # Create mapping for reassignment
            reassignment_map = {}
            
            for i, old_subplot in enumerate(current_subplots):
                new_subplot = min(i + 1, max_positions)
                reassignment_map[old_subplot] = new_subplot
            
            # Apply reassignment
            for item in self.plot_items:
                old_subplot = item['subplot']
                if old_subplot in reassignment_map:
                    item['subplot'] = reassignment_map[old_subplot]
                    print(f"[PlotWizard] Reassigned subplot {old_subplot} → {reassignment_map[old_subplot]}")
            
            # Update subplot configs
            new_subplot_configs = {}
            for old_subplot, new_subplot in reassignment_map.items():
                if old_subplot in self.subplot_configs:
                    new_subplot_configs[new_subplot] = self.subplot_configs[old_subplot]
                    
            # Replace subplot configs with reassigned ones
            for old_subplot in list(self.subplot_configs.keys()):
                if old_subplot not in reassignment_map.values():
                    del self.subplot_configs[old_subplot]
            
            self.subplot_configs.update(new_subplot_configs)
            
        except Exception as e:
            print(f"[PlotWizard] Error reassigning subplot numbers: {e}")
            
    def _schedule_plot_update(self):
        """Schedule a plot update with debouncing to avoid excessive refreshes"""
        try:
            # Stop any pending update
            self.update_timer.stop()
            
            # Schedule new update
            self.update_timer.start(self.update_delay)
            
        except Exception as e:
            print(f"[PlotWizard] Error scheduling plot update: {e}")
            
    def _perform_plot_update(self):
        """Perform the actual plot update"""
        try:
            if self.plot_update_in_progress:
                return
                
            self.plot_update_in_progress = True
            self._update_plot_optimized()
            
        except Exception as e:
            print(f"[PlotWizard] Error performing plot update: {e}")
        finally:
            self.plot_update_in_progress = False
            
    def _auto_update_dimensions(self):
        """Automatically calculate and update subplot dimensions based on number of subplots"""
        try:
            if not self.plot_items:
                self.subplot_rows = 1
                self.subplot_cols = 1
                self.user_set_dimensions = False  # Reset when no plots
            else:
                # Get unique subplot numbers
                subplot_nums = set(item['subplot'] for item in self.plot_items)
                n_subplots = len(subplot_nums)
                
                if not self.user_set_dimensions:
                    # User hasn't manually set dimensions - use N×1 layout
                    rows = n_subplots
                    cols = 1
                    print(f"[PlotWizard] Auto-setting dimensions: {rows}×{cols} (N×1 layout)")
                else:
                    # User has set dimensions - check if current capacity is sufficient
                    current_capacity = self.subplot_rows * self.subplot_cols
                    
                    if n_subplots <= current_capacity:
                        # Current dimensions can accommodate all subplots
                        rows = self.subplot_rows
                        cols = self.subplot_cols
                        print(f"[PlotWizard] Keeping user dimensions: {rows}×{cols} (capacity: {current_capacity})")
                    else:
                        # Need more capacity - increment first dimension (rows)
                        rows = self.subplot_rows + 1
                        cols = self.subplot_cols
                        print(f"[PlotWizard] Expanding user dimensions: {self.subplot_rows}×{self.subplot_cols} → {rows}×{cols}")
                    
                self.subplot_rows = rows
                self.subplot_cols = cols
            
            # Update spinboxes programmatically (set flag to prevent marking as user change)
            self.updating_dimensions_programmatically = True
            
            self.window.rows_spinbox.setValue(self.subplot_rows)
            self.window.cols_spinbox.setValue(self.subplot_cols)
            
            # Reset flag
            self.updating_dimensions_programmatically = False
            
            print(f"[PlotWizard] Final dimensions: {self.subplot_rows}×{self.subplot_cols}")
            
        except Exception as e:
            print(f"[PlotWizard] Error auto-updating dimensions: {str(e)}")
            
    def _apply_figure_settings(self):
        """Apply global figure settings"""
        try:
            # Font settings
            import matplotlib
            font_props = {}
            if 'font_size' in self.global_config:
                font_props['size'] = self.global_config['font_size']
            if 'font_family' in self.global_config:
                font_props['family'] = self.global_config['font_family']
            if 'font_weight' in self.global_config:
                font_props['weight'] = self.global_config['font_weight']
            
            if font_props:
                try:
                    matplotlib.rc('font', **font_props)
                except Exception as font_error:
                    print(f"[PlotWizard] Warning: Could not apply font settings: {font_error}")
                
        except Exception as e:
            print(f"[PlotWizard] Error applying figure settings: {e}")
            
    def _apply_axis_styling(self, ax):
        """Apply axis styling from global configuration"""
        try:
            # Tick properties
            tick_direction = self.global_config.get('tick_direction', 'in')
            tick_width = self.global_config.get('tick_width', 1.0)
            tick_length = self.global_config.get('tick_length', 4.0)
            axis_linewidth = self.global_config.get('axis_linewidth', 1.5)
            
            # Apply tick settings
            ax.tick_params(
                direction=tick_direction,
                width=tick_width,
                length=tick_length,
                which='both'  # Apply to both major and minor ticks
            )
            
            # Apply axis line width
            for spine in ax.spines.values():
                spine.set_linewidth(axis_linewidth)
            
            # Box style
            box_style = self.global_config.get('box_style', 'full')
            if box_style == 'full':
                # Keep all spines
                pass
            elif box_style == 'left_bottom':
                # Only left and bottom spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            elif box_style == 'none':
                # No spines
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    
        except Exception as e:
            print(f"[PlotWizard] Error applying axis styling: {e}")
            
    def _apply_advanced_settings(self, ax_left, ax_right, config):
        """Apply advanced subplot settings to axes"""
        try:
            # Apply axis scales
            x_scale = config.get('x_scale', 'linear')
            y_scale = config.get('y_scale', 'linear')
            
            ax_left.set_xscale(x_scale)
            ax_left.set_yscale(y_scale)
            
            if ax_right is not None:
                ax_right.set_xscale(x_scale)
                # Right axis might have different scale, but for now use same as left
                ax_right.set_yscale(y_scale)
            
            # Apply axis limits
            xlim = config.get('xlim', [None, None])
            ylim = config.get('ylim', [None, None])
            
            if xlim[0] is not None or xlim[1] is not None:
                current_xlim = ax_left.get_xlim()
                new_xlim = [
                    xlim[0] if xlim[0] is not None else current_xlim[0],
                    xlim[1] if xlim[1] is not None else current_xlim[1]
                ]
                ax_left.set_xlim(new_xlim)
                if ax_right is not None:
                    ax_right.set_xlim(new_xlim)
            
            if ylim[0] is not None or ylim[1] is not None:
                current_ylim = ax_left.get_ylim()
                new_ylim = [
                    ylim[0] if ylim[0] is not None else current_ylim[0],
                    ylim[1] if ylim[1] is not None else current_ylim[1]
                ]
                ax_left.set_ylim(new_ylim)
            
            # Apply tick label visibility
            show_x_tick_labels = config.get('show_x_tick_labels', True)
            show_y_tick_labels = config.get('show_y_tick_labels', True)
            
            if not show_x_tick_labels:
                ax_left.set_xticklabels([])
                if ax_right is not None:
                    ax_right.set_xticklabels([])
            
            if not show_y_tick_labels:
                ax_left.set_yticklabels([])
                if ax_right is not None:
                    ax_right.set_yticklabels([])
            
            # Apply tick controls
            self._apply_tick_controls(ax_left, ax_right, config)
            
            # Apply custom axis labels
            y_left_label = config.get('y_left_label', '').strip()
            y_right_label = config.get('y_right_label', '').strip()
            
            if y_left_label:
                ax_left.set_ylabel(y_left_label)
            if y_right_label and ax_right is not None:
                ax_right.set_ylabel(y_right_label)
            
        except Exception as e:
            print(f"[PlotWizard] Error applying advanced settings: {e}")
            
    def _apply_tick_controls(self, ax_left, ax_right, config):
        """Apply advanced tick controls to axes"""
        try:
            from matplotlib.ticker import MultipleLocator, FixedLocator, MaxNLocator, AutoLocator
            
            # Get tick control settings
            major_tick_spacing = config.get('major_tick_spacing', 'auto')
            major_tick_value = config.get('major_tick_value', 1.0)
            major_tick_count = config.get('major_tick_count', 10)
            minor_ticks_on = config.get('minor_ticks_on', False)
            minor_tick_spacing = config.get('minor_tick_spacing', 0.2)
            
            # Apply major tick settings to X axis
            if major_tick_spacing == 'auto':
                ax_left.xaxis.set_major_locator(AutoLocator())
            elif major_tick_spacing == 'multiple':
                ax_left.xaxis.set_major_locator(MultipleLocator(major_tick_value))
            elif major_tick_spacing == 'fixed':
                # For fixed mode, we'll use the current axis limits and spacing
                xlim = ax_left.get_xlim()
                num_ticks = int((xlim[1] - xlim[0]) / major_tick_value) + 1
                ticks = [xlim[0] + i * major_tick_value for i in range(num_ticks)]
                # Filter ticks within axis range
                ticks = [t for t in ticks if xlim[0] <= t <= xlim[1]]
                ax_left.xaxis.set_major_locator(FixedLocator(ticks))
            elif major_tick_spacing == 'max_n':
                ax_left.xaxis.set_major_locator(MaxNLocator(nbins=major_tick_count))
            
            # Apply major tick settings to Y axis
            if major_tick_spacing == 'auto':
                ax_left.yaxis.set_major_locator(AutoLocator())
            elif major_tick_spacing == 'multiple':
                ax_left.yaxis.set_major_locator(MultipleLocator(major_tick_value))
            elif major_tick_spacing == 'fixed':
                # For fixed mode, use current axis limits and spacing
                ylim = ax_left.get_ylim()
                num_ticks = int((ylim[1] - ylim[0]) / major_tick_value) + 1
                ticks = [ylim[0] + i * major_tick_value for i in range(num_ticks)]
                # Filter ticks within axis range
                ticks = [t for t in ticks if ylim[0] <= t <= ylim[1]]
                ax_left.yaxis.set_major_locator(FixedLocator(ticks))
            elif major_tick_spacing == 'max_n':
                ax_left.yaxis.set_major_locator(MaxNLocator(nbins=major_tick_count))
            
            # Apply same settings to right axis if it exists
            if ax_right is not None:
                if major_tick_spacing == 'auto':
                    ax_right.xaxis.set_major_locator(AutoLocator())
                    ax_right.yaxis.set_major_locator(AutoLocator())
                elif major_tick_spacing == 'multiple':
                    ax_right.xaxis.set_major_locator(MultipleLocator(major_tick_value))
                    ax_right.yaxis.set_major_locator(MultipleLocator(major_tick_value))
                elif major_tick_spacing == 'fixed':
                    # X axis
                    xlim = ax_right.get_xlim()
                    num_ticks = int((xlim[1] - xlim[0]) / major_tick_value) + 1
                    ticks = [xlim[0] + i * major_tick_value for i in range(num_ticks)]
                    ticks = [t for t in ticks if xlim[0] <= t <= xlim[1]]
                    ax_right.xaxis.set_major_locator(FixedLocator(ticks))
                    
                    # Y axis
                    ylim = ax_right.get_ylim()
                    num_ticks = int((ylim[1] - ylim[0]) / major_tick_value) + 1
                    ticks = [ylim[0] + i * major_tick_value for i in range(num_ticks)]
                    ticks = [t for t in ticks if ylim[0] <= t <= ylim[1]]
                    ax_right.yaxis.set_major_locator(FixedLocator(ticks))
                elif major_tick_spacing == 'max_n':
                    ax_right.xaxis.set_major_locator(MaxNLocator(nbins=major_tick_count))
                    ax_right.yaxis.set_major_locator(MaxNLocator(nbins=major_tick_count))
            
            # Apply minor tick settings
            if minor_ticks_on:
                from matplotlib.ticker import MultipleLocator as MinorMultipleLocator
                
                # Apply minor ticks to X axis
                ax_left.xaxis.set_minor_locator(MinorMultipleLocator(minor_tick_spacing))
                ax_left.tick_params(which='minor', length=2, width=0.5)
                
                # Apply minor ticks to Y axis
                ax_left.yaxis.set_minor_locator(MinorMultipleLocator(minor_tick_spacing))
                ax_left.tick_params(which='minor', length=2, width=0.5)
                
                # Apply to right axis if it exists
                if ax_right is not None:
                    ax_right.xaxis.set_minor_locator(MinorMultipleLocator(minor_tick_spacing))
                    ax_right.yaxis.set_minor_locator(MinorMultipleLocator(minor_tick_spacing))
                    ax_right.tick_params(which='minor', length=2, width=0.5)
            else:
                # Turn off minor ticks
                ax_left.minorticks_off()
                if ax_right is not None:
                    ax_right.minorticks_off()
                    
        except Exception as e:
            print(f"[PlotWizard] Error applying tick controls: {e}")
            
    def _update_config_controls(self):
        """Update the configuration controls to match global_config"""
        try:
            # Temporarily disconnect signals to avoid recursion
            self._disconnect_config_signals()
            
            # Axes settings
            self.window.sharex_checkbox.setChecked(self.global_config.get('sharex', False))
            self.window.sharey_checkbox.setChecked(self.global_config.get('sharey', False))
            
            # Tick settings
            self.window.tick_direction_combo.setCurrentText(self.global_config.get('tick_direction', 'in'))
            self.window.tick_width_spinbox.setValue(self.global_config.get('tick_width', 1.0))
            self.window.tick_length_spinbox.setValue(self.global_config.get('tick_length', 4.0))
            
            # Line and axis settings
            self.window.line_width_spinbox.setValue(self.global_config.get('line_width', 2.0))
            self.window.axis_linewidth_spinbox.setValue(self.global_config.get('axis_linewidth', 1.5))
            
            # Font settings
            self.window.font_size_spinbox.setValue(self.global_config.get('font_size', 12))
            self.window.font_weight_combo.setCurrentText(self.global_config.get('font_weight', 'normal'))
            self.window.font_family_combo.setCurrentText(self.global_config.get('font_family', 'sans-serif'))
            
            # Display settings
            self.window.grid_checkbox.setChecked(self.global_config.get('grid', True))
            self.window.box_style_combo.setCurrentText(self.global_config.get('box_style', 'full'))
            
            # Legend settings
            self.window.legend_fontsize_spinbox.setValue(self.global_config.get('legend_fontsize', 10))
            self.window.legend_ncol_spinbox.setValue(self.global_config.get('legend_ncol', 1))
            self.window.legend_frameon_checkbox.setChecked(self.global_config.get('legend_frameon', True))
            self.window.tight_layout_checkbox.setChecked(self.global_config.get('tight_layout', True))
            
            # Reconnect signals
            self._connect_config_controls()
            
        except Exception as e:
            print(f"[PlotWizard] Error updating config controls: {str(e)}")
            
    def _disconnect_config_signals(self):
        """Temporarily disconnect config control signals to avoid recursion"""
        try:
            # Axes settings
            self.window.sharex_checkbox.stateChanged.disconnect()
            self.window.sharey_checkbox.stateChanged.disconnect()
            
            # Tick settings
            self.window.tick_direction_combo.currentTextChanged.disconnect()
            self.window.tick_width_spinbox.valueChanged.disconnect()
            self.window.tick_length_spinbox.valueChanged.disconnect()
            
            # Line and axis settings
            self.window.line_width_spinbox.valueChanged.disconnect()
            self.window.axis_linewidth_spinbox.valueChanged.disconnect()
            
            # Font settings
            self.window.font_size_spinbox.valueChanged.disconnect()
            self.window.font_weight_combo.currentTextChanged.disconnect()
            self.window.font_family_combo.currentTextChanged.disconnect()
            
            # Display settings
            self.window.grid_checkbox.stateChanged.disconnect()
            self.window.box_style_combo.currentTextChanged.disconnect()
            
            # Legend settings
            self.window.legend_fontsize_spinbox.valueChanged.disconnect()
            self.window.legend_ncol_spinbox.valueChanged.disconnect()
            self.window.legend_frameon_checkbox.stateChanged.disconnect()
            self.window.tight_layout_checkbox.stateChanged.disconnect()
            
        except RuntimeError:
            # Some signals may not be connected, which is fine
            pass
            
    def _update_plot(self):
        """Update the plot with current configuration"""
        try:
            if not self.plot_items:
                # Clear plot if no items
                self.window.figure.clear()
                self.window.canvas.draw()
                return
            
            # Apply global figure settings
            self._apply_figure_settings()
            
            # Clear previous plots completely
            self.window.figure.clear()
            
            # Force matplotlib to clear all cached state
            self.window.figure.clf()
            
            # Group items by subplot
            subplot_items = {}
            for item in self.plot_items:
                subplot_num = item['subplot']
                if subplot_num not in subplot_items:
                    subplot_items[subplot_num] = []
                subplot_items[subplot_num].append(item)
            
            # Create subplots
            subplot_nums = sorted(subplot_items.keys())
            n_subplots = len(subplot_nums)
            
            if n_subplots == 0:
                return
            
            # Use user-specified subplot layout
            rows, cols = self.subplot_rows, self.subplot_cols
            
            # Validate that we have enough subplot positions
            if n_subplots > rows * cols:
                print(f"[PlotWizard] Warning: {n_subplots} subplots but only {rows}×{cols} positions available")
                # Auto-expand if needed
                if n_subplots <= 4:
                    rows, cols = 2, 2
                elif n_subplots <= 6:
                    rows, cols = 2, 3
                elif n_subplots <= 9:
                    rows, cols = 3, 3
                else:
                    rows, cols = 4, max(3, (n_subplots + 3) // 4)
                
                # Update the controls to reflect the change
                self.subplot_rows, self.subplot_cols = rows, cols
                self.window.rows_spinbox.setValue(rows)
                self.window.cols_spinbox.setValue(cols)
                print(f"[PlotWizard] Auto-expanded to {rows}×{cols} to fit all subplots")
            
            # Create axes with subplot sharing options
            axes = {}
            sharex = self.global_config.get('sharex', False)
            sharey = self.global_config.get('sharey', False)
            
            # Map subplot numbers to grid positions
            # Sort subplot numbers to ensure consistent positioning
            sorted_subplot_nums = sorted(subplot_nums)
            
            first_ax = None
            for i, subplot_num in enumerate(sorted_subplot_nums):
                # Calculate grid position (1-indexed for matplotlib)
                grid_position = i + 1
                
                # Don't exceed available grid positions
                if grid_position > rows * cols:
                    print(f"[PlotWizard] Warning: Subplot {subplot_num} exceeds grid capacity, skipping")
                    continue
                
                if i == 0:
                    ax = self.window.figure.add_subplot(rows, cols, grid_position)
                    first_ax = ax
                else:
                    share_x = first_ax if sharex else None
                    share_y = first_ax if sharey else None
                    ax = self.window.figure.add_subplot(rows, cols, grid_position, sharex=share_x, sharey=share_y)
                
                axes[subplot_num] = {'left': ax, 'right': None}
                print(f"[PlotWizard] Created subplot {subplot_num} at grid position {grid_position} in {rows}×{cols} layout")
            
            # Store axes for access in advanced settings
            self.current_axes = axes
            
            # Plot each item
            colorbars_added = set()  # Track which subplots already have colorbars
            
            # Only plot subplots that have valid axes
            for subplot_num, items in subplot_items.items():
                if subplot_num not in axes:
                    print(f"[PlotWizard] Skipping subplot {subplot_num} - no axis created")
                    continue
                ax_left = axes[subplot_num]['left']
                ax_right = None
                
                # Check for mixed plot types and inform user
                plot_types = [item['plot_type'] for item in items]
                unique_types = set(plot_types)
                if len(unique_types) > 1:
                    print(f"[PlotWizard] Subplot {subplot_num}: Mixed plot types detected ({', '.join(unique_types)})")
                
                for item in items:
                    try:
                        # Get the axis
                        if item['y_axis'] == 'right':
                            if ax_right is None:
                                ax_right = ax_left.twinx()
                                axes[subplot_num]['right'] = ax_right
                            ax = ax_right
                        else:
                            ax = ax_left
                        
                        # Get channel data
                        channel = item['channel']
                        if not hasattr(channel, 'xdata') or not hasattr(channel, 'ydata'):
                            continue
                            
                        x_data = channel.xdata
                        y_data = channel.ydata
                        
                        if x_data is None or y_data is None:
                            continue
                        
                        # Ensure data is numpy arrays
                        x_data = np.asarray(x_data)
                        y_data = np.asarray(y_data)
                        
                        # Handle data validation differently for spectrograms vs other plot types
                        plot_type = item['plot_type']
                        
                        if plot_type == 'Spectrogram':
                            # For spectrograms, x_data and y_data are axes (different lengths OK)
                            # Just validate each axis independently
                            x_plot = x_data[np.isfinite(x_data)]
                            y_plot = y_data[np.isfinite(y_data)]
                            
                            if len(x_plot) == 0 or len(y_plot) == 0:
                                continue
                                
                            # Use original arrays for plotting (axes don't need pairing)
                            x_plot = x_data
                            y_plot = y_data
                        else:
                            # For other plot types, x_data and y_data are paired coordinates
                            # Ensure they have the same length and filter paired invalid data
                            if len(x_data) != len(y_data):
                                print(f"[PlotWizard] Length mismatch for {item['legend_name']}: x_data={len(x_data)}, y_data={len(y_data)}")
                                continue
                                
                            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                            x_plot = x_data[valid_mask]
                            y_plot = y_data[valid_mask]
                            
                            if len(x_plot) == 0:
                                continue
                        
                        # Plot based on type
                        color = item['color']
                        line_style = item['line_style']
                        marker = item['marker'] if item['marker'] != 'None' else None
                        label = item['legend_name']
                        
                        if plot_type == 'Line':
                            # Use global line width setting
                            default_linewidth = self.global_config.get('line_width', 2.0)
                            linewidth = default_linewidth if len(items) == 1 else max(default_linewidth, 2.0)
                            markersize = 5 if len(items) > 1 else 4
                            ax.plot(x_plot, y_plot, color=color, linestyle=line_style, 
                                   marker=marker, label=label, markersize=markersize, linewidth=linewidth)
                        elif plot_type == 'Scatter':
                            ax.scatter(x_plot, y_plot, color=color, marker=marker or 'o', 
                                     label=label, s=20, alpha=0.7)
                        elif plot_type == 'Bar':
                            # For bar plots, use limited data points to avoid overcrowding
                            if len(x_plot) > 50:
                                indices = np.linspace(0, len(x_plot)-1, 50, dtype=int)
                                x_plot = x_plot[indices]
                                y_plot = y_plot[indices]
                            
                            # Use lower alpha for bars when mixed with other plot types
                            alpha_val = 0.5 if len(items) > 1 else 0.7
                            ax.bar(x_plot, y_plot, color=color, label=label, alpha=alpha_val, 
                                  width=(x_plot[-1] - x_plot[0]) / len(x_plot) * 0.8 if len(x_plot) > 1 else 1)
                        elif plot_type == 'Spectrogram':
                            # Enhanced spectrogram with configurable colormap and clim
                            config = self.subplot_configs.get(subplot_num, {})
                            colormap = config.get('colormap', 'viridis')
                            clim = config.get('clim', [None, None])
                            

                            # Check if this is a real spectrogram channel with metadata
                            if hasattr(channel, 'metadata') and 'Zxx' in channel.metadata:
                                # Real spectrogram data with pre-computed Zxx
                                Zxx = channel.metadata['Zxx']
                                t_axis = x_data  # Time axis
                                f_axis = y_data  # Frequency axis
                                
                                print(f"[PlotWizard] Spectrogram shapes - Zxx: {Zxx.shape}, t_axis: {len(t_axis)}, f_axis: {len(f_axis)}")
                                
                                # Validate dimensions
                                if Zxx.ndim == 2 and len(t_axis) > 1 and len(f_axis) > 1:
                                    # For pcolormesh, we need meshgrid or correct orientation
                                    # Zxx should be (freq, time) for pcolormesh(time, freq, Zxx)
                                    if Zxx.shape == (len(f_axis), len(t_axis)):
                                        # Correct orientation
                                        Zxx_plot = Zxx
                                    elif Zxx.shape == (len(t_axis), len(f_axis)):
                                        # Need to transpose
                                        Zxx_plot = Zxx.T
                                        print(f"[PlotWizard] Transposed Zxx from {Zxx.shape} to {Zxx_plot.shape}")
                                    else:
                                        print(f"[PlotWizard] Dimension mismatch: Zxx {Zxx.shape}, expected ({len(f_axis)}, {len(t_axis)}) or ({len(t_axis)}, {len(f_axis)})")
                                        continue
                                    
                                    # Convert to dB
                                    Zxx_db = 10 * np.log10(Zxx_plot + 1e-10)
                                    
                                    # Apply color limits
                                    if clim[0] is not None and clim[1] is not None:
                                        vmin, vmax = clim[0], clim[1]
                                    elif clim[0] is not None:
                                        vmin, vmax = clim[0], np.max(Zxx_db)
                                    elif clim[1] is not None:
                                        vmin, vmax = np.min(Zxx_db), clim[1]
                                    else:
                                        vmin, vmax = None, None
                                    
                                    # Create the plot with enhanced transparency for overlaying
                                    alpha_val = 0.7 if len(items) > 1 else 0.9
                                    
                                    # Use pcolormesh with proper axis order: (time, freq, power)
                                    im = ax.pcolormesh(t_axis, f_axis, Zxx_db, cmap=colormap, 
                                                     shading='gouraud', alpha=alpha_val, vmin=vmin, vmax=vmax)
                                else:
                                    print(f"[PlotWizard] Invalid spectrogram data - Zxx shape: {Zxx.shape if hasattr(Zxx, 'shape') else 'unknown'}, t_axis: {len(t_axis)}, f_axis: {len(f_axis)}")
                                    continue
                                    
                            elif len(y_plot) > 100:
                                # Generate spectrogram from 1D signal
                                from scipy import signal
                                f, t, Sxx = signal.spectrogram(y_plot, fs=1.0)
                                
                                # Create spectrogram plot
                                Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
                                
                                # Apply color limits if specified
                                if clim[0] is not None and clim[1] is not None:
                                    vmin, vmax = clim[0], clim[1]
                                elif clim[0] is not None:
                                    vmin, vmax = clim[0], np.max(Sxx_db)
                                elif clim[1] is not None:
                                    vmin, vmax = np.min(Sxx_db), clim[1]
                                else:
                                    vmin, vmax = None, None
                                
                                # Create the plot with enhanced transparency for overlaying
                                alpha_val = 0.7 if len(items) > 1 else 0.9
                                im = ax.pcolormesh(t, f, Sxx_db, cmap=colormap, shading='gouraud', 
                                                 alpha=alpha_val, vmin=vmin, vmax=vmax)
                            else:
                                print(f"[PlotWizard] Spectrogram requires at least 100 data points, got {len(y_plot)}")
                                continue
                            
                            # Only add colorbar if this subplot doesn't have one already and colorbars are enabled
                            colorbar_key = (subplot_num, item['y_axis'])
                            show_colorbar = config.get('show_colorbar', True)
                            if colorbar_key not in colorbars_added and show_colorbar:
                                try:
                                    # Get colorbar label and position from config
                                    colorbar_label = config.get('colorbar_label', f'Power (dB) - {label}')
                                    colorbar_position = config.get('colorbar_position', 'bottom')
                                    
                                    # Create colorbar with specified position and proper spacing
                                    if colorbar_position in ['bottom', 'top']:
                                        orientation = 'horizontal'
                                        pad = 0.15 if colorbar_position == 'bottom' else 0.08
                                        shrink = 0.8  # Make colorbar smaller horizontally
                                        aspect = 30   # Control thickness
                                    else:  # 'left' or 'right'
                                        orientation = 'vertical'
                                        pad = 0.05
                                        shrink = 0.8  # Make colorbar smaller vertically
                                        aspect = 20   # Control thickness
                                    
                                    # Create colorbar with proper positioning and sizing
                                    if colorbar_position == 'left':
                                        # For left position, create on the left side
                                        cbar = self.window.figure.colorbar(im, ax=ax, label=colorbar_label,
                                                                         orientation=orientation, pad=pad, 
                                                                         location='left', shrink=shrink, aspect=aspect)
                                    elif colorbar_position == 'right':
                                        # For right position (default matplotlib behavior)
                                        cbar = self.window.figure.colorbar(im, ax=ax, label=colorbar_label,
                                                                         orientation=orientation, pad=pad,
                                                                         shrink=shrink, aspect=aspect)
                                    elif colorbar_position == 'top':
                                        # For top position
                                        cbar = self.window.figure.colorbar(im, ax=ax, label=colorbar_label,
                                                                         orientation=orientation, pad=pad,
                                                                         location='top', shrink=shrink, aspect=aspect)
                                    else:  # 'bottom' (default)
                                        # For bottom position
                                        cbar = self.window.figure.colorbar(im, ax=ax, label=colorbar_label,
                                                                         orientation=orientation, pad=pad,
                                                                         location='bottom', shrink=shrink, aspect=aspect)
                                    
                                    colorbars_added.add(colorbar_key)
                                    print(f"[PlotWizard] Added colorbar at {colorbar_position} for subplot {subplot_num}")
                                except Exception as cb_error:
                                    print(f"[PlotWizard] Colorbar warning: {cb_error}")
                            
                            # Don't override axis labels here - let subplot config handle them
                            # Only set if this is the only item in the subplot
                            if len(items) == 1:
                                ax.set_ylabel('Frequency (Hz)')
                                ax.set_xlabel('Time (s)')
                        
                    except Exception as e:
                        print(f"[PlotWizard] Error plotting item {item['legend_name']}: {str(e)}")
                        continue
                
                # Configure subplot
                if subplot_num in self.subplot_configs:
                    config = self.subplot_configs[subplot_num]
                    
                    # Apply advanced settings first
                    self._apply_advanced_settings(ax_left, ax_right, config)
                    
                    # Check if this subplot contains spectrograms
                    has_spectrogram = any(item['plot_type'] == 'Spectrogram' for item in items)
                    has_other_types = any(item['plot_type'] != 'Spectrogram' for item in items)
                    
                    # Set labels - handle mixed plot types intelligently
                    xlabel = config['xlabel']
                    ylabel = config['ylabel']
                    
                    # If mixing spectrogram with other types, use more generic labels
                    if has_spectrogram and has_other_types:
                        if ylabel == 'Amplitude':  # Default ylabel
                            ylabel = 'Frequency / Amplitude'
                        if xlabel == 'Time':  # Default xlabel
                            xlabel = 'Time'
                    elif has_spectrogram and not has_other_types:
                        # Pure spectrogram subplot
                        if ylabel == 'Amplitude':  # Default ylabel
                            ylabel = 'Frequency (Hz)'
                        if xlabel == 'Time':  # Default xlabel
                            xlabel = 'Time (s)'
                    
                    # Override with advanced Y-axis labels if set
                    if config.get('y_left_label'):
                        ylabel = config['y_left_label']
                    
                    ax_left.set_xlabel(xlabel)
                    ax_left.set_ylabel(ylabel)
                    
                    # Set right Y-axis label if specified and right axis exists
                    if ax_right is not None and config.get('y_right_label'):
                        ax_right.set_ylabel(config['y_right_label'])
                    
                    # Apply axis styling
                    self._apply_axis_styling(ax_left)
                    if ax_right is not None:
                        self._apply_axis_styling(ax_right)
                    
                    # Configure legend
                    if config['show_legend']:
                        # Combine legends from both axes
                        lines_left, labels_left = ax_left.get_legend_handles_labels()
                        lines_right, labels_right = [], []
                        
                        if ax_right is not None:
                            lines_right, labels_right = ax_right.get_legend_handles_labels()
                        
                        if lines_left or lines_right:
                            legend_props = {
                                'loc': config['legend_position'],
                                'fontsize': self.global_config.get('legend_fontsize', 10),
                                'ncol': self.global_config.get('legend_ncol', 1),
                                'frameon': self.global_config.get('legend_frameon', True)
                            }
                            ax_left.legend(lines_left + lines_right, labels_left + labels_right, **legend_props)
                    
                    # Grid
                    if self.global_config.get('grid', True):
                        ax_left.grid(True, alpha=0.3)
            
            # Apply global configuration with colorbar considerations
            if self.global_config.get('tight_layout', True):
                try:
                    # Check if we have any colorbars that need extra space
                    has_bottom_colorbar = False
                    has_top_colorbar = False
                    has_side_colorbar = False
                    
                    for subplot_num in self.subplot_configs:
                        if subplot_num in [item['subplot'] for item in self.plot_items]:
                            config = self.subplot_configs[subplot_num]
                            subplot_items = [item for item in self.plot_items if item['subplot'] == subplot_num]
                            has_spectrogram = any(item['plot_type'] == 'Spectrogram' for item in subplot_items)
                            if has_spectrogram:
                                colorbar_pos = config.get('colorbar_position', 'bottom')
                                if colorbar_pos == 'bottom':
                                    has_bottom_colorbar = True
                                elif colorbar_pos == 'top':
                                    has_top_colorbar = True
                                else:
                                    has_side_colorbar = True
                    
                    # Adjust layout based on colorbar positions
                    if has_bottom_colorbar or has_top_colorbar or has_side_colorbar:
                        # Use subplots_adjust for better control when colorbars are present
                        bottom = 0.15 if has_bottom_colorbar else 0.1
                        top = 0.85 if has_top_colorbar else 0.95
                        left = 0.15 if has_side_colorbar else 0.1
                        right = 0.85 if has_side_colorbar else 0.9
                        
                        self.window.figure.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=0.3, wspace=0.3)
                    else:
                        # Use tight_layout for regular plots
                        self.window.figure.tight_layout()
                except Exception as e:
                    print(f"[PlotWizard] Layout adjustment warning: {e}")
                    pass  # Ignore layout errors
            
            # Update canvas
            self.window.canvas.draw()
            
        except Exception as e:
            print(f"[PlotWizard] Error updating plot: {str(e)}")
            traceback.print_exc()
            
    def show(self):
        """Show the plot wizard window"""
        self.window.show()
        
    def close(self):
        """Close the plot wizard"""
        if self.window:
            self.window.close()
            
    def save_configuration(self, filepath=None):
        """Save current plot configuration to file"""
        try:
            if not filepath:
                filepath, _ = QFileDialog.getSaveFileName(
                    self.window,
                    "Save Plot Configuration",
                    "",
                    "JSON Files (*.json);;All Files (*.*)"
                )
                
            if not filepath:
                return False
                
            config = {
                'plot_items': self.plot_items,
                'subplot_configs': self.subplot_configs,
                'global_config': self.global_config,
                'subplot_dimensions': {
                    'rows': self.subplot_rows,
                    'cols': self.subplot_cols
                }
            }
            
            # Custom JSON encoder to handle tuples properly
            def json_encoder(obj):
                if isinstance(obj, tuple):
                    return {'__tuple__': True, 'items': list(obj)}
                return str(obj)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, default=json_encoder)
                
            QMessageBox.information(self.window, "Success", 
                                  f"Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"[PlotWizard] Error saving configuration: {str(e)}")
            QMessageBox.critical(self.window, "Error", 
                               f"Failed to save configuration: {str(e)}")
            return False
            
    def load_configuration(self, filepath=None):
        """Load plot configuration from file"""
        try:
            if not filepath:
                filepath, _ = QFileDialog.getOpenFileName(
                    self.window,
                    "Load Plot Configuration",
                    "",
                    "JSON Files (*.json);;All Files (*.*)"
                )
                
            if not filepath or not Path(filepath).exists():
                return False
                
            # Custom JSON decoder to handle tuples
            def json_decoder(dct):
                if isinstance(dct, dict) and dct.get('__tuple__'):
                    return tuple(dct['items'])
                return dct
            
            with open(filepath, 'r') as f:
                config = json.load(f, object_hook=json_decoder)
                
            # Restore configuration
            self.plot_items = config.get('plot_items', [])
            self.subplot_configs = config.get('subplot_configs', {})
            self.global_config = config.get('global_config', {})
            
            # Restore subplot dimensions
            dimensions = config.get('subplot_dimensions', {'rows': 1, 'cols': 1})
            self.subplot_rows = dimensions.get('rows', 1)
            self.subplot_cols = dimensions.get('cols', 1)
            
            # Convert string keys back to integers for subplot configs
            self.subplot_configs = {int(k): v for k, v in self.subplot_configs.items()}
            
            # Update spinboxes
            self.window.rows_spinbox.setValue(self.subplot_rows)
            self.window.cols_spinbox.setValue(self.subplot_cols)
            
            # Update tables and plot
            self._update_line_config_table()
            self._update_subplot_config_table()
            self._update_config_controls()
            self._update_plot()
            
            QMessageBox.information(self.window, "Success", 
                                  f"Configuration loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"[PlotWizard] Error loading configuration: {str(e)}")
            QMessageBox.critical(self.window, "Error", 
                               f"Failed to load configuration: {str(e)}")
            return False 