from PySide6.QtCore import QObject, Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QMessageBox, QTableWidgetItem, QColorDialog, QComboBox, QCheckBox, 
    QSpinBox, QFileDialog, QInputDialog, QHeaderView, QDialog, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox, QDoubleSpinBox,
    QWidget
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
        
        # Connect signals and populate dropdowns
        self._connect_signals()
        self._connect_config_controls()
        self._populate_dropdowns()
        
        # Update config controls to match initial state
        self._update_config_controls()

    def _setup_line_config_table(self):
        """Setup the line configuration table"""
        table = self.window.line_config_table
        
        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Subplot#
        header.setSectionResizeMode(1, QHeaderView.Stretch)          # Channel Name - stretches
        header.setSectionResizeMode(2, QHeaderView.Fixed)            # Actions - fixed width
        
        # Set minimum width for actions column to prevent squishing
        header.setMinimumSectionSize(80)
        
        # Set initial width for actions column
        header.resizeSection(2, 100)  # Give actions column enough space for 3 buttons
        
    def _setup_subplot_config_table(self):
        """Setup the subplot configuration table"""
        table = self.window.subplot_config_table
        
        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Subplot#
        header.setSectionResizeMode(1, QHeaderView.Stretch)          # Subplot Name - stretches
        header.setSectionResizeMode(2, QHeaderView.Fixed)            # Actions - fixed width
        
        # Set minimum width for actions column to prevent squishing
        header.setMinimumSectionSize(80)
        
        # Set initial width for actions column
        header.resizeSection(2, 100)  # Give actions column enough space for 2 buttons
        
    def _connect_signals(self):
        """Connect signals to their respective slots"""
        try:
            # Disconnect any existing connections to prevent duplicates
            try:
                self.window.add_btn.clicked.disconnect()
                self.window.file_dropdown.currentTextChanged.disconnect()
                self.window.channel_dropdown.currentTextChanged.disconnect()
                self.window.rows_spinbox.valueChanged.disconnect()
                self.window.cols_spinbox.valueChanged.disconnect()
            except:
                pass  # Ignore if no connections exist
            
            # Connect signals
            self.window.add_btn.clicked.connect(self._on_add_to_plot)
            self.window.file_dropdown.currentTextChanged.connect(self._on_file_changed)
            self.window.channel_dropdown.currentTextChanged.connect(self._on_channel_changed)
            
            # Dimension controls
            self.window.rows_spinbox.valueChanged.connect(self._on_dimension_changed)
            self.window.cols_spinbox.valueChanged.connect(self._on_dimension_changed)
            
        except Exception as e:
            print(f"[PlotWizard] Error connecting signals: {e}")

    def _on_plot_type_changed(self, plot_type):
        """Handle plot type selection change"""
        try:
            print(f"[PlotWizard] Plot type changed to: {plot_type}")
        except Exception as e:
            print(f"[PlotWizard] Error handling plot type change: {e}")

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
                print(f"[PlotWizard] File changed to: {filename}, found {len(channel_names)} channels")
                
        except Exception as e:
            print(f"[PlotWizard] Error updating channels: {str(e)}")
            

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
                # Determine plot type based on channel (for internal use)
                plot_type = self._determine_plot_type_for_channel(channel)
                print(f"[PlotWizard] Channel changed to: {channel_name}, detected plot type: {plot_type}")
                
        except Exception as e:
            print(f"[PlotWizard] Error handling channel change: {e}")

    def _determine_plot_type_for_channel(self, channel):
        """Determine the appropriate plot type for a channel"""
        try:
            # Check channel tags to determine plot type
            if hasattr(channel, 'tags') and channel.tags:
                if 'spectrogram' in channel.tags:
                    return 'Spectrogram'
                elif 'scatter' in channel.tags:
                    return 'Scatter'
                elif 'bar' in channel.tags:
                    return 'Bar'
                else:
                    return 'Line'  # Default for time-series
            else:
                return 'Line'  # Default fallback
        except Exception as e:
            print(f"[PlotWizard] Error determining plot type: {e}")
            return 'Line'

    def _on_add_to_plot(self):
        """Handle adding a channel to the plot"""
        try:
            # Get current selections
            filename = self.window.file_dropdown.currentText()
            channel_name = self.window.channel_dropdown.currentText()
            
            if not all([filename, channel_name]):
                print("[PlotWizard] Missing file or channel selection")
                return
            
            if filename == "No parsed files available":
                print("[PlotWizard] No parsed files available")
                return
            
            # Get the channel object
            channel = self._get_channel(filename, channel_name)
            if not channel:
                print(f"[PlotWizard] Could not find channel: {channel_name}")
                return
            
            # Determine plot type automatically
            plot_type = self._determine_plot_type_for_channel(channel)
            
            # Generate legend name from channel name
            legend_name = channel_name or channel.legend_label or channel.channel_id
            
            print(f"[PlotWizard] Adding to plot: {filename} - {channel_name} ({plot_type})")
            
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
                'visible': True,
                'size': self._get_channel_size(channel)  # Add size information
            }
            
            print(f"[PlotWizard] Added {plot_type} to subplot {subplot_num} on {y_axis} axis")
            
            # Add to plot items
            self.plot_items.append(plot_item)
            
            # Ensure subplot configuration exists
            if subplot_num not in self.subplot_configs:
                self._create_subplot_config(subplot_num)
            
            # Auto-update subplot dimensions
            self._auto_update_dimensions()
            
            # Update tables and plot
            self._update_line_config_table()
            self._update_subplot_config_table()
            self._update_plot()
            
        except Exception as e:
            print(f"[PlotWizard] Error adding to plot: {str(e)}")
            traceback.print_exc()

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
            'subplot_name': f"Subplot {subplot_num}",  # Add subplot name
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
        
    def _create_line_button_slot(self, method, index):
        """Create a properly bound slot for line action buttons"""
        return lambda checked=False: method(index)
    
    def _create_spinbox_slot(self, index):
        """Create a properly bound slot for spinbox value changes"""
        return lambda v: self._on_subplot_changed(index, v)
    
    def _update_line_config_table(self):
        """Update the line configuration table"""
        try:
            table = self.window.line_config_table
            
            # Temporarily disconnect signals to prevent recursive calls
            table.blockSignals(True)
            
            # Store current column widths before clearing
            header = table.horizontalHeader()
            current_widths = []
            for i in range(table.columnCount()):
                current_widths.append(header.sectionSize(i))
            
            # Clear existing contents to avoid widget reuse issues
            table.clearContents()
            table.setRowCount(len(self.plot_items))
            
            for i, item in enumerate(self.plot_items):
                # Subplot number (editable)
                subplot_spinbox = QSpinBox()
                subplot_spinbox.setMinimum(1)
                subplot_spinbox.setMaximum(10)
                subplot_spinbox.setValue(item['subplot'])
                subplot_spinbox.valueChanged.connect(self._create_spinbox_slot(i))
                table.setCellWidget(i, 0, subplot_spinbox)
                
                # Channel Name (display only)
                channel_name = item.get('channel_name', item.get('legend_name', 'Unknown'))
                channel_item = QTableWidgetItem(str(channel_name))
                channel_item.setFlags(channel_item.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, 1, channel_item)
                
                # Actions (buttons)
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(2, 2, 2, 2)
                actions_layout.setSpacing(2)
                
                # Info button - match main window icon
                info_button = QPushButton("❗")
                info_button.setMaximumWidth(25)
                info_button.setMaximumHeight(25)
                info_button.setToolTip("Channel information and metadata")
                info_button.clicked.connect(self._create_line_button_slot(self._on_line_info_clicked, i))
                actions_layout.addWidget(info_button)
                
                # Paint button - match main window icon
                paint_button = QPushButton("🎨")
                paint_button.setMaximumWidth(25)
                paint_button.setMaximumHeight(25)
                paint_button.setToolTip("Edit line style")
                paint_button.clicked.connect(self._create_line_button_slot(self._on_line_paint_clicked, i))
                actions_layout.addWidget(paint_button)
                
                # Delete button - match main window icon
                delete_button = QPushButton("🗑️")
                delete_button.setMaximumWidth(25)
                delete_button.setMaximumHeight(25)
                delete_button.setToolTip("Remove from plot")
                delete_button.clicked.connect(self._create_line_button_slot(self._on_line_delete_clicked, i))
                actions_layout.addWidget(delete_button)
                
                table.setCellWidget(i, 2, actions_widget)
            
            # Restore column widths and ensure proper sizing
            for i, width in enumerate(current_widths):
                if i < table.columnCount():
                    header.setSectionResizeMode(i, QHeaderView.Interactive)
                    header.resizeSection(i, max(width, 80))  # Ensure minimum width
            
            # Re-apply column resize modes
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Subplot#
            header.setSectionResizeMode(1, QHeaderView.Stretch)          # Channel Name - stretches
            header.setSectionResizeMode(2, QHeaderView.Fixed)            # Actions - fixed width
            
            # Set minimum width for actions column to prevent squishing
            header.setMinimumSectionSize(80)
            
            # Re-enable signals
            table.blockSignals(False)
            
        except Exception as e:
            print(f"[PlotWizard] Error updating line config table: {e}")
            traceback.print_exc()
    
    def _create_subplot_button_slot(self, method, subplot_num):
        """Create a properly bound slot for subplot action buttons"""
        return lambda checked=False: method(subplot_num)
    
    def _update_subplot_config_table(self):
        """Update the subplot configuration table"""
        try:
            table = self.window.subplot_config_table
            
            # Temporarily disconnect signals to prevent recursive calls
            table.blockSignals(True)
            
            # Store current column widths before clearing
            header = table.horizontalHeader()
            current_widths = []
            for i in range(table.columnCount()):
                current_widths.append(header.sectionSize(i))
            
            # Get unique subplot numbers from plot items
            subplot_nums = sorted(set(item['subplot'] for item in self.plot_items))
            
            # Clear existing contents
            table.clearContents()
            table.setRowCount(len(subplot_nums))
            
            for i, subplot_num in enumerate(subplot_nums):
                # Ensure subplot config exists
                if subplot_num not in self.subplot_configs:
                    self._create_subplot_config(subplot_num)
                
                config = self.subplot_configs[subplot_num]
                
                # Subplot number (display only)
                subplot_item = QTableWidgetItem(str(subplot_num))
                subplot_item.setFlags(subplot_item.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, 0, subplot_item)
                
                # Subplot name (editable)
                subplot_name = config.get('subplot_name', f"Subplot {subplot_num}")
                name_item = QTableWidgetItem(str(subplot_name))
                table.setItem(i, 1, name_item)
                
                # Actions (buttons)
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(2, 2, 2, 2)
                actions_layout.setSpacing(2)
                
                # Paint button - match main window icon
                paint_button = QPushButton("🎨")
                paint_button.setMaximumWidth(25)
                paint_button.setMaximumHeight(25)
                paint_button.setToolTip("Edit subplot style")
                paint_button.clicked.connect(self._create_subplot_button_slot(self._on_subplot_paint_clicked, subplot_num))
                actions_layout.addWidget(paint_button)
                
                # Delete button - match main window icon
                delete_button = QPushButton("🗑️")
                delete_button.setMaximumWidth(25)
                delete_button.setMaximumHeight(25)
                delete_button.setToolTip("Remove subplot")
                delete_button.clicked.connect(self._create_subplot_button_slot(self._on_subplot_delete_clicked, subplot_num))
                actions_layout.addWidget(delete_button)
                
                table.setCellWidget(i, 2, actions_widget)
            
            # Restore column widths and ensure proper sizing
            for i, width in enumerate(current_widths):
                if i < table.columnCount():
                    header.setSectionResizeMode(i, QHeaderView.Interactive)
                    header.resizeSection(i, max(width, 80))  # Ensure minimum width
            
            # Re-apply column resize modes
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Subplot#
            header.setSectionResizeMode(1, QHeaderView.Stretch)          # Subplot Name - stretches
            header.setSectionResizeMode(2, QHeaderView.Fixed)            # Actions - fixed width
            
            # Set minimum width for actions column to prevent squishing
            header.setMinimumSectionSize(80)
            
            # Re-enable signals
            table.blockSignals(False)
            
            # Auto-update dimensions based on number of subplots
            self._auto_update_dimensions()
            
        except Exception as e:
            print(f"[PlotWizard] Error updating subplot config table: {e}")
            traceback.print_exc()
    
    def _on_line_delete_clicked(self, item_index):
        """Handle line delete button click"""
        try:
            if 0 <= item_index < len(self.plot_items):
                item = self.plot_items[item_index]
                reply = QMessageBox.question(
                    self.window, 
                    "Confirm Delete", 
                    f"Remove '{item['legend_name']}' from plot?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Remove the item
                    del self.plot_items[item_index]
                    
                    # Update tables and plot
                    self._update_line_config_table()
                    self._update_subplot_config_table()
                    self._update_plot()
                    
                    # Auto-update dimensions after line deletion
                    self._auto_update_dimensions()
                    
                    print(f"[PlotWizard] Removed item {item_index} from plot")
                    
        except Exception as e:
            print(f"[PlotWizard] Error deleting line: {e}")

    # New action handler methods for the updated table structure
    def _on_line_info_clicked(self, item_index):
        """Handle line info button click - opens metadata wizard for the channel"""
        try:
            if 0 <= item_index < len(self.plot_items):
                item = self.plot_items[item_index]
                channel = item.get('channel')
                
                if channel:
                    # Open the comprehensive metadata wizard for this channel
                    from metadata_wizard import MetadataWizard
                    wizard = MetadataWizard(channel, self.window, self.file_manager)
                    wizard.exec()
                else:
                    # Fallback: show basic info if channel object is not available
                    channel_name = item.get('channel_name', item['legend_name'])
                    plot_type = item['plot_type']
                    subplot_num = item['subplot']
                    
                    info_text = f"""
                    Channel: {channel_name}
                    Plot Type: {plot_type}
                    Subplot: {subplot_num}
                    Legend: {item['legend_name']}
                    Color: {item['color']}
                    Line Style: {item['line_style']}
                    Marker: {item['marker']}
                    Y-Axis: {item['y_axis']}
                    """
                    
                    from PySide6.QtWidgets import QMessageBox
                    msg = QMessageBox(self.window)
                    msg.setWindowTitle("Line Information")
                    msg.setText(info_text)
                    msg.exec()
        except Exception as e:
            print(f"[PlotWizard] Error showing line info: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_line_paint_clicked(self, item_index):
        """Handle line paint button click - opens appropriate wizard based on plot type"""
        try:
            if 0 <= item_index < len(self.plot_items):
                item = self.plot_items[item_index]
                plot_type = item['plot_type']
                
                print(f"[PlotWizard] Paint clicked for {item['legend_name']} (type: {plot_type})")
                
                # Try to get the actual channel object if available
                channel = item.get('channel')
                
                if plot_type == 'Spectrogram' and channel:
                    # Open spectrogram wizard for spectrogram plots
                    try:
                        from spectrogram_wizard import SpectrogramWizard
                        wizard = SpectrogramWizard(channel, self.window)
                        wizard.spectrogram_updated.connect(lambda ch_id: self._on_spectrogram_updated_from_wizard(item_index))
                        wizard.exec()
                        return
                    except ImportError:
                        print("[PlotWizard] SpectrogramWizard not available")
                elif plot_type == 'Scatter' and channel:
                    # Open marker wizard for scatter plots
                    try:
                        from marker_wizard import MarkerWizard
                        # Create a pair config dict that matches MarkerWizard's expected format
                        pair_config = {
                            'name': item['legend_name'],
                            'ref_channel': item['channel_name'],
                            'test_channel': item['channel_name'],
                            'marker_type': '○ Circle',
                            'marker_color': item['color'],
                            'marker_size': 50,
                            'marker_alpha': 0.7,
                            'edge_color': '#000000',
                            'edge_width': 0.0,
                            'fill_style': 'full',
                            'z_order': 0
                        }
                        wizard = MarkerWizard(pair_config, self.window)
                        # Connect the marker_updated signal to handle Apply button
                        wizard.marker_updated.connect(lambda config: self._on_marker_updated_from_wizard(item_index, config))
                        if wizard.exec():
                            # Update item properties from wizard results
                            self._on_marker_updated_from_wizard(item_index, wizard.pair_config)
                        return
                    except ImportError:
                        print("[PlotWizard] MarkerWizard not available")
                elif channel and hasattr(channel, 'channel_id'):
                    # Open line wizard for line plots
                    try:
                        from line_wizard import LineWizard
                        wizard = LineWizard(channel, self.window)
                        wizard.channel_updated.connect(lambda ch_id: self._on_channel_updated_from_wizard(item_index))
                        wizard.exec()
                        return
                    except ImportError:
                        print("[PlotWizard] LineWizard not available")
                
                # Fallback: Show property editing dialog
                self._show_line_properties_dialog(item_index)
                
        except Exception as e:
            print(f"[PlotWizard] Error opening line paint dialog: {str(e)}")
    
    def _on_channel_updated_from_wizard(self, item_index):
        """Handle channel updates from line wizard"""
        try:
            if 0 <= item_index < len(self.plot_items):
                item = self.plot_items[item_index]
                channel = item.get('channel')
                if channel:
                    # Update item properties from channel
                    item['color'] = getattr(channel, 'color', item['color'])
                    item['line_style'] = getattr(channel, 'style', item['line_style'])
                    item['marker'] = getattr(channel, 'marker', item['marker'])
                    item['legend_name'] = getattr(channel, 'legend_label', item['legend_name'])
                    
                    # Map channel y-axis to plot item y-axis
                    channel_yaxis = getattr(channel, 'yaxis', 'y-left')
                    item['y_axis'] = 'right' if channel_yaxis == 'y-right' else 'left'
                    
                    # Handle z-order
                    item['z_order'] = getattr(channel, 'z_order', 0)
                    
                    print(f"[PlotWizard] Updated channel properties - Y-axis: {item['y_axis']}")
                    
                    # Update tables and plot
                    self._update_line_config_table()
                    self._update_plot()
                    
                    # Ensure proper column widths after update
                    self._ensure_table_column_widths()
        except Exception as e:
            print(f"[PlotWizard] Error updating from channel wizard: {str(e)}")
    
    def _on_marker_updated_from_wizard(self, item_index, marker_config):
        """Handle marker updates from marker wizard"""
        try:
            if 0 <= item_index < len(self.plot_items):
                item = self.plot_items[item_index]
                
                # Update item properties from marker config
                item['color'] = marker_config.get('marker_color_hex', marker_config.get('marker_color', item['color']))
                item['marker'] = marker_config.get('marker_symbol', item['marker'])
                item['legend_name'] = marker_config.get('name', item['legend_name'])
                
                # Store additional marker properties for scatter plots
                item['marker_size'] = marker_config.get('marker_size', 50)
                item['marker_alpha'] = marker_config.get('marker_alpha', 0.7)
                item['edge_color'] = marker_config.get('edge_color', '#000000')
                item['edge_width'] = marker_config.get('edge_width', 0.0)
                item['z_order'] = marker_config.get('z_order', 0)
                
                # Store x-axis preference for marker plots
                x_axis = marker_config.get('x_axis', 'bottom')
                item['x_axis'] = f"x-{x_axis}"  # Convert to channel format
                
                print(f"[PlotWizard] Updated marker properties for {item['legend_name']}, x-axis: {item['x_axis']}")
                
                # Update tables and plot
                self._update_line_config_table()
                self._update_plot()
                
                # Ensure proper column widths after update
                self._ensure_table_column_widths()
        except Exception as e:
            print(f"[PlotWizard] Error updating from marker wizard: {str(e)}")
    
    def _on_spectrogram_updated_from_wizard(self, item_index):
        """Handle spectrogram updates from spectrogram wizard"""
        try:
            if 0 <= item_index < len(self.plot_items):
                item = self.plot_items[item_index]
                channel = item.get('channel')
                if channel:
                    # Update item properties from channel
                    item['legend_name'] = getattr(channel, 'legend_label', item['legend_name'])
                    
                    # Update subplot config with spectrogram-specific settings
                    subplot_num = item['subplot']
                    if subplot_num in self.subplot_configs:
                        config = self.subplot_configs[subplot_num]
                        
                        # Update colorbar settings
                        config['show_colorbar'] = getattr(channel, 'show_colorbar', True)
                        config['colorbar_position'] = getattr(channel, 'colorbar_position', 'right')
                        config['colorbar_label'] = getattr(channel, 'colorbar_label', 'Power (dB)')
                        
                        # Update colormap and color limits
                        config['colormap'] = getattr(channel, 'colormap', 'viridis')
                        config['clim'] = [
                            getattr(channel, 'clim_min', None),
                            getattr(channel, 'clim_max', None)
                        ]
                        
                        # Update axis scales
                        config['y_scale'] = getattr(channel, 'freq_scale', 'linear')
                        config['x_scale'] = getattr(channel, 'time_scale', 'linear')
                        
                        # Update axis limits
                        freq_limits = getattr(channel, 'freq_limits', [None, None])
                        time_limits = getattr(channel, 'time_limits', [None, None])
                        config['ylim'] = freq_limits
                        config['xlim'] = time_limits
                        
                        print(f"[PlotWizard] Updated spectrogram properties for {item['legend_name']}")
                        print(f"[PlotWizard] Colormap: {config['colormap']}, Colorbar: {config['show_colorbar']}")
                    
                    # Update tables and plot
                    self._update_line_config_table()
                    self._update_subplot_config_table()
                    self._update_plot()
                    
                    # Ensure proper column widths after update
                    self._ensure_table_column_widths()
        except Exception as e:
            print(f"[PlotWizard] Error updating from spectrogram wizard: {str(e)}")
    
    def _show_line_properties_dialog(self, item_index):
        """Show a simple properties dialog as fallback"""
        try:
            if 0 <= item_index < len(self.plot_items):
                item = self.plot_items[item_index]
                
                from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QColorDialog, QDialogButtonBox
                from PySide6.QtGui import QColor
                
                dialog = QDialog(self.window)
                dialog.setWindowTitle(f"Line Properties - {item['legend_name']}")
                dialog.setModal(True)
                dialog.setMinimumSize(300, 200)
                
                layout = QVBoxLayout(dialog)
                
                # Legend name
                legend_layout = QHBoxLayout()
                legend_layout.addWidget(QLabel("Legend:"))
                legend_edit = QLineEdit(item['legend_name'])
                legend_layout.addWidget(legend_edit)
                layout.addLayout(legend_layout)
                
                # Color
                color_layout = QHBoxLayout()
                color_layout.addWidget(QLabel("Color:"))
                color_button = QPushButton()
                color_button.setStyleSheet(f"background-color: {item['color']}; min-height: 30px;")
                current_color = item['color']
                
                def choose_color():
                    nonlocal current_color
                    color = QColorDialog.getColor(QColor(current_color), dialog)
                    if color.isValid():
                        current_color = color.name()
                        color_button.setStyleSheet(f"background-color: {current_color}; min-height: 30px;")
                
                color_button.clicked.connect(choose_color)
                color_layout.addWidget(color_button)
                layout.addLayout(color_layout)
                
                # Line style
                style_layout = QHBoxLayout()
                style_layout.addWidget(QLabel("Line Style:"))
                style_combo = QComboBox()
                styles = [('-', 'Solid'), ('--', 'Dashed'), ('-.', 'Dash-dot'), (':', 'Dotted'), ('None', 'None')]
                for style_val, style_name in styles:
                    style_combo.addItem(style_name, style_val)
                    if style_val == item['line_style']:
                        style_combo.setCurrentText(style_name)
                style_layout.addWidget(style_combo)
                layout.addLayout(style_layout)
                
                # Marker
                marker_layout = QHBoxLayout()
                marker_layout.addWidget(QLabel("Marker:"))
                marker_combo = QComboBox()
                markers = [('None', 'None'), ('o', 'Circle'), ('s', 'Square'), ('^', 'Triangle'), ('D', 'Diamond'), ('+', 'Plus'), ('x', 'X')]
                for marker_val, marker_name in markers:
                    marker_combo.addItem(marker_name, marker_val)
                    if marker_val == item['marker']:
                        marker_combo.setCurrentText(marker_name)
                marker_layout.addWidget(marker_combo)
                layout.addLayout(marker_layout)
                
                # Y-axis selection
                y_axis_layout = QHBoxLayout()
                y_axis_layout.addWidget(QLabel("Y-Axis:"))
                y_axis_combo = QComboBox()
                y_axis_combo.addItem("Left", "left")
                y_axis_combo.addItem("Right", "right")
                current_y_axis = item.get('y_axis', 'left')
                if current_y_axis == 'right':
                    y_axis_combo.setCurrentIndex(1)
                else:
                    y_axis_combo.setCurrentIndex(0)
                y_axis_layout.addWidget(y_axis_combo)
                layout.addLayout(y_axis_layout)
                
                # Dialog buttons
                button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                button_box.accepted.connect(dialog.accept)
                button_box.rejected.connect(dialog.reject)
                layout.addWidget(button_box)
                
                if dialog.exec() == QDialog.Accepted:
                    # Apply changes
                    item['legend_name'] = legend_edit.text()
                    item['color'] = current_color
                    item['line_style'] = style_combo.currentData()
                    item['marker'] = marker_combo.currentData()
                    item['y_axis'] = y_axis_combo.currentData()
                    
                    print(f"[PlotWizard] Updated line properties - Y-axis: {item['y_axis']}")
                    
                    # Update tables and plot
                    self._update_line_config_table()
                    self._update_plot()
                    
                    # Ensure proper column widths after update
                    self._ensure_table_column_widths()
        except Exception as e:
            print(f"[PlotWizard] Error showing line properties dialog: {str(e)}")
    
    def _on_subplot_info_clicked(self, subplot_num):
        """Handle subplot info button click"""
        try:
            # Count items in this subplot
            subplot_items = [item for item in self.plot_items if item['subplot'] == subplot_num]
            plot_types = list(set(item['plot_type'] for item in subplot_items))
            
            config = self.subplot_configs.get(subplot_num, {})
            
            info_text = f"""
            Subplot: {subplot_num}
            Number of items: {len(subplot_items)}
            Plot types: {', '.join(plot_types)}
            X Label: {config.get('xlabel', 'Not set')}
            Y Label: {config.get('ylabel', 'Not set')}
            Legend: {'Show' if config.get('show_legend', True) else 'Hide'}
            """
            
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox(self.window)
            msg.setWindowTitle("Subplot Information")
            msg.setText(info_text)
            msg.exec()
        except Exception as e:
            print(f"[PlotWizard] Error showing subplot info: {str(e)}")
    
    def _on_subplot_paint_clicked(self, subplot_num):
        """Handle subplot paint button click - opens subplot wizard"""
        try:
            # Ensure subplot config exists
            if subplot_num not in self.subplot_configs:
                self._create_subplot_config(subplot_num)
            
            config = self.subplot_configs[subplot_num]
            
            # Import and open the subplot wizard
            from subplot_wizard import SubplotWizard
            wizard = SubplotWizard(subplot_num, config, self.window)
            wizard.subplot_updated.connect(self._on_subplot_wizard_updated)
            wizard.exec()
        except Exception as e:
            print(f"[PlotWizard] Error opening subplot paint dialog: {str(e)}")
    
    def _on_subplot_delete_clicked(self, subplot_num):
        """Handle subplot delete button click"""
        try:
            # Count items in this subplot
            items_in_subplot = [item for item in self.plot_items if item['subplot'] == subplot_num]
            
            if not items_in_subplot:
                # No items in subplot, just remove config
                if subplot_num in self.subplot_configs:
                    del self.subplot_configs[subplot_num]
                self._update_subplot_config_table()
                return
            
            reply = QMessageBox.question(
                self.window, 
                "Confirm Delete", 
                f"Remove subplot {subplot_num} and all its {len(items_in_subplot)} items?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Remove all items in this subplot
                self.plot_items = [item for item in self.plot_items if item['subplot'] != subplot_num]
                
                # Remove subplot config
                if subplot_num in self.subplot_configs:
                    del self.subplot_configs[subplot_num]
                
                # Update tables and plot
                self._update_line_config_table()
                self._update_subplot_config_table()
                self._update_plot()
        except Exception as e:
            print(f"[PlotWizard] Error deleting subplot: {e}")
    
    def _on_subplot_wizard_updated(self, subplot_num, updated_config):
        """Handle subplot wizard updates"""
        try:
            # Update the subplot configuration
            if subplot_num in self.subplot_configs:
                self.subplot_configs[subplot_num].update(updated_config)
                # Update the plot to reflect changes
                self._update_plot()
        except Exception as e:
            print(f"[PlotWizard] Error updating subplot config: {str(e)}")
            
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
                    # User hasn't manually set dimensions - calculate optimal layout
                    rows, cols = self._calculate_optimal_dimensions(n_subplots)
                    print(f"[PlotWizard] Auto-setting dimensions: {rows}×{cols} (optimal layout for {n_subplots} subplots)")
                else:
                    # User has set dimensions - check if current capacity is sufficient
                    current_capacity = self.subplot_rows * self.subplot_cols
                    
                    if n_subplots <= current_capacity:
                        # Current dimensions can accommodate all subplots
                        rows = self.subplot_rows
                        cols = self.subplot_cols
                        print(f"[PlotWizard] Keeping user dimensions: {rows}×{cols} (capacity: {current_capacity})")
                    else:
                        # Need more capacity - calculate optimal dimensions that can fit all subplots
                        rows, cols = self._calculate_optimal_dimensions(n_subplots)
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
    
    def _calculate_optimal_dimensions(self, n_subplots):
        """Calculate optimal rows and columns for given number of subplots"""
        try:
            if n_subplots <= 0:
                return 1, 1
            
            # For small numbers, prefer vertical layout
            if n_subplots <= 3:
                return n_subplots, 1
            
            # For larger numbers, try to make it more square-like
            # Start with square root and adjust
            sqrt_n = int(n_subplots ** 0.5)
            
            # Try to find factors that are close to each other
            best_ratio = float('inf')
            best_rows = n_subplots
            best_cols = 1
            
            for rows in range(1, min(sqrt_n + 3, n_subplots + 1)):
                if n_subplots % rows == 0:
                    cols = n_subplots // rows
                    # Calculate aspect ratio (closer to 1 is better)
                    ratio = abs(rows - cols)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_rows = rows
                        best_cols = cols
            
            # If no perfect factors found, use the best approximation
            if best_ratio == float('inf'):
                # Fallback: prefer more rows than columns for better readability
                best_rows = int(n_subplots ** 0.5)
                best_cols = (n_subplots + best_rows - 1) // best_rows  # Ceiling division
            
            return best_rows, best_cols
            
        except Exception as e:
            print(f"[PlotWizard] Error calculating optimal dimensions: {str(e)}")
            return n_subplots, 1
    
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
            
    def _apply_axis_styling(self, ax, has_right_axis=False, is_right_axis=False):
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
            
            # Box style - handle right axis specially
            box_style = self.global_config.get('box_style', 'full')
            if box_style == 'full':
                # Keep all spines
                pass
            elif box_style == 'left_bottom':
                if is_right_axis:
                    # For right axis, only show right and bottom spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                else:
                    # For left axis, show left and bottom, but preserve right if needed
                    ax.spines['top'].set_visible(False)
                    if not has_right_axis:
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
            minor_ticks_on = config.get('minor_ticks_on', False)
            minor_tick_spacing = config.get('minor_tick_spacing', 0.2)
            minor_tick_value = config.get('minor_tick_value', 0.2)
            
            # Apply major tick settings to X axis
            if major_tick_spacing == 'auto':
                ax_left.xaxis.set_major_locator(AutoLocator())
            elif major_tick_spacing == 'custom':
                # Use MultipleLocator for custom spacing
                if isinstance(major_tick_value, (int, float)):
                    ax_left.xaxis.set_major_locator(MultipleLocator(major_tick_value))
                else:
                    ax_left.xaxis.set_major_locator(AutoLocator())
            
            # Apply major tick settings to Y axis
            if major_tick_spacing == 'auto':
                ax_left.yaxis.set_major_locator(AutoLocator())
            elif major_tick_spacing == 'custom':
                # Use MultipleLocator for custom spacing
                if isinstance(major_tick_value, (int, float)):
                    ax_left.yaxis.set_major_locator(MultipleLocator(major_tick_value))
                else:
                    ax_left.yaxis.set_major_locator(AutoLocator())
            
            # Apply same settings to right axis if it exists
            if ax_right is not None:
                if major_tick_spacing == 'auto':
                    ax_right.xaxis.set_major_locator(AutoLocator())
                    ax_right.yaxis.set_major_locator(AutoLocator())
                elif major_tick_spacing == 'custom':
                    # Use MultipleLocator for custom spacing
                    if isinstance(major_tick_value, (int, float)):
                        ax_right.xaxis.set_major_locator(MultipleLocator(major_tick_value))
                        ax_right.yaxis.set_major_locator(MultipleLocator(major_tick_value))
                    else:
                        ax_right.xaxis.set_major_locator(AutoLocator())
                        ax_right.yaxis.set_major_locator(AutoLocator())
            
            # Apply minor tick settings
            if minor_ticks_on:
                from matplotlib.ticker import MultipleLocator as MinorMultipleLocator
                
                # Determine minor tick spacing
                minor_spacing = minor_tick_value if config.get('minor_tick_spacing') == 'custom' else minor_tick_spacing
                
                # Apply minor ticks to X axis
                if isinstance(minor_spacing, (int, float)):
                    ax_left.xaxis.set_minor_locator(MinorMultipleLocator(minor_spacing))
                    ax_left.tick_params(which='minor', length=2, width=0.5)
                
                # Apply minor ticks to Y axis
                if isinstance(minor_spacing, (int, float)):
                    ax_left.yaxis.set_minor_locator(MinorMultipleLocator(minor_spacing))
                    ax_left.tick_params(which='minor', length=2, width=0.5)
                
                # Apply to right axis if it exists
                if ax_right is not None:
                    if isinstance(minor_spacing, (int, float)):
                        ax_right.xaxis.set_minor_locator(MinorMultipleLocator(minor_spacing))
                        ax_right.yaxis.set_minor_locator(MinorMultipleLocator(minor_spacing))
                        ax_right.tick_params(which='minor', length=2, width=0.5)
            else:
                # Turn off minor ticks
                ax_left.minorticks_off()
                if ax_right is not None:
                    ax_right.minorticks_off()
                    
        except Exception as e:
            print(f"[PlotWizard] Error applying tick controls: {e}")
            import traceback
            traceback.print_exc()
    
    def _configure_x_axis_positioning(self, ax_left, ax_right, items):
        """Configure x-axis positioning based on channel preferences"""
        try:
            # Determine if any channels prefer top x-axis
            has_top_x_axis = False
            has_bottom_x_axis = False
            
            for item in items:
                # Check channel xaxis property (for line/spectrogram wizards)
                channel = item.get('channel')
                if channel:
                    xaxis_pref = getattr(channel, 'xaxis', 'x-bottom')
                    if xaxis_pref == 'x-top':
                        has_top_x_axis = True
                    else:
                        has_bottom_x_axis = True
                
                # Check item x_axis property (for marker wizard)
                item_x_axis = item.get('x_axis', 'x-bottom')
                if item_x_axis == 'x-top':
                    has_top_x_axis = True
                else:
                    has_bottom_x_axis = True
            
            # Configure x-axis positioning
            if has_top_x_axis and not has_bottom_x_axis:
                # Only top x-axis requested
                ax_left.xaxis.tick_top()
                ax_left.xaxis.set_label_position('top')
                if ax_right is not None:
                    ax_right.xaxis.tick_top()
                    ax_right.xaxis.set_label_position('top')
                print(f"[PlotWizard] Configured top x-axis only")
                
            elif has_top_x_axis and has_bottom_x_axis:
                # Both top and bottom x-axis requested (mixed channels)
                ax_left.xaxis.set_ticks_position('both')
                ax_left.xaxis.set_label_position('bottom')  # Default label position
                if ax_right is not None:
                    ax_right.xaxis.set_ticks_position('both')
                    ax_right.xaxis.set_label_position('bottom')
                print(f"[PlotWizard] Configured both top and bottom x-axis")
                
            else:
                # Default to bottom x-axis
                ax_left.xaxis.tick_bottom()
                ax_left.xaxis.set_label_position('bottom')
                if ax_right is not None:
                    ax_right.xaxis.tick_bottom()
                    ax_right.xaxis.set_label_position('bottom')
                print(f"[PlotWizard] Configured bottom x-axis only")
                
        except Exception as e:
            print(f"[PlotWizard] Error configuring x-axis positioning: {e}")
            
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
                                print(f"[PlotWizard] Created right y-axis for subplot {subplot_num}")
                            ax = ax_right
                            print(f"[PlotWizard] Using right y-axis for {item['legend_name']}")
                        else:
                            ax = ax_left
                            print(f"[PlotWizard] Using left y-axis for {item['legend_name']}")
                        
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
                        if line_style == 'None':
                            line_style = 'none'  # Convert to matplotlib format
                        marker = item['marker'] if item['marker'] != 'None' else None
                        label = item['legend_name']
                        
                        if plot_type == 'Line':
                            # Use global line width setting
                            default_linewidth = self.global_config.get('line_width', 2.0)
                            linewidth = default_linewidth if len(items) == 1 else max(default_linewidth, 2.0)
                            markersize = 5 if len(items) > 1 else 4
                            z_order = item.get('z_order', 0)
                            ax.plot(x_plot, y_plot, color=color, linestyle=line_style, 
                                   marker=marker, label=label, markersize=markersize, linewidth=linewidth,
                                   zorder=z_order)
                        elif plot_type == 'Scatter':
                            # Get marker properties from item (set by marker wizard)
                            marker_size = item.get('marker_size', 20)
                            marker_alpha = item.get('marker_alpha', 0.7)
                            edge_color = item.get('edge_color', 'none')
                            edge_width = item.get('edge_width', 0)
                            z_order = item.get('z_order', 0)
                            
                            ax.scatter(x_plot, y_plot, 
                                     c=color, 
                                     marker=marker or 'o', 
                                     label=label, 
                                     s=marker_size, 
                                     alpha=marker_alpha,
                                     edgecolors=edge_color if edge_width > 0 else 'none',
                                     linewidths=edge_width,
                                     zorder=z_order)
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
                            
                            # Get channel-specific properties from spectrogram wizard
                            channel_alpha = getattr(channel, 'alpha', 1.0)
                            channel_interpolation = getattr(channel, 'interpolation', 'nearest')
                            channel_shading = getattr(channel, 'shading', 'gouraud')
                            
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
                                    
                                    # Apply color limits (from channel or config)
                                    if clim[0] is not None and clim[1] is not None:
                                        vmin, vmax = clim[0], clim[1]
                                    elif clim[0] is not None:
                                        vmin, vmax = clim[0], np.max(Zxx_db)
                                    elif clim[1] is not None:
                                        vmin, vmax = np.min(Zxx_db), clim[1]
                                    else:
                                        vmin, vmax = None, None
                                    
                                    # Use channel alpha, but adjust for overlaying
                                    alpha_val = channel_alpha * (0.7 if len(items) > 1 else 1.0)
                                    
                                    # Use pcolormesh with channel-specific shading
                                    im = ax.pcolormesh(t_axis, f_axis, Zxx_db, cmap=colormap, 
                                                     shading=channel_shading, alpha=alpha_val, vmin=vmin, vmax=vmax)
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
                                
                                # Use channel alpha, but adjust for overlaying
                                alpha_val = channel_alpha * (0.7 if len(items) > 1 else 1.0)
                                im = ax.pcolormesh(t, f, Sxx_db, cmap=colormap, shading=channel_shading, 
                                                 alpha=alpha_val, vmin=vmin, vmax=vmax)
                            else:
                                print(f"[PlotWizard] Spectrogram requires at least 100 data points, got {len(y_plot)}")
                                continue
                            
                            # Only add colorbar if this subplot doesn't have one already and colorbars are enabled
                            colorbar_key = (subplot_num, item['y_axis'])
                            show_colorbar = config.get('show_colorbar', True)
                            if colorbar_key not in colorbars_added and show_colorbar:
                                try:
                                    # Get colorbar properties from channel and config
                                    colorbar_label = config.get('colorbar_label', f'Power (dB) - {label}')
                                    colorbar_position = config.get('colorbar_position', 'bottom')
                                    
                                    # Get enhanced colorbar properties from channel
                                    colorbar_pad = getattr(channel, 'colorbar_pad', 0.05)
                                    colorbar_shrink = getattr(channel, 'colorbar_shrink', 0.8)
                                    colorbar_aspect = getattr(channel, 'colorbar_aspect', 20)
                                    colorbar_ticks = getattr(channel, 'colorbar_ticks', 5)
                                    tick_format = getattr(channel, 'tick_format', '%.1f')
                                    label_fontsize = getattr(channel, 'colorbar_label_fontsize', 10)
                                    tick_fontsize = getattr(channel, 'colorbar_tick_fontsize', 8)
                                    
                                    # Create colorbar with specified position and properties
                                    if colorbar_position in ['bottom', 'top']:
                                        orientation = 'horizontal'
                                        location = colorbar_position
                                    else:  # 'left' or 'right'
                                        orientation = 'vertical'
                                        location = colorbar_position
                                    
                                    # Create colorbar with enhanced properties
                                    cbar = self.window.figure.colorbar(
                                        im, ax=ax, 
                                        label=colorbar_label,
                                        orientation=orientation, 
                                        location=location,
                                        pad=colorbar_pad, 
                                        shrink=colorbar_shrink, 
                                        aspect=colorbar_aspect
                                    )
                                    
                                    # Configure colorbar ticks and labels
                                    if orientation == 'horizontal':
                                        cbar.ax.tick_params(labelsize=tick_fontsize)
                                        cbar.set_label(colorbar_label, fontsize=label_fontsize)
                                        
                                        # Set custom tick locations and format
                                        vmin, vmax = im.get_clim()
                                        tick_locations = np.linspace(vmin, vmax, colorbar_ticks)
                                        cbar.set_ticks(tick_locations)
                                        tick_labels = [tick_format % val for val in tick_locations]
                                        cbar.set_ticklabels(tick_labels)
                                    else:  # vertical
                                        cbar.ax.tick_params(labelsize=tick_fontsize)
                                        cbar.set_label(colorbar_label, fontsize=label_fontsize)
                                        
                                        # Set custom tick locations and format
                                        vmin, vmax = im.get_clim()
                                        tick_locations = np.linspace(vmin, vmax, colorbar_ticks)
                                        cbar.set_ticks(tick_locations)
                                        tick_labels = [tick_format % val for val in tick_locations]
                                        cbar.set_ticklabels(tick_labels)
                                    
                                    colorbars_added.add(colorbar_key)
                                    print(f"[PlotWizard] Added enhanced colorbar at {colorbar_position} for subplot {subplot_num}")
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
                    
                    # Set subplot title if specified
                    if config.get('title'):
                        ax_left.set_title(config['title'])
                    
                    # Apply axis styling
                    self._apply_axis_styling(ax_left, has_right_axis=(ax_right is not None))
                    if ax_right is not None:
                        self._apply_axis_styling(ax_right, is_right_axis=True)
                    
                    # Configure x-axis positioning based on channel preferences
                    self._configure_x_axis_positioning(ax_left, ax_right, items)
                    
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
            
            # Add subtle watermark
            self._add_watermark()
            
            # Update canvas
            self.window.canvas.draw()
            
        except Exception as e:
            print(f"[PlotWizard] Error updating plot: {str(e)}")
            traceback.print_exc()
    
    def _add_watermark(self):
        """Add subtle watermark to the plot"""
        try:
            # Add watermark text in lower right corner
            self.window.figure.text(
                0.98, 0.02,  # Position: 98% from left, 2% from bottom
                "Generated via Raw Dog",
                fontsize=8,
                color='gray',
                alpha=0.6,
                ha='right',  # Right-aligned
                va='bottom',  # Bottom-aligned
                transform=self.window.figure.transFigure,  # Use figure coordinates
                style='italic'
            )
        except Exception as e:
            print(f"[PlotWizard] Error adding watermark: {e}")
            # Don't let watermark errors break the plot
            pass
            
    def show(self):
        """Show the plot wizard window"""
        self.window.show()
        
        # Ensure proper table column widths after window is shown
        self._ensure_table_column_widths()
        
    def _ensure_table_column_widths(self):
        """Ensure proper column widths for both tables"""
        try:
            # Line config table
            line_table = self.window.line_config_table
            line_header = line_table.horizontalHeader()
            line_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Subplot#
            line_header.setSectionResizeMode(1, QHeaderView.Stretch)          # Channel Name
            line_header.setSectionResizeMode(2, QHeaderView.Fixed)            # Actions
            line_header.setMinimumSectionSize(80)
            line_header.resizeSection(2, 100)  # Ensure actions column has proper width
            
            # Subplot config table
            subplot_table = self.window.subplot_config_table
            subplot_header = subplot_table.horizontalHeader()
            subplot_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Subplot#
            subplot_header.setSectionResizeMode(1, QHeaderView.Stretch)          # Subplot Name
            subplot_header.setSectionResizeMode(2, QHeaderView.Fixed)            # Actions
            subplot_header.setMinimumSectionSize(80)
            subplot_header.resizeSection(2, 100)  # Ensure actions column has proper width
            
        except Exception as e:
            print(f"[PlotWizard] Error ensuring table column widths: {e}")
    
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

    def _get_channel_size(self, channel):
        """Get the size/dimensions of channel data"""
        try:
            if hasattr(channel, 'xdata') and hasattr(channel, 'ydata'):
                if channel.xdata is not None and channel.ydata is not None:
                    return f"({len(channel.xdata)}, 2)"
                elif channel.ydata is not None:
                    return f"({len(channel.ydata)},)"
            return "No data"
        except Exception as e:
            print(f"[PlotWizard] Error getting channel size: {e}")
            return "Unknown"