from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QLineEdit, QPushButton, QListWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QTextEdit, QCheckBox, QFrame, QTabWidget, QRadioButton, QButtonGroup,
    QGroupBox, QSpinBox, QHeaderView, QAbstractItemView, QDoubleSpinBox, QFormLayout
)
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from base_plot_wizard import BasePlotWizard
from channel import Channel
from plot_manager import StylePreviewWidget


class PlotWizardRefactored(BasePlotWizard):
    """
    Refactored Plot Wizard inheriting from BasePlotWizard
    Demonstrates consolidation of plotting functionality for custom plot building
    """
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        # Initialize plot-specific state
        self.subplot_configs = []
        self.line_configs = []
        
        # UI components specific to plot wizard
        self.file_dropdown = None
        self.channel_dropdown = None
        self.add_btn = None
        self.line_config_table = None
        self.subplot_config_table = None
        self.rows_spinbox = None
        self.cols_spinbox = None
        
        # Plot configuration controls
        self.sharex_checkbox = None
        self.sharey_checkbox = None
        self.tick_direction_combo = None
        self.tick_width_spinbox = None
        self.tick_length_spinbox = None
        self.line_width_spinbox = None
        self.axis_linewidth_spinbox = None
        self.font_size_spinbox = None
        self.font_weight_combo = None
        self.font_family_combo = None
        self.grid_checkbox = None
        self.box_style_combo = None
        self.legend_fontsize_spinbox = None
        self.legend_ncol_spinbox = None
        self.legend_frameon_checkbox = None
        self.tight_layout_checkbox = None
        
        # Initialize base class
        super().__init__(file_manager, channel_manager, signal_bus, parent)
        
        # Set window properties
        self.setWindowTitle("Custom Plot Builder")
        self.setMinimumSize(1400, 800)
        
        # Initialize plot-specific components
        self._initialize_plot_components()
        
        self._log_message("Plot wizard initialized successfully")
    
    def _get_wizard_type(self) -> str:
        """Get the wizard type name"""
        return "Plot"
    
    def _initialize_plot_components(self):
        """Initialize plot-specific components"""
        # Initialize with default subplot configuration
        self.subplot_configs = [{'name': 'Main Plot', 'channels': []}]
        
        # Update UI
        self._update_file_dropdown()
        self._update_subplot_config_table()
    
    def _create_main_content(self, layout: QVBoxLayout):
        """Create the main content area specific to plot wizard"""
        # Channel selector row
        self._create_channel_selector_section(layout)
        
        # Configuration tables
        self._create_config_tables_section(layout)
        
        # Subplot dimension controls
        self._create_subplot_dimension_section(layout)
        
        # Plot configuration panel
        self._create_plot_configuration_section(layout)
    
    def _create_channel_selector_section(self, layout: QVBoxLayout):
        """Create the channel selector section"""
        selector_layout = QHBoxLayout()
        
        self.file_dropdown = QComboBox()
        self.file_dropdown.currentTextChanged.connect(self._on_file_changed)
        
        self.channel_dropdown = QComboBox()
        
        self.add_btn = QPushButton("Add to Plot")
        self.add_btn.clicked.connect(self._on_add_channel)
        
        selector_layout.addWidget(QLabel("File:"))
        selector_layout.addWidget(self.file_dropdown)
        selector_layout.addWidget(QLabel("Channel:"))
        selector_layout.addWidget(self.channel_dropdown)
        selector_layout.addWidget(self.add_btn)
        
        layout.addLayout(selector_layout)
    
    def _create_config_tables_section(self, layout: QVBoxLayout):
        """Create the configuration tables section"""
        # Line configurations table
        layout.addWidget(QLabel("Line Configurations"))
        self.line_config_table = QTableWidget(0, 6)
        self.line_config_table.setHorizontalHeaderLabels([
            "Subplot#", "Channel Name", "Type", "Size", "Actions", ""
        ])
        self.line_config_table.setMaximumHeight(150)
        layout.addWidget(self.line_config_table)
        
        # Subplot configurations table
        layout.addWidget(QLabel("Subplot Configurations"))
        self.subplot_config_table = QTableWidget(0, 3)
        self.subplot_config_table.setHorizontalHeaderLabels([
            "Subplot#", "Subplot Name", "Actions"
        ])
        self.subplot_config_table.setMaximumHeight(100)
        layout.addWidget(self.subplot_config_table)
    
    def _create_subplot_dimension_section(self, layout: QVBoxLayout):
        """Create the subplot dimension controls section"""
        dimension_layout = QHBoxLayout()
        
        dimension_label = QLabel("Subplot Dimension:")
        
        self.rows_spinbox = QSpinBox()
        self.rows_spinbox.setMinimum(1)
        self.rows_spinbox.setMaximum(10)
        self.rows_spinbox.setValue(1)
        self.rows_spinbox.setToolTip("Number of subplot rows")
        self.rows_spinbox.valueChanged.connect(self._on_dimension_changed)
        
        dimension_x_label = QLabel("×")
        
        self.cols_spinbox = QSpinBox()
        self.cols_spinbox.setMinimum(1)
        self.cols_spinbox.setMaximum(10)
        self.cols_spinbox.setValue(1)
        self.cols_spinbox.setToolTip("Number of subplot columns")
        self.cols_spinbox.valueChanged.connect(self._on_dimension_changed)
        
        dimension_layout.addWidget(dimension_label)
        dimension_layout.addWidget(self.rows_spinbox)
        dimension_layout.addWidget(dimension_x_label)
        dimension_layout.addWidget(self.cols_spinbox)
        dimension_layout.addStretch()
        
        layout.addLayout(dimension_layout)
    
    def _create_plot_configuration_section(self, layout: QVBoxLayout):
        """Create the plot configuration section"""
        config_group = QGroupBox("Plot Configuration")
        config_layout = QFormLayout(config_group)
        
        # Share axes
        axes_layout = QHBoxLayout()
        self.sharex_checkbox = QCheckBox("Share X Axis")
        self.sharey_checkbox = QCheckBox("Share Y Axis")
        axes_layout.addWidget(self.sharex_checkbox)
        axes_layout.addWidget(self.sharey_checkbox)
        config_layout.addRow("Axes:", axes_layout)
        
        # Tick settings
        tick_layout = QHBoxLayout()
        self.tick_direction_combo = QComboBox()
        self.tick_direction_combo.addItems(["in", "out", "inout"])
        self.tick_width_spinbox = QDoubleSpinBox()
        self.tick_width_spinbox.setRange(0.1, 5.0)
        self.tick_width_spinbox.setValue(1.0)
        tick_layout.addWidget(QLabel("Direction:"))
        tick_layout.addWidget(self.tick_direction_combo)
        tick_layout.addWidget(QLabel("Width:"))
        tick_layout.addWidget(self.tick_width_spinbox)
        config_layout.addRow("Ticks:", tick_layout)
        
        # Line settings
        line_layout = QHBoxLayout()
        self.tick_length_spinbox = QDoubleSpinBox()
        self.tick_length_spinbox.setRange(1.0, 20.0)
        self.tick_length_spinbox.setValue(4.0)
        self.line_width_spinbox = QDoubleSpinBox()
        self.line_width_spinbox.setRange(0.1, 10.0)
        self.line_width_spinbox.setValue(2.0)
        line_layout.addWidget(QLabel("Tick Length:"))
        line_layout.addWidget(self.tick_length_spinbox)
        line_layout.addWidget(QLabel("Line Width:"))
        line_layout.addWidget(self.line_width_spinbox)
        config_layout.addRow("Lines:", line_layout)
        
        # Font settings
        font_layout = QHBoxLayout()
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(6, 72)
        self.font_size_spinbox.setValue(12)
        self.font_weight_combo = QComboBox()
        self.font_weight_combo.addItems(["normal", "bold", "light", "ultralight", "heavy", "black"])
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(["sans-serif", "serif", "monospace", "cursive", "fantasy"])
        font_layout.addWidget(QLabel("Size:"))
        font_layout.addWidget(self.font_size_spinbox)
        font_layout.addWidget(QLabel("Weight:"))
        font_layout.addWidget(self.font_weight_combo)
        font_layout.addWidget(QLabel("Family:"))
        font_layout.addWidget(self.font_family_combo)
        config_layout.addRow("Font:", font_layout)
        
        # Grid and style
        style_layout = QHBoxLayout()
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(True)
        self.box_style_combo = QComboBox()
        self.box_style_combo.addItems(["full", "left_bottom", "none"])
        style_layout.addWidget(self.grid_checkbox)
        style_layout.addWidget(QLabel("Box Style:"))
        style_layout.addWidget(self.box_style_combo)
        config_layout.addRow("Style:", style_layout)
        
        # Legend settings
        legend_layout = QHBoxLayout()
        self.legend_fontsize_spinbox = QSpinBox()
        self.legend_fontsize_spinbox.setRange(6, 72)
        self.legend_fontsize_spinbox.setValue(10)
        self.legend_ncol_spinbox = QSpinBox()
        self.legend_ncol_spinbox.setRange(1, 10)
        self.legend_ncol_spinbox.setValue(1)
        self.legend_frameon_checkbox = QCheckBox("Frame")
        self.legend_frameon_checkbox.setChecked(True)
        legend_layout.addWidget(QLabel("Font Size:"))
        legend_layout.addWidget(self.legend_fontsize_spinbox)
        legend_layout.addWidget(QLabel("Columns:"))
        legend_layout.addWidget(self.legend_ncol_spinbox)
        legend_layout.addWidget(self.legend_frameon_checkbox)
        config_layout.addRow("Legend:", legend_layout)
        
        # Layout
        self.tight_layout_checkbox = QCheckBox("Tight Layout")
        self.tight_layout_checkbox.setChecked(True)
        config_layout.addRow("Layout:", self.tight_layout_checkbox)
        
        layout.addWidget(config_group)
        
        # Connect all configuration controls to update plot
        self._connect_config_controls()
    
    def _connect_config_controls(self):
        """Connect configuration controls to plot update"""
        controls = [
            self.sharex_checkbox, self.sharey_checkbox, self.tick_direction_combo,
            self.tick_width_spinbox, self.tick_length_spinbox, self.line_width_spinbox,
            self.font_size_spinbox, self.font_weight_combo, self.font_family_combo,
            self.grid_checkbox, self.box_style_combo, self.legend_fontsize_spinbox,
            self.legend_ncol_spinbox, self.legend_frameon_checkbox, self.tight_layout_checkbox
        ]
        
        for control in controls:
            if hasattr(control, 'toggled'):
                control.toggled.connect(self._on_config_changed)
            elif hasattr(control, 'valueChanged'):
                control.valueChanged.connect(self._on_config_changed)
            elif hasattr(control, 'currentTextChanged'):
                control.currentTextChanged.connect(self._on_config_changed)
    
    def _get_channels_to_plot(self) -> List[Channel]:
        """Get channels to plot from line configurations"""
        channels = []
        for config in self.line_configs:
            channel = config.get('channel')
            if channel:
                channels.append(channel)
        return channels
    
    def _update_file_dropdown(self):
        """Update file dropdown with available files"""
        if not self.file_dropdown or not self.file_manager:
            return
        
        self.file_dropdown.clear()
        
        files = self.file_manager.get_all_files()
        for file_info in files:
            self.file_dropdown.addItem(file_info.filename, file_info)
        
        # Update channel dropdown for first file
        if files:
            self._update_channel_dropdown(files[0])
    
    def _update_channel_dropdown(self, file_info):
        """Update channel dropdown for selected file"""
        if not self.channel_dropdown or not self.channel_manager:
            return
        
        self.channel_dropdown.clear()
        
        channels = self.channel_manager.get_channels_by_file(file_info.file_id)
        for channel in channels:
            if channel.show and channel.ydata is not None:
                display_name = channel.legend_label or channel.channel_id
                self.channel_dropdown.addItem(display_name, channel)
    
    def _update_line_config_table(self):
        """Update line configuration table"""
        if not self.line_config_table:
            return
        
        self.line_config_table.setRowCount(len(self.line_configs))
        
        for row, config in enumerate(self.line_configs):
            channel = config.get('channel')
            
            # Subplot number
            subplot_item = QTableWidgetItem(str(config.get('subplot', 1)))
            self.line_config_table.setItem(row, 0, subplot_item)
            
            # Channel name
            name = channel.legend_label if channel else "Unknown"
            name_item = QTableWidgetItem(name)
            self.line_config_table.setItem(row, 1, name_item)
            
            # Type
            type_item = QTableWidgetItem(config.get('type', 'Line'))
            self.line_config_table.setItem(row, 2, type_item)
            
            # Size
            size_item = QTableWidgetItem(str(config.get('size', 1)))
            self.line_config_table.setItem(row, 3, size_item)
            
            # Actions
            actions_btn = QPushButton("⚙")
            actions_btn.clicked.connect(lambda: self._open_line_config(config))
            self.line_config_table.setCellWidget(row, 4, actions_btn)
    
    def _update_subplot_config_table(self):
        """Update subplot configuration table"""
        if not self.subplot_config_table:
            return
        
        self.subplot_config_table.setRowCount(len(self.subplot_configs))
        
        for row, config in enumerate(self.subplot_configs):
            # Subplot number
            subplot_item = QTableWidgetItem(str(row + 1))
            self.subplot_config_table.setItem(row, 0, subplot_item)
            
            # Subplot name
            name_item = QTableWidgetItem(config.get('name', f'Subplot {row + 1}'))
            self.subplot_config_table.setItem(row, 1, name_item)
            
            # Actions
            actions_btn = QPushButton("⚙")
            actions_btn.clicked.connect(lambda: self._open_subplot_config(config))
            self.subplot_config_table.setCellWidget(row, 2, actions_btn)
    
    def _update_plot(self):
        """Override to create custom subplot layout"""
        if not self.figure:
            return
        
        # Clear the figure
        self.figure.clear()
        
        # Get plot dimensions
        rows = self.rows_spinbox.value() if self.rows_spinbox else 1
        cols = self.cols_spinbox.value() if self.cols_spinbox else 1
        
        # Get channels to plot
        channels = self._get_channels_to_plot()
        visible_channels = [ch for ch in channels if ch.channel_id in self.visible_channels]
        
        if not visible_channels:
            self.canvas.draw()
            return
        
        # Create subplots
        if rows == 1 and cols == 1:
            ax = self.figure.add_subplot(111)
            axes = [ax]
        else:
            axes = []
            for i in range(rows * cols):
                ax = self.figure.add_subplot(rows, cols, i + 1)
                axes.append(ax)
        
        # Plot channels on appropriate subplots
        for config in self.line_configs:
            channel = config.get('channel')
            if channel and channel.channel_id in self.visible_channels:
                subplot_idx = config.get('subplot', 1) - 1
                
                if 0 <= subplot_idx < len(axes):
                    ax = axes[subplot_idx]
                    
                    if hasattr(channel, 'xdata') and hasattr(channel, 'ydata'):
                        if channel.xdata is not None and channel.ydata is not None:
                            color = getattr(channel, 'color', '#1f77b4')
                            style = getattr(channel, 'style', '-')
                            marker = getattr(channel, 'marker', None)
                            label = channel.legend_label or channel.ylabel or channel.channel_id
                            
                            ax.plot(channel.xdata, channel.ydata,
                                   color=color, linestyle=style, marker=marker,
                                   label=label, linewidth=self.plot_config['line_width'],
                                   markersize=self.plot_config['marker_size'])
        
        # Configure all subplots
        for ax in axes:
            if self.plot_config['grid']:
                ax.grid(True, alpha=0.3)
            
            if self.plot_config['legend_show']:
                ax.legend(loc=self.plot_config['legend_position'],
                         fontsize=self.plot_config['font_size'])
        
        # Apply tight layout
        if self.plot_config['tight_layout']:
            self.figure.tight_layout()
        
        # Update canvas
        self.canvas.draw()
    
    def _open_line_config(self, config):
        """Open line configuration dialog"""
        channel = config.get('channel')
        if channel:
            self._open_channel_config(channel)
    
    def _open_subplot_config(self, config):
        """Open subplot configuration dialog"""
        from subplot_wizard import SubplotWizard
        try:
            wizard = SubplotWizard(config, self)
            wizard.config_updated.connect(self._on_subplot_config_updated)
            wizard.exec()
        except Exception as e:
            self._log_message(f"Error opening subplot wizard: {str(e)}")
    
    def _on_subplot_config_updated(self, config):
        """Handle subplot configuration update"""
        self._update_subplot_config_table()
        self._schedule_plot_update()
    
    # Event handlers
    def _on_file_changed(self, filename):
        """Handle file selection change"""
        if not self.file_dropdown:
            return
        
        file_info = self.file_dropdown.currentData()
        if file_info:
            self._update_channel_dropdown(file_info)
    
    def _on_add_channel(self):
        """Handle add channel button click"""
        if not self.channel_dropdown:
            return
        
        channel = self.channel_dropdown.currentData()
        if channel:
            # Add to line configurations
            config = {
                'channel': channel,
                'subplot': 1,
                'type': 'Line',
                'size': 1
            }
            self.line_configs.append(config)
            
            # Add to plot
            self.add_channel(channel, visible=True)
            
            # Update tables
            self._update_line_config_table()
            
            self._log_message(f"Added channel: {channel.legend_label}")
    
    def _on_dimension_changed(self):
        """Handle subplot dimension change"""
        # Update subplot configurations to match new dimensions
        rows = self.rows_spinbox.value() if self.rows_spinbox else 1
        cols = self.cols_spinbox.value() if self.cols_spinbox else 1
        total_subplots = rows * cols
        
        # Adjust subplot configurations
        while len(self.subplot_configs) < total_subplots:
            self.subplot_configs.append({
                'name': f'Subplot {len(self.subplot_configs) + 1}',
                'channels': []
            })
        
        while len(self.subplot_configs) > total_subplots:
            self.subplot_configs.pop()
        
        self._update_subplot_config_table()
        self._schedule_plot_update()
    
    def _on_config_changed(self):
        """Handle configuration control changes"""
        # Update plot configuration
        if self.grid_checkbox:
            self.plot_config['grid'] = self.grid_checkbox.isChecked()
        if self.legend_frameon_checkbox:
            self.plot_config['legend_show'] = self.legend_frameon_checkbox.isChecked()
        if self.font_size_spinbox:
            self.plot_config['font_size'] = self.font_size_spinbox.value()
        if self.line_width_spinbox:
            self.plot_config['line_width'] = self.line_width_spinbox.value()
        if self.tight_layout_checkbox:
            self.plot_config['tight_layout'] = self.tight_layout_checkbox.isChecked()
        
        # Schedule plot update
        self._schedule_plot_update()
    
    def get_plot_configuration(self) -> Dict:
        """Get current plot configuration"""
        config = {
            'dimensions': {
                'rows': self.rows_spinbox.value() if self.rows_spinbox else 1,
                'cols': self.cols_spinbox.value() if self.cols_spinbox else 1
            },
            'axes': {
                'sharex': self.sharex_checkbox.isChecked() if self.sharex_checkbox else False,
                'sharey': self.sharey_checkbox.isChecked() if self.sharey_checkbox else False
            },
            'ticks': {
                'direction': self.tick_direction_combo.currentText() if self.tick_direction_combo else 'in',
                'width': self.tick_width_spinbox.value() if self.tick_width_spinbox else 1.0,
                'length': self.tick_length_spinbox.value() if self.tick_length_spinbox else 4.0
            },
            'font': {
                'size': self.font_size_spinbox.value() if self.font_size_spinbox else 12,
                'weight': self.font_weight_combo.currentText() if self.font_weight_combo else 'normal',
                'family': self.font_family_combo.currentText() if self.font_family_combo else 'sans-serif'
            },
            'style': {
                'grid': self.grid_checkbox.isChecked() if self.grid_checkbox else True,
                'box_style': self.box_style_combo.currentText() if self.box_style_combo else 'full'
            },
            'legend': {
                'fontsize': self.legend_fontsize_spinbox.value() if self.legend_fontsize_spinbox else 10,
                'ncol': self.legend_ncol_spinbox.value() if self.legend_ncol_spinbox else 1,
                'frameon': self.legend_frameon_checkbox.isChecked() if self.legend_frameon_checkbox else True
            },
            'layout': {
                'tight': self.tight_layout_checkbox.isChecked() if self.tight_layout_checkbox else True
            }
        }
        return config
    
    def set_plot_configuration(self, config: Dict):
        """Set plot configuration"""
        # Update UI controls based on configuration
        dimensions = config.get('dimensions', {})
        if self.rows_spinbox:
            self.rows_spinbox.setValue(dimensions.get('rows', 1))
        if self.cols_spinbox:
            self.cols_spinbox.setValue(dimensions.get('cols', 1))
        
        axes = config.get('axes', {})
        if self.sharex_checkbox:
            self.sharex_checkbox.setChecked(axes.get('sharex', False))
        if self.sharey_checkbox:
            self.sharey_checkbox.setChecked(axes.get('sharey', False))
        
        # Update other controls similarly...
        
        # Update plot
        self._schedule_plot_update() 