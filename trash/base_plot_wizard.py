from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QGroupBox, QLabel, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit, QFrame,
    QFormLayout, QGridLayout, QHeaderView, QAbstractItemView, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod
from datetime import datetime

from channel import Channel
from plot_manager import PlotCanvas, StylePreviewWidget


class BasePlotWizard(QWidget, ABC):
    """
    Base class for all plot wizards (Process, Mixer, Comparison, Plot)
    Provides common plotting functionality and standardized UI patterns
    """
    
    # Common signals
    plot_updated = Signal()
    channel_selected = Signal(str)  # channel_id
    plot_exported = Signal(str)     # filename
    wizard_closed = Signal()
    state_changed = Signal(str)     # state message
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        # Store managers
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        # Initialize state
        self._stats = {
            'plots_created': 0,
            'channels_plotted': 0,
            'last_update_time': None,
            'session_start': time.time()
        }
        
        # Plot configuration
        self.plot_config = {
            'figure_size': (10, 6),
            'dpi': 100,
            'tight_layout': True,
            'grid': True,
            'legend_show': True,
            'legend_position': 'upper right',
            'font_size': 10,
            'line_width': 1.5,
            'marker_size': 4
        }
        
        # UI components (will be created by subclasses)
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.plot_area = None
        self.console_output = None
        self.results_table = None
        
        # Channel tracking
        self.plotted_channels = {}  # channel_id -> plot_info
        self.visible_channels = set()
        
        # Update timer for performance
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_plot_update)
        self.update_delay = 200  # ms
        
        # Configuration wizard connections
        self._config_wizard_connections = {}
        
        # Initialize UI
        self._setup_base_ui()
        
    @abstractmethod
    def _get_wizard_type(self) -> str:
        """Get the wizard type name (e.g., 'Process', 'Mixer', 'Comparison')"""
        pass
    
    @abstractmethod
    def _create_main_content(self, layout: QVBoxLayout):
        """Create the main content area specific to this wizard type"""
        pass
    
    @abstractmethod
    def _get_channels_to_plot(self) -> List[Channel]:
        """Get the list of channels that should be plotted"""
        pass
    
    def _setup_base_ui(self):
        """Setup the base UI structure common to all plot wizards"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel for controls
        self.left_panel = QWidget()
        self.left_panel.setMinimumWidth(400)
        left_layout = QVBoxLayout(self.left_panel)
        
        # Let subclasses create their specific content
        self._create_main_content(left_layout)
        
        main_splitter.addWidget(self.left_panel)
        
        # Right panel for plot and results
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        
        # Create tabbed interface for plot and results
        self.plot_tabs = QTabWidget()
        
        # Plot tab
        plot_tab = self._create_plot_tab()
        self.plot_tabs.addTab(plot_tab, "Plot")
        
        # Results tab (optional, can be overridden)
        results_tab = self._create_results_tab()
        if results_tab:
            self.plot_tabs.addTab(results_tab, "Results")
        
        # Console tab
        console_tab = self._create_console_tab()
        self.plot_tabs.addTab(console_tab, "Console")
        
        right_layout.addWidget(self.plot_tabs)
        main_splitter.addWidget(self.right_panel)
        
        # Set initial splitter sizes
        main_splitter.setSizes([400, 800])
        
    def _create_plot_tab(self) -> QWidget:
        """Create the plot tab with matplotlib canvas"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=self.plot_config['figure_size'], 
                           dpi=self.plot_config['dpi'],
                           tight_layout=self.plot_config['tight_layout'])
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, tab)
        
        # Add toolbar and canvas
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create plot controls
        controls_frame = self._create_plot_controls()
        if controls_frame:
            layout.addWidget(controls_frame)
        
        return tab
    
    def _create_plot_controls(self) -> Optional[QWidget]:
        """Create plot control buttons - can be overridden by subclasses"""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        # Export button
        export_btn = QPushButton("Export Plot")
        export_btn.clicked.connect(self._export_plot)
        layout.addWidget(export_btn)
        
        # Clear button
        clear_btn = QPushButton("Clear Plot")
        clear_btn.clicked.connect(self._clear_plot)
        layout.addWidget(clear_btn)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._force_plot_update)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
        
        # Plot config button
        config_btn = QPushButton("Plot Settings")
        config_btn.clicked.connect(self._show_plot_config)
        layout.addWidget(config_btn)
        
        return frame
    
    def _create_results_tab(self) -> Optional[QWidget]:
        """Create results tab - can be overridden by subclasses"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.results_table)
        
        return tab
    
    def _create_console_tab(self) -> QWidget:
        """Create console tab for logging and output"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Consolas", 9))
        self.console_output.setMaximumHeight(200)
        layout.addWidget(self.console_output)
        
        # Add initial message
        wizard_type = self._get_wizard_type()
        self._log_message(f"{wizard_type} Wizard initialized successfully")
        
        return tab
    
    def _create_channel_table(self, parent_layout: QVBoxLayout, title: str = "Channels") -> QTableWidget:
        """Create a standardized channel table"""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        
        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Standard columns: Show, Name, Style, Actions
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Show", "Channel", "Style", "Actions"])
        
        # Configure column widths
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Show
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Channel
        header.setSectionResizeMode(2, QHeaderView.Fixed)   # Style
        header.setSectionResizeMode(3, QHeaderView.Fixed)   # Actions
        
        table.setColumnWidth(0, 50)   # Show checkbox
        table.setColumnWidth(2, 100)  # Style preview
        table.setColumnWidth(3, 80)   # Actions
        
        layout.addWidget(table)
        parent_layout.addWidget(group)
        
        return table
    
    def _add_channel_to_table(self, table: QTableWidget, channel: Channel, row: int = None):
        """Add a channel to a table with standard format"""
        if row is None:
            row = table.rowCount()
            table.insertRow(row)
        
        # Show checkbox
        show_checkbox = QCheckBox()
        show_checkbox.setChecked(channel.channel_id in self.visible_channels)
        show_checkbox.toggled.connect(lambda checked: self._on_channel_visibility_changed(channel.channel_id, checked))
        table.setCellWidget(row, 0, show_checkbox)
        
        # Channel name
        name_item = QTableWidgetItem(channel.legend_label or channel.ylabel or channel.channel_id)
        name_item.setData(Qt.UserRole, channel.channel_id)
        table.setItem(row, 1, name_item)
        
        # Style preview
        style_widget = StylePreviewWidget(
            color=getattr(channel, 'color', '#1f77b4'),
            style=getattr(channel, 'style', '-'),
            marker=getattr(channel, 'marker', None)
        )
        table.setCellWidget(row, 2, style_widget)
        
        # Actions button
        actions_btn = QPushButton("⚙")
        actions_btn.setMaximumWidth(30)
        actions_btn.clicked.connect(lambda: self._open_channel_config(channel))
        table.setCellWidget(row, 3, actions_btn)
    
    def _open_channel_config(self, channel: Channel):
        """Open configuration wizard for a channel"""
        try:
            # Determine which wizard to open based on channel type or plot type
            wizard_type = self._determine_config_wizard_type(channel)
            
            if wizard_type == 'line':
                from line_wizard import LineWizard
                wizard = LineWizard(channel, self)
                wizard.channel_updated.connect(lambda ch_id: self._on_channel_updated_from_wizard(ch_id))
                wizard.exec()
                
            elif wizard_type == 'marker':
                from marker_wizard import MarkerWizard
                # For marker wizard, we need to create a marker config dict
                marker_config = self._create_marker_config_from_channel(channel)
                wizard = MarkerWizard(marker_config, self)
                wizard.marker_updated.connect(lambda config: self._on_marker_updated_from_wizard(channel.channel_id, config))
                wizard.exec()
                
            elif wizard_type == 'spectrogram':
                from spectrogram_wizard import SpectrogramWizard
                wizard = SpectrogramWizard(channel, self)
                wizard.spectrogram_updated.connect(lambda ch_id: self._on_spectrogram_updated_from_wizard(ch_id))
                wizard.exec()
                
            else:
                self._log_message(f"No configuration wizard available for channel type: {wizard_type}")
                
        except Exception as e:
            self._log_message(f"Error opening configuration wizard: {str(e)}")
    
    def _determine_config_wizard_type(self, channel: Channel) -> str:
        """Determine which configuration wizard to use for a channel"""
        # Check channel tags or metadata for plot type hints
        if hasattr(channel, 'tags'):
            if 'spectrogram' in channel.tags:
                return 'spectrogram'
            elif 'scatter' in channel.tags or 'marker' in channel.tags:
                return 'marker'
        
        # Check if channel has spectrogram data
        if hasattr(channel, 'metadata') and 'Zxx' in getattr(channel, 'metadata', {}):
            return 'spectrogram'
        
        # Default to line wizard
        return 'line'
    
    def _create_marker_config_from_channel(self, channel: Channel) -> Dict[str, Any]:
        """Create a marker configuration dict from a channel"""
        return {
            'name': channel.legend_label or channel.ylabel or channel.channel_id,
            'marker_style': getattr(channel, 'marker', 'o'),
            'marker_size': getattr(channel, 'marker_size', 20),
            'marker_color': getattr(channel, 'color', '#1f77b4'),
            'marker_alpha': getattr(channel, 'alpha', 1.0),
            'edge_color': getattr(channel, 'edge_color', '#000000'),
            'edge_width': getattr(channel, 'edge_width', 1.0),
            'x_axis': getattr(channel, 'xaxis', 'x-bottom').replace('x-', ''),
            'z_order': getattr(channel, 'z_order', 0),
            'channel_id': channel.channel_id
        }
    
    def _on_channel_updated_from_wizard(self, channel_id: str):
        """Handle channel updates from line/spectrogram wizards"""
        self._log_message(f"Channel {channel_id} updated from wizard")
        self._schedule_plot_update()
    
    def _on_marker_updated_from_wizard(self, channel_id: str, marker_config: Dict[str, Any]):
        """Handle marker updates from marker wizard"""
        self._log_message(f"Marker {channel_id} updated from wizard")
        # Update channel properties from marker config if needed
        self._schedule_plot_update()
    
    def _on_spectrogram_updated_from_wizard(self, channel_id: str):
        """Handle spectrogram updates from spectrogram wizard"""
        self._log_message(f"Spectrogram {channel_id} updated from wizard")
        self._schedule_plot_update()
    
    def _on_channel_visibility_changed(self, channel_id: str, visible: bool):
        """Handle channel visibility changes"""
        if visible:
            self.visible_channels.add(channel_id)
        else:
            self.visible_channels.discard(channel_id)
        
        self._schedule_plot_update()
    
    def _schedule_plot_update(self):
        """Schedule a plot update with debouncing"""
        self.update_timer.start(self.update_delay)
    
    def _delayed_plot_update(self):
        """Perform the actual plot update"""
        try:
            self._update_plot()
            self._stats['last_update_time'] = datetime.now()
            self.plot_updated.emit()
        except Exception as e:
            self._log_message(f"Error updating plot: {str(e)}")
    
    def _force_plot_update(self):
        """Force immediate plot update"""
        self.update_timer.stop()
        self._delayed_plot_update()
    
    def _update_plot(self):
        """Update the plot with current channels - should be overridden by subclasses"""
        if not self.figure:
            return
        
        # Clear the figure
        self.figure.clear()
        
        # Get channels to plot
        channels = self._get_channels_to_plot()
        visible_channels = [ch for ch in channels if ch.channel_id in self.visible_channels]
        
        if not visible_channels:
            self.canvas.draw()
            return
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Plot each visible channel
        for channel in visible_channels:
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
        
        # Configure plot
        if self.plot_config['grid']:
            ax.grid(True, alpha=0.3)
        
        if self.plot_config['legend_show'] and len(visible_channels) > 0:
            ax.legend(loc=self.plot_config['legend_position'],
                     fontsize=self.plot_config['font_size'])
        
        # Apply tight layout
        if self.plot_config['tight_layout']:
            self.figure.tight_layout()
        
        # Update canvas
        self.canvas.draw()
        
        # Update stats
        self._stats['plots_created'] += 1
        self._stats['channels_plotted'] = len(visible_channels)
    
    def _clear_plot(self):
        """Clear the plot"""
        if self.figure:
            self.figure.clear()
            self.canvas.draw()
        self._log_message("Plot cleared")
    
    def _export_plot(self):
        """Export the current plot"""
        try:
            from PySide6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Plot", "", 
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*.*)"
            )
            
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                self._log_message(f"Plot exported to {filename}")
                self.plot_exported.emit(filename)
            
        except Exception as e:
            self._log_message(f"Error exporting plot: {str(e)}")
    
    def _show_plot_config(self):
        """Show plot configuration dialog"""
        try:
            from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Plot Configuration")
            dialog.setModal(True)
            layout = QFormLayout(dialog)
            
            # Grid checkbox
            grid_checkbox = QCheckBox()
            grid_checkbox.setChecked(self.plot_config['grid'])
            layout.addRow("Show Grid:", grid_checkbox)
            
            # Legend checkbox
            legend_checkbox = QCheckBox()
            legend_checkbox.setChecked(self.plot_config['legend_show'])
            layout.addRow("Show Legend:", legend_checkbox)
            
            # Legend position
            legend_pos_combo = QComboBox()
            positions = ['upper right', 'upper left', 'lower right', 'lower left', 'center']
            legend_pos_combo.addItems(positions)
            legend_pos_combo.setCurrentText(self.plot_config['legend_position'])
            layout.addRow("Legend Position:", legend_pos_combo)
            
            # Font size
            font_size_spin = QSpinBox()
            font_size_spin.setRange(6, 24)
            font_size_spin.setValue(self.plot_config['font_size'])
            layout.addRow("Font Size:", font_size_spin)
            
            # Line width
            line_width_spin = QDoubleSpinBox()
            line_width_spin.setRange(0.5, 5.0)
            line_width_spin.setSingleStep(0.1)
            line_width_spin.setValue(self.plot_config['line_width'])
            layout.addRow("Line Width:", line_width_spin)
            
            # Dialog buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            if dialog.exec() == QDialog.Accepted:
                # Update configuration
                self.plot_config['grid'] = grid_checkbox.isChecked()
                self.plot_config['legend_show'] = legend_checkbox.isChecked()
                self.plot_config['legend_position'] = legend_pos_combo.currentText()
                self.plot_config['font_size'] = font_size_spin.value()
                self.plot_config['line_width'] = line_width_spin.value()
                
                # Update plot
                self._force_plot_update()
                self._log_message("Plot configuration updated")
        
        except Exception as e:
            self._log_message(f"Error showing plot configuration: {str(e)}")
    
    def _log_message(self, message: str):
        """Log a message to the console"""
        if self.console_output:
            timestamp = time.strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
            self.console_output.append(formatted_message)
            print(f"[{self._get_wizard_type()}Wizard] {formatted_message}")
        
        self.state_changed.emit(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wizard statistics"""
        return {
            **self._stats,
            'visible_channels': len(self.visible_channels),
            'session_duration': time.time() - self._stats['session_start'],
            'wizard_type': self._get_wizard_type()
        }
    
    def get_plot_config(self) -> Dict[str, Any]:
        """Get current plot configuration"""
        return self.plot_config.copy()
    
    def set_plot_config(self, config: Dict[str, Any]):
        """Set plot configuration"""
        self.plot_config.update(config)
        self._force_plot_update()
    
    def add_channel(self, channel: Channel, visible: bool = True):
        """Add a channel to the wizard"""
        self.plotted_channels[channel.channel_id] = {
            'channel': channel,
            'added_time': datetime.now()
        }
        
        if visible:
            self.visible_channels.add(channel.channel_id)
        
        self._schedule_plot_update()
    
    def remove_channel(self, channel_id: str):
        """Remove a channel from the wizard"""
        self.plotted_channels.pop(channel_id, None)
        self.visible_channels.discard(channel_id)
        self._schedule_plot_update()
    
    def clear_all_channels(self):
        """Clear all channels"""
        self.plotted_channels.clear()
        self.visible_channels.clear()
        self._schedule_plot_update()
    
    def closeEvent(self, event):
        """Handle wizard closing"""
        self.wizard_closed.emit()
        super().closeEvent(event)


# Utility functions for plot wizard integration
def create_plot_wizard_factory(wizard_type: str):
    """Create a factory function for creating plot wizards of specific types"""
    
    def factory(file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        if wizard_type == 'process':
            from process_wizard_window import ProcessWizardWindow
            return ProcessWizardWindow(file_manager, channel_manager, signal_bus, parent)
        elif wizard_type == 'mixer':
            from signal_mixer_wizard_window import SignalMixerWizardWindow
            return SignalMixerWizardWindow(file_manager, channel_manager, signal_bus, parent)
        elif wizard_type == 'comparison':
            from comparison_wizard_window import ComparisonWizardWindow
            return ComparisonWizardWindow(file_manager, channel_manager, signal_bus, parent)
        elif wizard_type == 'plot':
            from plot_wizard_manager import PlotWizardManager
            return PlotWizardManager(file_manager, channel_manager, signal_bus, parent)
        else:
            raise ValueError(f"Unknown wizard type: {wizard_type}")
    
    return factory


def link_config_wizard_to_plot(plot_wizard: BasePlotWizard, config_wizard_type: str):
    """Link a configuration wizard to a plot wizard for automatic updates"""
    
    def open_config_wizard(channel_or_config):
        """Open the appropriate configuration wizard"""
        try:
            if config_wizard_type == 'line':
                from line_wizard_refactored import LineWizard
                wizard = LineWizard(channel_or_config, plot_wizard)
                wizard.config_updated.connect(lambda obj: plot_wizard._on_channel_updated_from_wizard(obj.channel_id))
                
            elif config_wizard_type == 'marker':
                from marker_wizard_refactored import MarkerWizard
                wizard = MarkerWizard(channel_or_config, plot_wizard)
                wizard.config_updated.connect(lambda obj: plot_wizard._on_marker_updated_from_wizard(obj.get('channel_id', ''), obj))
                
            elif config_wizard_type == 'spectrogram':
                from spectrogram_wizard_refactored import SpectrogramWizard
                wizard = SpectrogramWizard(channel_or_config, plot_wizard)
                wizard.config_updated.connect(lambda obj: plot_wizard._on_spectrogram_updated_from_wizard(obj.channel_id))
            
            wizard.exec()
            
        except Exception as e:
            plot_wizard._log_message(f"Error opening {config_wizard_type} wizard: {str(e)}")
    
    return open_config_wizard 