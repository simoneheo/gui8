from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QLineEdit, QPushButton, QListWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QTextEdit, QCheckBox, QFrame, QTabWidget, QRadioButton, QButtonGroup,
    QGroupBox, QSpinBox, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, QEvent, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import sys
import time
from typing import Optional, Dict, List

from base_plot_wizard import BasePlotWizard
from steps.process_registry import load_all_steps, ProcessRegistry
from process_wizard_manager import ProcessWizardManager
from file_manager import FileManager
from channel_manager import ChannelManager
from channel import SourceType, Channel
from steps.base_step import BaseStep
from plot_manager import StylePreviewWidget
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import zoom


class ProcessWizardRefactored(BasePlotWizard):
    """
    Refactored Process Wizard inheriting from BasePlotWizard
    Demonstrates consolidation of plotting functionality
    """
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        # Initialize specific state for process wizard
        self.input_ch = None
        self._adding_filter = False
        self.radio_button_group = None
        self.wizard_manager = None
        self.process_registry = None
        self.all_filters = []
        
        # UI components specific to process wizard
        self.file_selector = None
        self.channel_selector = None
        self.filter_list = None
        self.filter_search = None
        self.category_filter = None
        self.step_table = None
        self.add_filter_btn = None
        self.input_channel_combo = None
        self.input_channel_name = None
        
        # Plot-specific components for different tabs
        self.time_series_figure = None
        self.time_series_ax = None
        self.time_series_canvas = None
        
        self.spectrogram_figure = None
        self.spectrogram_ax = None
        self.spectrogram_canvas = None
        self.log_scale_checkbox = None
        self.downsample_checkbox = None
        
        self.bar_chart_figure = None
        self.bar_chart_ax = None
        self.bar_chart_canvas = None
        
        # Initialize base class
        super().__init__(file_manager, channel_manager, signal_bus, parent)
        
        # Set window properties
        self.setWindowTitle("Process File")
        self.setMinimumSize(1200, 800)
        
        # Initialize process-specific components
        self._initialize_process_components()
        
        # Update UI
        self._update_file_selector()
        self._update_channel_selector()
        
        self._log_message("Process wizard initialized successfully")
    
    def _get_wizard_type(self) -> str:
        """Get the wizard type name"""
        return "Process"
    
    def _initialize_process_components(self):
        """Initialize process-specific components"""
        try:
            # Load processing steps
            load_all_steps("steps")
            self.process_registry = ProcessRegistry
            self.all_filters = self.process_registry.all_steps()
            
            # Create wizard manager
            self.wizard_manager = ProcessWizardManager(
                ui=self,
                registry=self.process_registry,
                channel_lookup=self.get_active_channel_info
            )
            
            # Populate UI components
            if self.filter_list:
                self.filter_list.addItems(self.all_filters)
                self.filter_list.itemClicked.connect(self.wizard_manager._on_filter_selected)
            
            if self.category_filter:
                self._populate_category_filter()
            
        except Exception as e:
            self._log_message(f"Error initializing process components: {str(e)}")
    
    def _create_main_content(self, layout: QVBoxLayout):
        """Create the main content area specific to process wizard"""
        # Create horizontal splitter for left panel columns
        left_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(left_splitter)
        
        # Left column - Filters
        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        self._create_filters_section(left_column_layout)
        
        # Right column - Controls
        right_column = QWidget()
        right_column_layout = QVBoxLayout(right_column)
        self._create_file_selection_section(right_column_layout)
        self._create_input_channel_section(right_column_layout)
        self._create_console_section(right_column_layout)
        
        # Add columns to splitter
        left_splitter.addWidget(left_column)
        left_splitter.addWidget(right_column)
        left_splitter.setSizes([350, 500])
        
        # Add step table below
        self._create_step_table_section(layout)
    
    def _create_filters_section(self, layout: QVBoxLayout):
        """Create the filters section"""
        group = QGroupBox("Filters")
        group_layout = QVBoxLayout(group)
        
        # Filter search
        self.filter_search = QLineEdit()
        self.filter_search.setPlaceholderText("Search filters...")
        self.filter_search.textChanged.connect(self._on_filter_search)
        
        # Category filter
        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories")
        self.category_filter.currentTextChanged.connect(self._on_category_filter_changed)
        
        # Filter list
        self.filter_list = QListWidget()
        
        group_layout.addWidget(self.filter_search)
        group_layout.addWidget(self.category_filter)
        group_layout.addWidget(self.filter_list)
        
        layout.addWidget(group)
    
    def _create_file_selection_section(self, layout: QVBoxLayout):
        """Create the file selection section"""
        group = QGroupBox("File Selection")
        group_layout = QVBoxLayout(group)
        
        # File selector
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.file_selector = QComboBox()
        self.file_selector.currentIndexChanged.connect(self._on_file_selected)
        file_layout.addWidget(self.file_selector)
        
        # Channel selector
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_selector = QComboBox()
        self.channel_selector.currentIndexChanged.connect(self._on_channel_selected)
        channel_layout.addWidget(self.channel_selector)
        
        group_layout.addLayout(file_layout)
        group_layout.addLayout(channel_layout)
        
        layout.addWidget(group)
    
    def _create_input_channel_section(self, layout: QVBoxLayout):
        """Create the input channel section"""
        group = QGroupBox("Input Channel")
        group_layout = QVBoxLayout(group)
        
        # Input channel combo
        self.input_channel_combo = QComboBox()
        self.input_channel_combo.currentIndexChanged.connect(self._on_input_channel_changed)
        
        # Input channel name display
        self.input_channel_name = QLabel("No channel selected")
        
        group_layout.addWidget(self.input_channel_combo)
        group_layout.addWidget(self.input_channel_name)
        
        layout.addWidget(group)
    
    def _create_console_section(self, layout: QVBoxLayout):
        """Create the console section"""
        group = QGroupBox("Console")
        group_layout = QVBoxLayout(group)
        
        # Add Filter button
        self.add_filter_btn = QPushButton("Add Filter")
        self.add_filter_btn.clicked.connect(self._on_add_filter)
        
        group_layout.addWidget(self.add_filter_btn)
        layout.addWidget(group)
    
    def _create_step_table_section(self, layout: QVBoxLayout):
        """Create the step table section"""
        # Step table
        self.step_table = QTableWidget(0, 6)
        self.step_table.setHorizontalHeaderLabels(["Show", "Style", "Channel Name", "Shape", "fs (Hz)", "Actions"])
        self.step_table.verticalHeader().setVisible(False)
        
        # Configure column properties
        header = self.step_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)     # Show
        header.setSectionResizeMode(1, QHeaderView.Fixed)     # Style
        header.setSectionResizeMode(2, QHeaderView.Stretch)   # Channel Name
        header.setSectionResizeMode(3, QHeaderView.Fixed)     # Shape
        header.setSectionResizeMode(4, QHeaderView.Fixed)     # fs (Hz)
        header.setSectionResizeMode(5, QHeaderView.Fixed)     # Actions
        
        # Set column widths
        self.step_table.setColumnWidth(0, 60)   # Show
        self.step_table.setColumnWidth(1, 80)   # Style
        self.step_table.setColumnWidth(3, 80)   # Shape
        self.step_table.setColumnWidth(4, 100)  # fs (Hz)
        self.step_table.setColumnWidth(5, 180)  # Actions
        
        self.step_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.step_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.step_table.setMaximumHeight(150)
        
        layout.addWidget(self.step_table)
    
    def _create_plot_tab(self) -> QWidget:
        """Override to create tabbed plot interface for process wizard"""
        tab_widget = QTabWidget()
        
        # Time Series tab
        time_series_tab = self._create_time_series_tab()
        tab_widget.addTab(time_series_tab, "Time Series")
        
        # Spectrogram tab
        spectrogram_tab = self._create_spectrogram_tab()
        tab_widget.addTab(spectrogram_tab, "Spectrogram")
        
        # Bar Chart tab
        bar_chart_tab = self._create_bar_chart_tab()
        tab_widget.addTab(bar_chart_tab, "Bar Chart")
        
        # Connect tab changes
        tab_widget.currentChanged.connect(self._on_tab_changed)
        
        return tab_widget
    
    def _create_time_series_tab(self) -> QWidget:
        """Create time series plot tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create figure and canvas
        self.time_series_figure = plt.figure(figsize=(8, 6), dpi=100)
        self.time_series_ax = self.time_series_figure.add_subplot(111)
        self.time_series_canvas = FigureCanvas(self.time_series_figure)
        toolbar = NavigationToolbar(self.time_series_canvas, tab)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.time_series_canvas)
        
        return tab
    
    def _create_spectrogram_tab(self) -> QWidget:
        """Create spectrogram plot tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.log_scale_checkbox = QCheckBox("Logarithmic Scaling")
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.stateChanged.connect(self._on_spectrogram_settings_changed)
        
        self.downsample_checkbox = QCheckBox("Downsample for Performance")
        self.downsample_checkbox.setChecked(True)
        self.downsample_checkbox.stateChanged.connect(self._on_spectrogram_settings_changed)
        
        controls_layout.addWidget(self.log_scale_checkbox)
        controls_layout.addWidget(self.downsample_checkbox)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Create figure and canvas
        self.spectrogram_figure = plt.figure(figsize=(8, 6), dpi=100)
        self.spectrogram_ax = self.spectrogram_figure.add_subplot(111)
        self.spectrogram_figure._colorbar_list = []
        self.spectrogram_canvas = FigureCanvas(self.spectrogram_figure)
        toolbar = NavigationToolbar(self.spectrogram_canvas, tab)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.spectrogram_canvas)
        
        return tab
    
    def _create_bar_chart_tab(self) -> QWidget:
        """Create bar chart plot tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create figure and canvas
        self.bar_chart_figure = plt.figure(figsize=(8, 6), dpi=100)
        self.bar_chart_ax = self.bar_chart_figure.add_subplot(111)
        self.bar_chart_canvas = FigureCanvas(self.bar_chart_figure)
        toolbar = NavigationToolbar(self.bar_chart_canvas, tab)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.bar_chart_canvas)
        
        return tab
    
    def _get_channels_to_plot(self) -> List[Channel]:
        """Get channels to plot based on current selection and tab"""
        active_channel = self.get_active_channel_info()
        if not active_channel:
            return []
        
        # Get lineage channels
        lineage_dict = self.channel_manager.get_channels_by_lineage(active_channel.lineage_id)
        all_lineage_channels = []
        all_lineage_channels.extend(lineage_dict.get('parents', []))
        all_lineage_channels.extend(lineage_dict.get('children', []))
        all_lineage_channels.extend(lineage_dict.get('siblings', []))
        
        # Filter by file and sort
        lineage = [ch for ch in all_lineage_channels if ch.file_id == active_channel.file_id]
        lineage.sort(key=lambda ch: ch.step)
        
        return lineage
    
    def _update_plot(self):
        """Override to handle multi-tab plotting"""
        channels = self._get_channels_to_plot()
        if not channels:
            return
        
        # Update the appropriate plot based on current tab
        if hasattr(self, 'plot_tabs') and self.plot_tabs:
            current_tab_index = self.plot_tabs.indexOf(self.plot_tabs.currentWidget())
            
            if current_tab_index == 0:  # Time Series
                self._update_time_series_plot(channels)
            elif current_tab_index == 1:  # Spectrogram
                self._update_spectrogram_plot(channels)
            elif current_tab_index == 2:  # Bar Chart
                self._update_bar_chart_plot(channels)
    
    def _update_time_series_plot(self, channels: List[Channel]):
        """Update time series plot"""
        if not self.time_series_ax:
            return
        
        # Filter for time series channels
        time_series_channels = [ch for ch in channels if "time-series" in ch.tags or ch.step == 0]
        
        self.time_series_ax.clear()
        
        for ch in time_series_channels:
            if ch.show and hasattr(ch, 'xdata') and hasattr(ch, 'ydata'):
                if ch.xdata is not None and ch.ydata is not None:
                    color = getattr(ch, 'color', '#1f77b4')
                    style = getattr(ch, 'style', '-')
                    marker = getattr(ch, 'marker', None)
                    
                    if style == "None":
                        style = None
                    if marker == "None":
                        marker = None
                    
                    self.time_series_ax.plot(
                        ch.xdata, ch.ydata,
                        color=color,
                        linestyle=style,
                        marker=marker,
                        label=ch.legend_label
                    )
        
        if len(time_series_channels) > 0:
            self.time_series_ax.set_title(f"File: {time_series_channels[0].filename}")
        
        self.time_series_canvas.draw()
    
    def _update_spectrogram_plot(self, channels: List[Channel]):
        """Update spectrogram plot"""
        if not self.spectrogram_ax:
            return
        
        # Filter for spectrogram channels
        spectrogram_channels = [ch for ch in channels if "spectrogram" in ch.tags]
        
        self.spectrogram_figure.clf()
        self.spectrogram_ax = self.spectrogram_figure.add_subplot(111)
        
        for channel in spectrogram_channels:
            if channel.show and hasattr(channel, 'metadata') and 'Zxx' in channel.metadata:
                Zxx = channel.metadata['Zxx'].copy()
                colormap = channel.metadata.get('colormap', 'viridis')
                
                # Apply scaling and downsampling based on checkboxes
                if self.log_scale_checkbox and self.log_scale_checkbox.isChecked():
                    epsilon = np.finfo(float).eps
                    Zxx_display = 10 * np.log10(Zxx + epsilon)
                else:
                    Zxx_display = Zxx
                
                if self.downsample_checkbox and self.downsample_checkbox.isChecked():
                    # Downsample logic (simplified)
                    max_size = (200, 1000)
                    if Zxx_display.shape[0] > max_size[0] or Zxx_display.shape[1] > max_size[1]:
                        freq_zoom = min(1.0, max_size[0] / Zxx_display.shape[0])
                        time_zoom = min(1.0, max_size[1] / Zxx_display.shape[1])
                        Zxx_display = zoom(Zxx_display, (freq_zoom, time_zoom), order=1)
                
                # Plot spectrogram
                vmin, vmax = np.percentile(Zxx_display[np.isfinite(Zxx_display)], [5, 95])
                im = self.spectrogram_ax.pcolormesh(
                    channel.xdata, channel.ydata, Zxx_display,
                    shading='gouraud', cmap=colormap, vmin=vmin, vmax=vmax
                )
                
                # Add colorbar
                cbar = self.spectrogram_figure.colorbar(im, ax=self.spectrogram_ax, 
                                                     orientation='horizontal', pad=0.05)
                cbar.set_label("Power")
                
                self.spectrogram_ax.set_ylabel(channel.ylabel)
                self.spectrogram_ax.set_xlabel(channel.xlabel)
                self.spectrogram_ax.set_yscale('log')
        
        self.spectrogram_canvas.draw()
    
    def _update_bar_chart_plot(self, channels: List[Channel]):
        """Update bar chart plot"""
        if not self.bar_chart_ax:
            return
        
        # Filter for bar chart channels
        bar_chart_channels = [ch for ch in channels if "bar-chart" in ch.tags]
        
        self.bar_chart_ax.clear()
        
        for ch in bar_chart_channels:
            if ch.show and hasattr(ch, 'xdata') and hasattr(ch, 'ydata'):
                if ch.xdata is not None and ch.ydata is not None:
                    color = getattr(ch, 'color', '#1f77b4')
                    self.bar_chart_ax.bar(ch.xdata, ch.ydata, color=color, label=ch.legend_label)
        
        if len(bar_chart_channels) > 0:
            self.bar_chart_ax.set_title(f"File: {bar_chart_channels[0].filename}")
            self.bar_chart_ax.legend()
        
        self.bar_chart_canvas.draw()
    
    # Process wizard specific methods
    def get_active_channel_info(self):
        """Get the currently selected channel"""
        if self.channel_selector:
            return self.channel_selector.currentData()
        return None
    
    def _update_file_selector(self):
        """Update file selector dropdown"""
        if not self.file_manager or not self.file_selector:
            return
        
        current_file_id = self.file_selector.currentData()
        self.file_selector.clear()
        
        all_files = self.file_manager.get_all_files()
        if not all_files:
            return
        
        from file import FileStatus
        parsed_files = [f for f in all_files if f.state.status in [FileStatus.PARSED, FileStatus.PROCESSED]]
        
        for file_info in parsed_files:
            display_name = f"{file_info.filename} ({file_info.state.status.value})"
            self.file_selector.addItem(display_name, file_info.file_id)
        
        # Restore selection
        if current_file_id:
            for i in range(self.file_selector.count()):
                if self.file_selector.itemData(i) == current_file_id:
                    self.file_selector.setCurrentIndex(i)
                    break
    
    def _update_channel_selector(self):
        """Update channel selector dropdown"""
        if not self.channel_manager or not self.channel_selector:
            return
        
        current_channel = self.get_active_channel_info()
        self.channel_selector.clear()
        
        selected_file_id = self.file_selector.currentData() if self.file_selector else None
        if not selected_file_id:
            return
        
        file_channels = self.channel_manager.get_channels_by_file(selected_file_id)
        raw_channels = [ch for ch in file_channels if ch.type == SourceType.RAW]
        
        for ch in raw_channels:
            self.channel_selector.addItem(ch.legend_label, ch)
        
        # Restore selection
        if current_channel:
            for i in range(self.channel_selector.count()):
                if self.channel_selector.itemData(i).channel_id == current_channel.channel_id:
                    self.channel_selector.setCurrentIndex(i)
                    break
    
    def _update_step_table(self):
        """Update the step table with current lineage"""
        if not self.step_table:
            return
        
        channels = self._get_channels_to_plot()
        
        self.step_table.setRowCount(len(channels))
        
        for row, channel in enumerate(channels):
            # Show checkbox
            show_checkbox = QCheckBox()
            show_checkbox.setChecked(channel.show)
            show_checkbox.toggled.connect(lambda checked, ch=channel: self._on_show_changed(ch, checked))
            self.step_table.setCellWidget(row, 0, show_checkbox)
            
            # Style preview
            style_widget = StylePreviewWidget(
                color=getattr(channel, 'color', '#1f77b4'),
                style=getattr(channel, 'style', '-'),
                marker=getattr(channel, 'marker', None)
            )
            self.step_table.setCellWidget(row, 1, style_widget)
            
            # Channel name
            name_item = QTableWidgetItem(channel.legend_label or channel.ylabel)
            self.step_table.setItem(row, 2, name_item)
            
            # Shape
            shape_text = f"{len(channel.ydata)}" if hasattr(channel, 'ydata') and channel.ydata is not None else "N/A"
            shape_item = QTableWidgetItem(shape_text)
            self.step_table.setItem(row, 3, shape_item)
            
            # Sampling rate
            fs_text = f"{channel.fs:.1f}" if hasattr(channel, 'fs') and channel.fs else "N/A"
            fs_item = QTableWidgetItem(fs_text)
            self.step_table.setItem(row, 4, fs_item)
            
            # Actions
            actions_btn = QPushButton("⚙")
            actions_btn.clicked.connect(lambda: self._handle_gear_button_clicked(channel.channel_id))
            self.step_table.setCellWidget(row, 5, actions_btn)
    
    # Event handlers
    def _on_file_selected(self, index):
        """Handle file selection change"""
        self._update_channel_selector()
        self._update_step_table()
        self._schedule_plot_update()
    
    def _on_channel_selected(self, index):
        """Handle channel selection change"""
        self._update_step_table()
        self._schedule_plot_update()
    
    def _on_show_changed(self, channel, state):
        """Handle show/hide checkbox change"""
        channel.show = state
        if state:
            self.visible_channels.add(channel.channel_id)
        else:
            self.visible_channels.discard(channel.channel_id)
        self._schedule_plot_update()
    
    def _on_add_filter(self):
        """Handle add filter button click"""
        if self.wizard_manager:
            self.wizard_manager._on_add_filter()
        self._update_step_table()
        self._schedule_plot_update()
    
    def _on_tab_changed(self, index):
        """Handle tab change"""
        self._schedule_plot_update()
    
    def _on_spectrogram_settings_changed(self):
        """Handle spectrogram settings change"""
        self._schedule_plot_update()
    
    def _on_input_channel_changed(self, index):
        """Handle input channel change"""
        if self.input_channel_combo:
            channel = self.input_channel_combo.currentData()
            if channel and self.input_channel_name:
                self.input_channel_name.setText(channel.legend_label or channel.ylabel)
    
    def _handle_gear_button_clicked(self, channel_id: str):
        """Handle gear button click to open configuration wizard"""
        # Find the channel
        channels = self._get_channels_to_plot()
        channel = next((ch for ch in channels if ch.channel_id == channel_id), None)
        
        if channel:
            self._open_channel_config(channel)
    
    def _populate_category_filter(self):
        """Populate category filter dropdown"""
        if not self.category_filter or not self.process_registry:
            return
        
        categories = set()
        for step_name in self.all_filters:
            step_class = self.process_registry.get(step_name)
            if step_class and hasattr(step_class, 'category'):
                categories.add(step_class.category)
        
        self.category_filter.clear()
        self.category_filter.addItem("All Categories")
        for category in sorted(categories):
            self.category_filter.addItem(category)
    
    def _on_category_filter_changed(self, category_text):
        """Handle category filter change"""
        self._apply_filters()
    
    def _on_filter_search(self, text):
        """Handle filter search text change"""
        self._apply_filters()
    
    def _apply_filters(self):
        """Apply current filters to the filter list"""
        if not self.filter_list:
            return
        
        search_text = self.filter_search.text().lower() if self.filter_search else ""
        category = self.category_filter.currentText() if self.category_filter else "All Categories"
        
        self.filter_list.clear()
        
        for step_name in self.all_filters:
            # Apply search filter
            if search_text and search_text not in step_name.lower():
                continue
            
            # Apply category filter
            if category != "All Categories":
                step_class = self.process_registry.get(step_name)
                if not step_class or getattr(step_class, 'category', '') != category:
                    continue
            
            self.filter_list.addItem(step_name) 